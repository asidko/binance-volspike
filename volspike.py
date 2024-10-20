import asyncio
import sys
from statistics import median, mean

import aiohttp
import argparse
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
from datetime import datetime
from dataclasses import dataclass
import re

# Initialize Rich Console
console = Console()

# Binance Futures API Constants
EXCHANGE_INFO_URL = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
KLINES_URL = 'https://fapi.binance.com/fapi/v1/klines'

# Asynchronous semaphore to limit concurrent requests
MAX_CONCURRENCY = 20
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)

# Type Definitions
KlineData = List[List[Any]]


@dataclass
class SymbolAnalysisResult:
    symbol: str
    change_percent: float
    min_passed: int = 0
    is_price_spike: bool = False

BINANCE_INTERVALS_IN_MINUTES = [1, 3, 5, 15, 30, 60, 120, 240, 360, 480, 720, 1440, 10080, 43200]

def convert_interval_to_minutes(interval: str) -> int:
    multipliers = {'m': 1, 'h': 60, 'd': 1440, 'w': 10080, 'M': 43200, 'y': 525600}
    return int(interval[:-1]) * multipliers.get(interval[-1], 0)

def make_more_human_readable_interval_label(label: str) -> str:
    transitions = {'m': ('h', 60), 'h': ('d', 24), 'd': ('M', 30)}
    while label[-1] in transitions:
        value, unit = int(label[:-1]), label[-1]
        new_unit, divisor = transitions[unit]
        if value % divisor == 0:
            label = f"{value // divisor}{new_unit}"
        else:
            break
    return label

def round_to_available_interval_minutes(interval_minutes: int) -> int:
    return min(BINANCE_INTERVALS_IN_MINUTES, key=lambda x: abs(x - interval_minutes))

def normalize_timeframe_label(label: str) -> str:
    return make_more_human_readable_interval_label(
        str(round_to_available_interval_minutes(convert_interval_to_minutes(label))) + 'm')

def parse_percentage(pct_str: str) -> float:
    try:
        return float(pct_str.strip('%'))
    except ValueError:
        console.print("[red]Invalid percentage format. Using default 2%.[/red]")
        return 2.0


def parse_timeframe(timeframe: str) -> int:
    match = re.match(r'^(\d+)([mhd])$', timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    value, unit = match.groups()
    multipliers = {'m': 1, 'h': 60, 'd': 1440}  # m -> 1, h -> 60, d -> 1440 (24 * 60)
    return int(value) * multipliers[unit]


def get_small_interval(timeframe: str) -> str:
    total_minutes = parse_timeframe(timeframe)
    if total_minutes <= 60:  # <= 1 hour
        return '1m'
    elif total_minutes <= 240:  # <= 4 hours
        return '15m'
    elif total_minutes <= 1440:  # <= 1 day
        return '1h'
    else:  # > 1 day
        return '4h'


def calculate_required_candles(total_time: str, candle_interval: str) -> int:
    total_minutes = parse_timeframe(total_time)
    candle_minutes = parse_timeframe(candle_interval)
    return max((total_minutes // candle_minutes) + 1, 1)


async def fetch_json(session: aiohttp.ClientSession, url: str, params: Dict[str, Any]) -> Any:
    try:
        async with SEMAPHORE:
            async with session.get(url, params=params, ssl=False, timeout=10) as response:
                if response.status != 200:
                    return None
                return await response.json()
    except Exception:
        return None


async def get_usdt_symbols(session: aiohttp.ClientSession) -> List[str]:
    data = await fetch_json(session, EXCHANGE_INFO_URL, {})
    if data is None:
        return []
    symbols = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
    return symbols


def calculate_avg_max(candles: KlineData, ratio_to_pick: float) -> float:
    top_x = max(int(len(candles) * ratio_to_pick), 1)
    max_values = [float(candle[2]) for candle in candles]
    top_max = sorted(max_values, reverse=True)[:top_x]
    return sum(top_max) / len(top_max) if top_max else 0.0


def trimmed_median(prices, trim_percent=5):
    trim_count = int(len(prices) * trim_percent / 100)
    return median(sorted(prices)[trim_count: -trim_count or None])


def calculate_avg_min(candles: KlineData, ration_to_pick: float) -> float:
    bottom_x = max(int(len(candles) * ration_to_pick), 1)
    min_values = [float(candle[3]) for candle in candles]
    bottom_min = sorted(min_values)[:bottom_x]
    return sum(bottom_min) / len(bottom_min) if bottom_min else 0.0


async def analyze_symbol(
        session: aiohttp.ClientSession,
        symbol: str,
        args: argparse.Namespace
) -> Optional[SymbolAnalysisResult]:
    try:
        interval_limit = calculate_required_candles(args.range, args.interval)
        current_data = await fetch_json(session, KLINES_URL, {
            'symbol': symbol,
            'interval': f'{args.interval}',
            'limit': f"{interval_limit}"
        })

        if not current_data:
            return None

        if len(current_data) < 3:
            return None

        # Ignore last item because it's not complete (volume and price is still forming)
        current_data = current_data[:-1]

        volumes = [float(candle[5]) for candle in current_data]
        if len(volumes) < 4:
            return None

        # Take last 3 volumes for comparison
        volumes_last = volumes[-3:]
        volume_median = median(volumes)
        volume_last_min = min(volumes_last)
        change_percent = ((volume_last_min - volume_median) / volume_median) * 100
        change_percent_min = ((volume_last_min - volume_median) / volume_median) * 100

        is_valid_spike = volume_last_min > volume_median
        is_valid_spike = is_valid_spike and change_percent_min >= args.threshold

        if not is_valid_spike:
            return None


        min_passed = 0
        is_price_spike=False
        sustained_growth_index = find_sustained_growth(volumes, threshold=args.threshold/100, consecutive=3)
        if sustained_growth_index != -1:
            # count items from this index to the end
            count_items_after = len(volumes) - sustained_growth_index
            min_passed = count_items_after * convert_interval_to_minutes(args.interval)
            # average diff between max and min of prices
            avg_minmax_before_list = [abs(float(candle[2]) - float(candle[3])) for candle in current_data[:sustained_growth_index]]
            avg_minmax_after_list = [abs(float(candle[2]) - float(candle[3])) for candle in current_data[sustained_growth_index:]]
            avg_minmax_before,avg_minmax_after = mean(avg_minmax_before_list), mean(avg_minmax_after_list)
            price_change_threshold_ratio = 0.25
            is_price_spike = avg_minmax_after > avg_minmax_before * price_change_threshold_ratio


        return SymbolAnalysisResult(
            symbol=symbol,
            change_percent=change_percent,
            min_passed=min_passed,
            is_price_spike=is_price_spike
        )
    except Exception:
        return None


def find_sustained_growth(data, threshold=0.5, consecutive=2):
    count = 0  # Keeps track of consecutive growth periods
    for i in range(1, len(data)):
        # Calculate relative growth between consecutive items
        relative_change = (data[i] - data[i - 1]) / data[i - 1]

        # Check if the growth exceeds the threshold
        if relative_change > threshold:
            count += 1  # Increment if there's a growth above the threshold
            # If sustained growth occurs for the required number of consecutive periods
            if count >= consecutive:
                return i - consecutive + 1  # Return the start of the sustained growth
        else:
            count = 0  # Reset count if growth is interrupted
    return -1  # Return -1 if no sustained growth is found

def create_table(results: List[SymbolAnalysisResult], last_updated: str, args: argparse.Namespace) -> Table:
    top_count = args.count
    table = Table(title=f"Binance Top {top_count} Boosters and Losers\nUpdated: {last_updated}")
    table.add_column(f"Symbol Futures", style="cyan", no_wrap=True)
    table.add_column(f"Volume Change", style="magenta", no_wrap=True)
    table.add_column(f"Time since", style="magenta", no_wrap=True)

    for res in results:
        symbol = res.symbol
        percent = round(res.change_percent)
        min_passed = res.min_passed
        time_passed = minutes_to_human_readable(min_passed)
        is_price_spike = res.is_price_spike

        symbol_display = f"[bright_cyan]{symbol}[/bright_cyan]" if is_price_spike else f"{symbol}"
        parent_display = f"[green]{percent}[/green]%" if percent > 0 else f"[red]{percent}[/red]%"
        time_passed_display = "-" if min_passed <= 0 else f"[green]{time_passed}[green]" if min_passed < 60 else f"[red]{time_passed}[/red]"
        table.add_row(symbol_display, parent_display, time_passed_display)
    return table


def minutes_to_human_readable(minutes):
    d, minutes = divmod(minutes, 1440)
    h, m = divmod(minutes, 60)
    return f"{d}d " * (d > 0) + f"{h}h " * (h > 0) + f"{m}m" * (m > 0) or "0m"

async def main(args: argparse.Namespace):
    range = normalize_timeframe_label(args.range)
    console.print(f"\nSearching for symbols. Analysing volume on [yellow]{args.interval}[/yellow] intervals of [yellow]{range}[/yellow] range. Looking for [magenta]{args.threshold}%[/magenta] spikes!\n")

    async with aiohttp.ClientSession() as session:
        # Fetching symbol list
        symbols = await get_usdt_symbols(session)
        if not symbols:
            console.print("[red]No available symbols for analysis.[/red]")
            return

        while True:
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tasks = [
                analyze_symbol(session, symbol, args)
                for symbol in symbols
            ]

            results = []

            with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>1.0f}%",
                    TimeRemainingColumn(),
                    console=console
            ) as progress:
                task = progress.add_task(f"Analyzing {len(symbols)} symbols...", total=len(tasks))
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    if result:
                        results.append(result)
                    progress.advance(task)


            # TOP N Limit for each category
            top_count = args.count

            # Sort results by change_percent
            final_results = sorted(results, key=lambda x: x.change_percent, reverse=True)
            # Trim the list to top N
            final_results = final_results[:top_count]

            # Create table
            table = create_table(final_results, start_time, args)
            console.print(table)

            if not args.watch:
                break

            await asyncio.sleep(args.wait)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze volume changes of USDT coins on Binance Futures.')
    parser.add_argument('--interval', type=str, default="15m", help='Timeframe for volume analysis (e.g. 15m, 1h, 4h, 1d)')
    parser.add_argument('--range', type=str, default="6h", help='Time range for volume analysis (e.g. 4h, 1d, 3d)')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--threshold', type=str, default="50%", help='Volume change threshold, by default filter everything without 50% spikes')
    parser.add_argument('--wait', type=int, default=30, help='Interval for continuous monitoring mode')
    parser.add_argument('--count', type=int, default=12, help='Number of top symbols to display')
    args = parser.parse_args()

    args.max_concurrency = MAX_CONCURRENCY
    args.interval = normalize_timeframe_label(args.interval)
    args.threshold = parse_percentage(args.threshold)

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        console.print("[red]Program terminated by user.[/red]")

# 🗽 Volume Spike Detector 💥

Binance Futures symbol lookup tool

[👉 Ukrainian version](README_uk.md)

## Usage examples

🎩Example. Find symbols that got boos in volume by **120%** in the last **4 hours**

```bash
 python volspike.py --interval=5m --range=4h --threshold=120%  
```

<img width="841" alt="image" src="https://github.com/user-attachments/assets/6a3f23c3-d9c2-4b84-8f8d-bd584f379afc">

🎩Example. Find 300% and more volume spikes in the last 30 minutes

```bash
python volspike.py --interval=3m --range=30m --threshold=300%
```

<img width="826" alt="image" src="https://github.com/user-attachments/assets/acee4e94-df54-4b57-b72f-cb7964b7e48a">

Downtrend can also be caused by a sudden sell volume boost.
![image](https://github.com/user-attachments/assets/17d5fb8f-4fe6-42ce-8fad-fde8f0c12421)

🎩Example. Find volume spikes in the last 15 minutes (at least 300% increase)

```bash
python volspike.py --interval=1m --range=15m --threshold=300%
```

Good opportunity to buy
<img width="816" alt="image" src="https://github.com/user-attachments/assets/f698bad6-a8f5-45bc-a5c7-a8c8f348d31b">


## Installation

1. Make sure python is installed on your machine

Example of installation on Ubuntu Linux:
```bash
sudo apt-get update -y && sudo apt-get install -y python3 python3-pip python-is-python3
```

Example of installation on Android (Termux):
```bash
pkg update && pkg upgrade -y && pkg install -y python
```

2. Download the script to your machine<br>

```bash
# Download the script form the repository
curl -O https://raw.githubusercontent.com/asidko/binance-volspike/main/volspike.py
# ☝️ Repeat this command later if you want to update the script to a newer version
```

3. Install required python packages

```bash
pip install aiohttp rich
```

4. Run the script (check the usage examples above)

```bash
python volspike.py --interval=5m --range=3h --threshold=300%
```

## Special params

### --help

Example: `python volspike.py --help`

See all available options

### --watch

Example: `python volspike.py --interval=3m --range=30m --threshold=300% --watch`

Automatically request new data every 30 seconds and show it

You can change the interval by passing `--wait=300` (in seconds) to request data every 5 minutes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
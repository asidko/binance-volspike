# 🗽 Volume Spike Detector 💥

Інструмент для пошуку сплесків об'єму на Binance Futures

[👉 Англійська версія](README.md)

## Приклади використання

🎩Приклад. Знайти символи, об'єм яких зріс на **120%** за останні **4 години**

```bash
python volspike.py --interval=5m --range=4h --threshold=120%  
```

<img width="841" alt="image" src="https://github.com/user-attachments/assets/6a3f23c3-d9c2-4b84-8f8d-bd584f379afc">

🎩Приклад. Знайти сплески об'єму на 300% і більше за останні 30 хвилин

```bash
python volspike.py --interval=3m --range=30m --threshold=300%
```

<img width="826" alt="image" src="https://github.com/user-attachments/assets/acee4e94-df54-4b57-b72f-cb7964b7e48a">

Зниження тренду може також бути спричинене різким збільшенням об'єму продажів.
![image](https://github.com/user-attachments/assets/17d5fb8f-4fe6-42ce-8fad-fde8f0c12421)

🎩Приклад. Знайти сплески об'єму за останні 15 хвилин (зростання мінімум на 300%)

```bash
python volspike.py --interval=1m --range=15m --threshold=300%
```

Це може бути гарною можливістю для покупки
<img width="816" alt="image" src="https://github.com/user-attachments/assets/f698bad6-a8f5-45bc-a5c7-a8c8f348d31b">

## Встановлення

1. Переконайтеся, що у вас встановлено Python

Приклад встановлення на Ubuntu Linux:
```bash
sudo apt-get update -y && sudo apt-get install -y python3 python3-pip python-is-python3
```

Приклад встановлення на Android (Termux):
```bash
pkg update && pkg upgrade -y && pkg install -y python
```

2. Завантажте скрипт на свій пристрій<br>

```bash
# Завантажте скрипт з репозиторію
curl -O https://raw.githubusercontent.com/asidko/binance-volspike/main/volspike.py
# ☝️ Повторіть цю команду пізніше, щоб оновити скрипт до останньої версії
```

3. Встановіть необхідні пакети Python

```bash
pip install aiohttp rich
```

4. Запустіть скрипт (див. приклади використання вище)

```bash
python volspike.py --interval=5m --range=3h --threshold=300%
```

## Спеціальні параметри

### --help

Приклад: `python volspike.py --help`

Перегляньте всі доступні опції

### --watch

Приклад: `python volspike.py --interval=3m --range=30m --threshold=300% --watch`

Автоматично запитує нові дані кожні 30 секунд і показує їх

Щоб змінити інтервал, можна додати параметр `--wait=300` (у секундах) для запитування даних кожні 5 хвилин.

## Ліцензія

Проєкт ліцензований за ліцензією MIT — див. файл [LICENSE](LICENSE) для деталей.

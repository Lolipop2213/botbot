# data_generation/label_trades.py
"""
Парсинг логов бота для извлечения реальных сделок.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import pandas as pd

# Импортируем логгер
from utils.logger import setup_logger

logger = setup_logger("label_trades")

LOG_FILE   = 'bot_log.log' # Стандартное имя файла лога бота
OUTPUT_CSV = 'labeled_trades.csv'

# Игнорируем «восстановленные» позиции
pattern_restore     = re.compile(r".*Восстановлена сделка.*")

# Открытие (две формы логов)
pattern_open_watch   = re.compile(
    r".*\[(?P<symbol>[^\]]+)\].*Отслеживаю\s+(?P<side>LONG|SHORT).*Вход:\s*(?P<price>\d+\.\d+)"
)
pattern_open_confirm = re.compile(
    r".*\[(?P<symbol>[^\]]+)\].*ОТКРЫТА\s+(?P<side>LONG|SHORT).*?@\s*(?P<price>\d+\.\d+)"
)

# Полные закрытия по TP1/TP2/TP3
pattern_tp1_close = re.compile(
    r".*\[(?P<symbol>[^\]]+)\].*Take Profit\s*1\s+достигнут.*?@\s*(?P<price>\d+\.\d+)"
)
pattern_tp2_close = re.compile(
    r".*\[(?P<symbol>[^\]]+)\].*Take Profit\s*2\s+достигнут.*?@\s*(?P<price>\d+\.\d+)"
)
pattern_tp3_close = re.compile(
    r".*\[(?P<symbol>[^\]]+)\].*СДЕЛКА\s+ЗАКРЫТА\s+по\s+Take\s+Profit\s*3\s*@\s*(?P<price>\d+\.\d+)"
)

# Полное SL‐закрытие
pattern_sl_close  = re.compile(
    r".*\[(?P<symbol>[^\]]+)\].*СДЕЛКА\s+ЗАКРЫТА\s+по\s+Stop\s+Loss\s*@\s*(?P<price>\d+\.\d+)"
)


def parse_and_label(path: str) -> pd.DataFrame:
    """Парсит лог-файл и создает DataFrame с размеченными сделками."""
    active, records = [], []

    # Функция, закрывающая сделку из active
    def close_trade(reason: str, sym: str, price: float, ts: pd.Timestamp):
        for i, t in enumerate(active):
            if t['symbol'] == sym and t['entry_time'].date() == ts.date():
                pnl = (price - t['entry_price']) / t['entry_price']
                if t['side'] == 'SHORT':
                    pnl = -pnl
                # label=1 для TP1/2/3, иначе по знаку PnL
                label = 1 if reason.startswith('TP') else (1 if pnl > 0 else 0)
                records.append({
                    'symbol':      sym,
                    'side':        t['side'],
                    'entry_time':  t['entry_time'],
                    'entry_price': t['entry_price'],
                    'exit_time':   ts,
                    'exit_price':  price,
                    'reason':      reason,
                    'pnl':         pnl,
                    'label':       label
                })
                active.pop(i)
                break

    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                # 0) Сброс «восстановленных» позиций
                if pattern_restore.search(line):
                    active.clear()
                    continue

                # 1) Парсим timestamp
                ts = pd.to_datetime(line.split(" - ")[0], errors='coerce')
                if pd.isna(ts):
                    continue

                # 2) Открытие
                m = (pattern_open_watch.search(line)
                     or pattern_open_confirm.search(line))
                if m:
                    active.append({
                        'symbol':      m.group('symbol'),
                        'side':        m.group('side'),
                        'entry_time':  ts,
                        'entry_price': float(m.group('price'))
                    })
                    continue

                # 3) Закрытие по TP1
                m = pattern_tp1_close.search(line)
                if m:
                    close_trade('TP1', m.group('symbol'),
                                float(m.group('price')), ts)
                    continue

                # 4) Закрытие по TP2
                m = pattern_tp2_close.search(line)
                if m:
                    close_trade('TP2', m.group('symbol'),
                                float(m.group('price')), ts)
                    continue

                # 5) Закрытие по TP3
                m = pattern_tp3_close.search(line)
                if m:
                    close_trade('TP3', m.group('symbol'),
                                float(m.group('price')), ts)
                    continue

                # 6) Закрытие по SL
                m = pattern_sl_close.search(line)
                if m:
                    close_trade('SL', m.group('symbol'),
                                float(m.group('price')), ts)
                    continue
    except FileNotFoundError:
        logger.error(f"Файл лога {path} не найден.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Ошибка при парсинге лога: {e}")
        return pd.DataFrame()

    return pd.DataFrame(records, columns=[
        'symbol','side','entry_time','entry_price',
        'exit_time','exit_price','reason','pnl','label'
    ])


def main():
    """Основная функция парсинга логов."""
    logger.info(f"Начало парсинга логов из {LOG_FILE}...")
    
    df = parse_and_label(LOG_FILE)
    
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False, float_format='%.8f')
        logger.info(f"✅ Извлечено и сохранено {len(df)} сделок в {OUTPUT_CSV}")
        logger.info(f"Распределение по меткам:\n{df['label'].value_counts()}")
    else:
        logger.warning(f"Не найдено сделок в {LOG_FILE}. Создан пустой файл.")
        # Создаем пустой CSV с заголовками
        empty_df_cols = ['symbol','side','entry_time','entry_price',
                         'exit_time','exit_price','reason','pnl','label']
        pd.DataFrame(columns=empty_df_cols).to_csv(OUTPUT_CSV, index=False)

if __name__ == '__main__':
    main()
# data_generation/generate_ideal_labels.py
"""
Генерация идеальных (экспертных) меток для обучения.
Анализирует исторические данные и создает гипотетические прибыльные/убыточные сделки.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import ccxt
import warnings
from datetime import datetime, timezone

# Импортируем логгер
from utils.logger import setup_logger

logger = setup_logger("generate_ideal_labels")

# Подавляем предупреждение о pkg_resources
warnings.filterwarnings('ignore', category=UserWarning, module='pandas_ta')

# --- Параметры ---
CONFIG_FILE = 'config.json'
OUTPUT_FILE = 'expert_trades.csv'

# Настройки для генерации идеальных сделок
TARGET_PROFIT_PERCENT = 1.5  # Целевой профит в %
MAX_DRAWDOWN_PERCENT = 1.5   # Максимальная просадка в %
STOP_LOSS_PERCENT = 1.5      # Стоп-лосс в %
LOOKAHEAD_CANDLES = 100      # Сколько свечей смотреть вперед

# 1) Загрузить конфиг
with open(CONFIG_FILE, encoding='utf-8') as f:
    cfg = json.load(f)
SYMBOLS = cfg['symbols']
TIMEFRAME = cfg['timeframe']

# 2) Инициализация биржи
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
    },
})

def fetch_ohlcv(symbol: str, limit: int = 2000) -> pd.DataFrame:
    """Получить OHLCV данные."""
    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df

def generate_ideal_labels_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Генерация идеальных меток для одного символа.
    """
    records = []
    
    # Итерируемся по свечам, начиная с достаточно ранней точки, чтобы было куда "смотреть вперед"
    for i in range(len(df) - LOOKAHEAD_CANDLES - 1):
        entry_idx = i
        entry_time = df.index[entry_idx]
        entry_price = df['close'].iloc[entry_idx]
        
        # LONG сделка
        tp_long = entry_price * (1 + TARGET_PROFIT_PERCENT / 100)
        sl_long = entry_price * (1 - STOP_LOSS_PERCENT / 100)
        
        # SHORT сделка
        tp_short = entry_price * (1 - TARGET_PROFIT_PERCENT / 100)
        sl_short = entry_price * (1 + STOP_LOSS_PERCENT / 100)
        
        # Анализируем следующие LOOKAHEAD_CANDLES свечей
        lookahead_window = df.iloc[entry_idx+1 : entry_idx+1+LOOKAHEAD_CANDLES]
        
        # --- Для LONG ---
        long_result = analyze_trade(lookahead_window, entry_price, tp_long, sl_long, "LONG")
        if long_result:
            reason, exit_time, exit_price, pnl_ratio = long_result
            records.append({
                'symbol': symbol,
                'side': 'LONG',
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'reason': reason,
                'pnl_ratio': pnl_ratio,
                'label': 1 if reason.startswith('TP') else 0
            })
            
        # --- Для SHORT ---
        short_result = analyze_trade(lookahead_window, entry_price, tp_short, sl_short, "SHORT")
        if short_result:
            reason, exit_time, exit_price, pnl_ratio = short_result
            records.append({
                'symbol': symbol,
                'side': 'SHORT',
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'reason': reason,
                'pnl_ratio': pnl_ratio,
                'label': 1 if reason.startswith('TP') else 0
            })
            
    return pd.DataFrame(records)

def analyze_trade(window: pd.DataFrame, entry_price: float, tp: float, sl: float, side: str):
    """
    Анализирует окно свечей для определения результата сделки.
    """
    for idx, row in window.iterrows():
        if side == 'LONG':
            # Проверяем TP
            if row['high'] >= tp:
                # TP может быть достигнут в любой момент свечи, возьмем цену TP
                exit_price = tp
                pnl = (exit_price - entry_price) / entry_price
                return 'TP', idx, exit_price, pnl / (STOP_LOSS_PERCENT / 100)
            # Проверяем SL
            if row['low'] <= sl:
                exit_price = sl
                pnl = (exit_price - entry_price) / entry_price
                return 'SL', idx, exit_price, pnl / (STOP_LOSS_PERCENT / 100)
        else: # SHORT
            # Проверяем TP
            if row['low'] <= tp:
                exit_price = tp
                pnl = (entry_price - exit_price) / entry_price
                return 'TP', idx, exit_price, pnl / (STOP_LOSS_PERCENT / 100)
            # Проверяем SL
            if row['high'] >= sl:
                exit_price = sl
                pnl = (entry_price - exit_price) / entry_price
                return 'SL', idx, exit_price, pnl / (STOP_LOSS_PERCENT / 100)
    # Если дошли до конца окна, сделка не закрылась
    return None

def main():
    """Основная функция генерации идеальных меток."""
    logger.info("Начало генерации идеальных меток...")
    all_labels = []

    for sym in SYMBOLS:
        logger.info(f"→ Обработка {sym}")
        try:
            df = fetch_ohlcv(sym, limit=2000)
            if df.empty:
                logger.warning(f"   Пропущен {sym} — нет данных")
                continue

            sym_labels = generate_ideal_labels_for_symbol(df, sym)
            if not sym_labels.empty:
                all_labels.append(sym_labels)
                logger.info(f"   Обработано {sym}: {len(sym_labels)} идеальных сделок")
            else:
                logger.warning(f"   Пропущен {sym} — не удалось сгенерировать метки")
                
        except Exception as e:
            logger.error(f"Ошибка обработки {sym}: {e}")
            continue

    if all_labels:
        final_df = pd.concat(all_labels, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False, float_format='%.8f')
        logger.info(f"✅ Идеальные метки сохранены в {OUTPUT_FILE}. Всего: {len(final_df)} сделок.")
        logger.info(f"Распределение по меткам:\n{final_df['label'].value_counts()}")
    else:
        logger.error("❌ Не удалось сгенерировать идеальные метки.")
        # Создаем пустой файл, чтобы избежать ошибок в дальнейшем
        pd.DataFrame(columns=[
            'symbol', 'side', 'entry_time', 'entry_price',
            'exit_time', 'exit_price', 'reason', 'pnl_ratio', 'label'
        ]).to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Создан пустой файл {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
# data/generate_enhanced_history.py
"""
Скрипт для генерации исторических данных с расширенным набором индикаторов и признаков.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import ccxt
from features.indicators import add_classic_indicators, add_advanced_indicators
from features.multi_timeframe import fetch_ohlcv_for_multiple_timeframes, add_indicators_to_multi_tf_data, merge_timeframes
from features.feature_engineering import create_final_features
import time # Для измерения времени

# --- Параметры ---
CONFIG_FILE = 'config.json'
# ИСПРАВЛЕНО: Путь к выходному файлу в подпапке
OUTPUT_FILE = 'data/enhanced/enhanced_full_history_with_indicators.csv'
HISTORY_LIMIT = 500 # Уменьшено для теста

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

def main():
    all_dfs = []
    multi_tf_timeframes = ['1m', '5m', '15m', '1h'] # Таймфреймы для анализа

    for sym in SYMBOLS:
        print(f"→ Fetching enhanced data for {sym}")
        start_time_sym = time.time() # Начало отсчета времени для символа
        
        try:
            # 1. Получить многотаймфреймовые данные
            print(f"   Запрашиваем данные для таймфреймов: {multi_tf_timeframes}")
            multi_tf_data = fetch_ohlcv_for_multiple_timeframes(exchange, sym, timeframes=multi_tf_timeframes, limit=HISTORY_LIMIT)
            print(f"   Получены данные для таймфреймов: {list(multi_tf_data.keys())}")
            
            # Проверим, есть ли данные
            data_available = any(not df.empty for df in multi_tf_data.values())
            if not data_available:
                 print(f"   Пропущен {sym} — нет данных с биржи.")
                 continue

            # 2. Добавить индикаторы ко всем таймфреймам
            print(f"   Добавляем индикаторы...")
            multi_tf_data = add_indicators_to_multi_tf_data(multi_tf_data)
            print(f"   Индикаторы добавлены.")

            # 3. Объединить данные в один DataFrame (основной таймфрейм 5m)
            print(f"   Объединяем данные с разных таймфреймов...")
            merged_df = merge_timeframes(multi_tf_data)
            print(f"   Данные объединены. Размер: {merged_df.shape}")
            
            if merged_df.empty:
                print(f"   skipped {sym} — no merged data")
                elapsed_sym = time.time() - start_time_sym
                print(f"   Обработка {sym} заняла {elapsed_sym:.2f} секунд")
                continue

            # 4. Добавить расширенные признаки (лаги, дельты, окна, комбинации)
            print(f"   Создаем расширенные признаки...")
            df_final = create_final_features(merged_df)
            print(f"   Расширенные признаки созданы. Размер: {df_final.shape}")
            
            if df_final.empty:
                print(f"   skipped {sym} — final features could not be computed")
                elapsed_sym = time.time() - start_time_sym
                print(f"   Обработка {sym} заняла {elapsed_sym:.2f} секунд")
                continue

            df_final['symbol'] = sym
            
            all_dfs.append(df_final)
            print(f"   ✓ Processed {sym}: {len(df_final)} rows")
            elapsed_sym = time.time() - start_time_sym
            print(f"   Обработка {sym} заняла {elapsed_sym:.2f} секунд")
            
        except Exception as e:
            elapsed_sym = time.time() - start_time_sym
            print(f"   ✗ Error processing {sym}: {e}")
            print(f"   Обработка {sym} заняла {elapsed_sym:.2f} секунд (с ошибкой)")
            continue

    if all_dfs:
        print("Объединяем все данные...")
        full_history = pd.concat(all_dfs).reset_index()
        # Обеспечить, что timestamp в UTC
        full_history['timestamp'] = pd.to_datetime(full_history['timestamp'], utc=True)
        
        # Убедиться, что временные признаки созданы (на случай, если create_final_features их не добавил)
        if 'hour' not in full_history.columns:
            full_history['hour'] = full_history['timestamp'].dt.hour
            
        # ИСПРАВЛЕНО: Убедимся, что директория существует перед сохранением
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        print(f"Сохраняем данные в {OUTPUT_FILE}...")
        full_history.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Расширенная история с индикаторами и признаками сохранена в {OUTPUT_FILE}. Всего: {len(full_history)} строк.")
    else:
        print("❌ Нет данных для сохранения расширенной истории.")
        # ИСПРАВЛЕНО: Убедимся, что директория существует и создаем пустой файл
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        # Создаем пустой CSV с заголовками
        empty_df_cols = ['timestamp', 'symbol'] # Добавь сюда ключевые колонки
        pd.DataFrame(columns=empty_df_cols).to_csv(OUTPUT_FILE, index=False)

if __name__ == '__main__':
    main()

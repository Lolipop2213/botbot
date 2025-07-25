# features/multi_timeframe.py
"""
Работа с данными из нескольких таймфреймов.
"""
import ccxt
import pandas as pd
from .indicators import add_classic_indicators, add_advanced_indicators

def fetch_ohlcv_for_multiple_timeframes(exchange, symbol, timeframes=['1m', '5m', '15m', '1h'], limit=1000):
    """Получить данные для нескольких таймфреймов"""
    data = {}
    for tf in timeframes:
        try:
            # Убедимся, что используем правильный формат для ccxt (например, '5m')
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            data[tf] = df
        except Exception as e:
            print(f"Ошибка при загрузке данных {symbol} {tf}: {e}")
            data[tf] = pd.DataFrame() # Возвращаем пустой DataFrame в случае ошибки
    return data

def add_indicators_to_multi_tf_data(data):
    """Добавить индикаторы ко всем таймфреймам"""
    for tf, df in data.items():
        if not df.empty:
            df = add_classic_indicators(df)
            df = add_advanced_indicators(df)
            data[tf] = df
    return data

def merge_timeframes(data, base_tf='5m'):
    """
    Объединить данные с разных таймфреймов в один DataFrame.
    Использует merge_asof для правильного выравнивания по времени.
    """
    # Базовый таймфрейм
    # ИСПРАВЛЕНО: правильная проверка наличия ключа
    if base_tf not in data: 
        # Если базовый таймфрейм не найден, берем первый из доступных
        base_tf = list(data.keys())[0] if data.keys() else None
        if not base_tf:
            return pd.DataFrame() # Возвращаем пустой DataFrame если данных нет

    base_df = data[base_tf].copy()
    if base_df.empty:
        return base_df

    # Сбросим индекс, чтобы 'timestamp' стал колонкой, необходимой для merge_asof
    base_df_reset = base_df.reset_index()

    # Объединяем данные с других таймфреймов
    for tf, df in data.items():
        if tf == base_tf or df.empty:
            continue

        # Сбросим индекс для другого таймфрейма
        df_reset = df.reset_index()
        
        # Используем merge_asof для поиска ближайшего предыдущего значения
        # direction='backward' означает искать последнее значение ДО или в момент времени base_df
        # suffixes=('', f'_{tf}') добавит суффикс только к колонкам из df
        try:
            merged_reset = pd.merge_asof(
                base_df_reset, 
                df_reset, 
                on='timestamp', 
                direction='backward', 
                suffixes=('', f'_{tf}')
            )
            # Обновляем base_df_reset для следующей итерации
            base_df_reset = merged_reset
        except Exception as e:
            print(f"Ошибка при объединении таймфреймов {base_tf} и {tf}: {e}")
            # В случае ошибки, продолжаем с текущим base_df_reset
            continue

    # Устанавливаем 'timestamp' обратно как индекс
    if 'timestamp' in base_df_reset.columns:
        final_df = base_df_reset.set_index('timestamp')
    else:
        # На случай, если что-то пошло не так
        print("Ошибка: 'timestamp' отсутствует в merged DataFrame")
        final_df = pd.DataFrame()
        
    return final_df

def get_multi_timeframe_features(exchange, symbol, timeframes=['1m', '5m', '15m', '1h'], base_tf='5m'):
    """Основная функция для получения мультитаймфреймовых признаков"""
    # 1. Получить данные
    data = fetch_ohlcv_for_multiple_timeframes(exchange, symbol, timeframes)
    
    # 2. Добавить индикаторы
    data = add_indicators_to_multi_tf_data(data)
    
    # 3. Объединить
    merged_df = merge_timeframes(data, base_tf)
    
    return merged_df

# Для тестирования модуля
if __name__ == "__main__":
    import yaml
    import os
    
    # Загрузим конфиг для теста
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    # Если config.yaml не найден, используем дефолтные значения
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("config.yaml не найден, используем дефолтные значения для теста")
        cfg = {
            'symbols': ['BTC/USDT'],
            'timeframes': {'multi': ['1m', '5m', '15m', '1h'], 'base': '5m'},
            'exchange': {
                'name': 'binance',
                'rateLimit': 1200,
                'enableRateLimit': True,
                'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
            }
        }
    except Exception as e:
        print(f"Ошибка при загрузке config.yaml: {e}")
        cfg = {
            'symbols': ['BTC/USDT'],
            'timeframes': {'multi': ['1m', '5m', '15m', '1h'], 'base': '5m'},
            'exchange': {
                'name': 'binance',
                'rateLimit': 1200,
                'enableRateLimit': True,
                'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
            }
        }
        
    symbols = cfg.get('symbols', ['BTC/USDT'])
    timeframes = cfg.get('timeframes', {}).get('multi', ['1m', '5m', '15m', '1h'])
    base_tf = cfg.get('timeframes', {}).get('base', '5m')
    
    exchange_config = cfg.get('exchange', {})
    exchange_name = exchange_config.get('name', 'binance')
    exchange = getattr(ccxt, exchange_name)({
        'rateLimit': exchange_config.get('rateLimit', 1200),
        'enableRateLimit': exchange_config.get('enableRateLimit', True),
        'options': exchange_config.get('options', {}),
    })
    
    symbol = symbols[0] # Берем первую пару для теста
    print(f"Тест получения мультитаймфреймовых данных для {symbol}")
    
    df_multi = get_multi_timeframe_features(exchange, symbol, timeframes, base_tf)
    
    print(f"Форма результирующего DataFrame: {df_multi.shape}")
    if not df_multi.empty:
        print("Первые строки:")
        print(df_multi.head())
        print("\nПоследние строки:")
        print(df_multi.tail())
        print("\nКолонки (первые 20):")
        print(df_multi.columns.tolist()[:20])
    else:
        print("Результирующий DataFrame пуст.")

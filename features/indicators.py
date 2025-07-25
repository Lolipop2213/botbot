# features/indicators.py
"""
Расширенный набор технических индикаторов, использующий pandas_ta.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np

def add_classic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить классические технические индикаторы с помощью pandas_ta"""
    df = df.copy()
    
    # Трендовые
    df.ta.ema(length=20, append=True)
    df.rename(columns={f'EMA_20': 'EMA_20'}, inplace=True)
    
    df.ta.ema(length=50, append=True)
    df.rename(columns={f'EMA_50': 'EMA_50'}, inplace=True)
    
    df.ta.sma(length=200, append=True)
    df.rename(columns={f'SMA_200': 'SMA_200'}, inplace=True)
    
    # Импульсные
    df.ta.rsi(length=14, append=True)
    df.rename(columns={f'RSI_14': 'RSI'}, inplace=True)
    
    df.ta.stoch(append=True)
    # pandas_ta обычно создает колонки STOCHk_... и STOCHd_...
    # Переименуем в стандартные имена, если они существуют
    if f'STOCHk_14_3_3' in df.columns:
        df.rename(columns={f'STOCHk_14_3_3': 'STOCH_K'}, inplace=True)
    if f'STOCHd_14_3_3' in df.columns:
        df.rename(columns={f'STOCHd_14_3_3': 'STOCH_D'}, inplace=True)
    
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    # Переименовываем стандартные колонки MACD
    if f'MACD_12_26_9' in df.columns:
        df.rename(columns={f'MACD_12_26_9': 'MACD'}, inplace=True)
    if f'MACDs_12_26_9' in df.columns:
        df.rename(columns={f'MACDs_12_26_9': 'MACD_SIGNAL'}, inplace=True)
    if f'MACDh_12_26_9' in df.columns:
        df.rename(columns={f'MACDh_12_26_9': 'MACD_HIST'}, inplace=True)
    
    # Волатильность
    df.ta.atr(length=14, append=True)
    df.rename(columns={f'ATR_14': 'ATR'}, inplace=True)
    
    df.ta.bbands(length=20, std=2, append=True)
    # Переименовываем Bollinger Bands
    if f'BBL_20_2.0' in df.columns:
        df.rename(columns={f'BBL_20_2.0': 'BB_LOWER'}, inplace=True)
    if f'BBM_20_2.0' in df.columns:
        df.rename(columns={f'BBM_20_2.0': 'BB_MIDDLE'}, inplace=True)
    if f'BBU_20_2.0' in df.columns:
        df.rename(columns={f'BBU_20_2.0': 'BB_UPPER'}, inplace=True)
    # Ширина полос (если нужна)
    # if f'BBB_20_2.0' in df.columns:
    #     df.rename(columns={f'BBB_20_2.0': 'BB_WIDTH'}, inplace=True)
        
    # Объемные (простой OBV из pandas_ta)
    df.ta.obv(append=True)
    df.rename(columns={'OBV': 'OBV'}, inplace=True)
    
    return df

def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить современные и экзотические индикаторы с помощью pandas_ta"""
    df = df.copy()
    
    # Примеры других индикаторов из pandas_ta
    # Supertrend (реализация через pandas_ta)
    df.ta.supertrend(length=10, multiplier=3, append=True)
    # pandas_ta создает колонки SUPERT_... и SUPERTd_...
    # Переименуем основную линию
    if f'SUPERT_10_3.0' in df.columns:
        df.rename(columns={f'SUPERT_10_3.0': 'SUPERTREND'}, inplace=True)
    # Направление тренда (если нужно)
    # if f'SUPERTd_10_3.0' in df.columns:
    #     df.rename(columns={f'SUPERTd_10_3.0': 'SUPERTREND_DIR'}, inplace=True)
    # if f'SUPERTl_10_3.0' in df.columns:
    #     df.rename(columns={f'SUPERTl_10_3.0': 'SUPERTREND_LONG'}, inplace=True)
    # if f'SUPERTs_10_3.0' in df.columns:
    #     df.rename(columns={f'SUPERTs_10_3.0': 'SUPERTREND_SHORT'}, inplace=True)
    
    # Ichimoku (упрощенная версия)
    # pandas_ta имеет ichimoku, но он создает много колонок
    # Для простоты можно использовать комбинации других индикаторов или пропустить
    
    # Fractals (pandas_ta не имеет встроенной функции, можно реализовать вручную или пропустить)
    # Для примера добавим ADX, который также полезен для определения силы тренда
    df.ta.adx(length=14, append=True)
    # Переименовываем
    if f'ADX_14' in df.columns:
        df.rename(columns={f'ADX_14': 'ADX'}, inplace=True)
    if f'ADXR_14' in df.columns:
        df.rename(columns={f'ADXR_14': 'ADXR'}, inplace=True)
        
    # Добавим еще несколько полезных индикаторов
    # CCI
    df.ta.cci(length=20, append=True)
    df.rename(columns={f'CCI_20_0.015': 'CCI'}, inplace=True)
    
    # VWAP (если есть данные по объему)
    # df.ta.vwap(append=True) # Может потребовать дополнительных параметров
    
    return df

# Функции calculate_supertrend, calculate_ichimoku, calculate_fractals из предыдущего примера
# больше не нужны, так как мы используем встроенные функции pandas_ta.
# Если потребуется их реализация вручную, можно добавить позже.

# Для тестирования модуля
if __name__ == "__main__":
    # Создадим фиктивный DataFrame для теста
    dates = pd.date_range('2023-01-01', periods=100, freq='5T')
    data = {
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100 + 1,
        'low': np.random.rand(100) * 100 - 1,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 10000,
    }
    df_test = pd.DataFrame(data, index=dates)
    df_test = df_test.astype(float)
    
    print("Исходный DataFrame:")
    print(df_test.head())
    
    df_with_indicators = add_classic_indicators(df_test)
    df_with_indicators = add_advanced_indicators(df_with_indicators)
    
    print("\nDataFrame с индикаторами (первые 10 колонок):")
    print(df_with_indicators.head())
    print("\nКолонки в DataFrame:")
    print(df_with_indicators.columns.tolist())
    print(f"\nФорма до: {df_test.shape}, форма после: {df_with_indicators.shape}")

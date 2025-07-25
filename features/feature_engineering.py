# features/feature_engineering.py
"""
Модуль для создания расширенного набора признаков из свечных данных.
"""
import pandas as pd
import numpy as np
from features.indicators import add_classic_indicators, add_advanced_indicators
import logging
logger = logging.getLogger(__name__)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить временные признаки."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def add_lag_features(df: pd.DataFrame, columns: list, lags: list = [1, 2, 3]) -> pd.DataFrame:
    """Добавить лаги для указанных колонок."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            for lag in lags:
                new_col_name = f'{col}_lag_{lag}'
                df[new_col_name] = df[col].shift(lag)
                # print(f"      Добавлен лаг: {new_col_name}") # Отладка
    return df

def add_delta_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Добавить дельты (изменения) и процентные изменения для указанных колонок."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            delta_col_name = f'{col}_delta'
            pct_col_name = f'{col}_pct_change'
            df[delta_col_name] = df[col].diff()
            # Явно указываем fill_method=None, чтобы избежать FutureWarning и потенциальных проблем
            df[pct_col_name] = df[col].pct_change(fill_method=None) 
            # print(f"      Добавлены дельты: {delta_col_name}, {pct_col_name}") # Отладка
    return df

def add_rolling_features(df: pd.DataFrame, columns: list, windows: list = [5, 10, 20]) -> pd.DataFrame:
    """Добавить скользящие окна (mean, std) для указанных колонок."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            for window in windows:
                mean_col_name = f'{col}_rolling_mean_{window}'
                std_col_name = f'{col}_rolling_std_{window}'
                df[mean_col_name] = df[col].rolling(window=window).mean()
                df[std_col_name] = df[col].rolling(window=window).std()
                # print(f"      Добавлены rolling: {mean_col_name}, {std_col_name}") # Отладка
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить комбинации признаков."""
    df = df.copy()
    # Примеры комбинаций (добавь больше по своему усмотрению)
    if 'RSI' in df.columns and 'ATR' in df.columns and 'close' in df.columns:
        # Избегаем деления на 0 или очень маленькое число
        df['RSI_ATR_ratio'] = df['RSI'] / (df['ATR'] / df['close'] + 1e-10) 
        
    if 'MACD' in df.columns and 'MACD_SIGNAL' in df.columns:
        df['MACD_histogram'] = df['MACD'] - df['MACD_SIGNAL']
        
    if 'close' in df.columns and 'SMA_200' in df.columns:
        # Избегаем деления на 0
        df['price_to_sma_ratio'] = df['close'] / (df['SMA_200'] + 1e-10) 
        
    if 'BB_UPPER' in df.columns and 'BB_LOWER' in df.columns and 'close' in df.columns:
        bb_width = df['BB_UPPER'] - df['BB_LOWER']
        # Избегаем деления на 0
        df['price_bb_position'] = (df['close'] - df['BB_LOWER']) / (bb_width + 1e-10) 
        
    return df

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить признаки рыночного режима (тренд/флет)."""
    df = df.copy()
    # Простой пример: если ATR выше среднего, то считаем это трендом
    if 'ATR' in df.columns:
        df['ATR_rolling_mean_20'] = df['ATR'].rolling(window=20).mean()
        # Заполняем потенциальные NaN в rolling_mean перед сравнением
        df['ATR_rolling_mean_20'].ffill(inplace=True)
        df['ATR_rolling_mean_20'].bfill(inplace=True)
        df['high_volatility_regime'] = (df['ATR'] > df['ATR_rolling_mean_20']).astype(int)
    return df

def create_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создать финальный набор признаков.
    Предполагается, что df уже содержит все базовые и продвинутые индикаторы.
    """
    # Минимальный лог
    logger.info(f"   create_final_features: Начальный размер df: {df.shape}")
    if df.empty:
        logger.warning(f"   create_final_features: Входной df пуст!")
        return df
        
    df = df.copy()
    
    # 1. Временные признаки
    df = add_time_features(df)
    
    # 2. Определим, какие колонки использовать для лагов, дельт и окон
    # ИСПРАВЛЕНО: Используем ТОЛЬКО базовые колонки, без суффиксов других таймфреймов
    core_price_cols = ['open', 'high', 'low', 'close', 'volume']
    core_indicator_cols = [
        'RSI', 'MACD', 'MACD_SIGNAL', 'ATR',
        'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER',
        'STOCH_K', 'STOCH_D', 'SUPERTREND', 'OBV', 'ADX'
    ]
    advanced_indicator_cols = [
        'ICHIMOKU_CONV', 'ICHIMOKU_BASE'
    ]
    
    def is_base_tf_column(col_name):
        return not any(col_name.endswith(f"_{tf}") for tf in ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d'])

    available_core_price_cols = [col for col in core_price_cols if col in df.columns and is_base_tf_column(col)]
    available_core_indicator_cols = [col for col in core_indicator_cols if col in df.columns and is_base_tf_column(col)]
    available_advanced_indicator_cols = [col for col in advanced_indicator_cols if col in df.columns and is_base_tf_column(col)]
    
    # Используем только признаки базового таймфрейма
    all_feature_cols = available_core_price_cols + available_core_indicator_cols + available_advanced_indicator_cols
    
    # 3. ЗАПОЛНЕНИЕ NaN ПЕРЕД расчетом новых признаков
    # ИСПРАВЛЕНО: Устранены FutureWarning, используем рекомендованный способ
    cols_to_fill = available_core_price_cols + available_core_indicator_cols + available_advanced_indicator_cols
    for col in cols_to_fill:
        if col in df.columns:
            # Вместо df[col].ffill(inplace=True) используем
            df[col] = df[col].ffill()
            # Вместо df[col].bfill(inplace=True) используем
            df[col] = df[col].bfill()
    
    # Удалим строки, где ВСЕ исходные базовые колонки оказались NaN 
    initial_core_cols = available_core_price_cols + available_core_indicator_cols
    if initial_core_cols:
        all_initial_nan_mask = df[initial_core_cols].isnull().all(axis=1)
        rows_to_drop_initial = all_initial_nan_mask.sum()
        if rows_to_drop_initial > 0:
            logger.info(f"   create_final_features: Удаляем {rows_to_drop_initial} строк, где ВСЕ исходные базовые колонки NaN.")
            df = df[~all_initial_nan_mask]

    if df.empty:
        logger.warning(f"   create_final_features: df стал пустым после удаления строк с полными NaN.")
        return df

    # 4. Лаги (только по базовым признакам)
    df = add_lag_features(df, all_feature_cols, lags=[1, 2, 3])
    
    # 5. Дельты (только по базовым признакам)
    df = add_delta_features(df, all_feature_cols)
    
    # 6. Скользящие окна (только по базовым ценам/объему)
    df = add_rolling_features(df, available_core_price_cols, windows=[5, 10, 20])
    
    # 7. Комбинации (только по базовым признакам)
    df = add_interaction_features(df)
    
    # 8. Режимы (только по базовым признакам)
    df = add_regime_features(df)
    
    # 9. ФИНАЛЬНАЯ обработка NaN (заполнение, а не удаление)
    # ИСПРАВЛЕНО: Устранены FutureWarning
    df = df.ffill() # Заполняем вперед
    df = df.bfill() # Заполняем назад

    logger.info(f"   create_final_features: Финальный размер df: {df.shape}")
    return df

# Для тестирования модуля (если запускается напрямую)
if __name__ == "__main__":
    # Этот блок будет выполнен, если файл запущен напрямую
    # Для демонстрации создадим фиктивный DataFrame
    import numpy as np
    dates = pd.date_range('2023-01-01', periods=100, freq='5T')
    # Создадим данные, похожие на реальные индикаторы
    data = {
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100 + 1,
        'low': np.random.rand(100) * 100 - 1,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 10000,
        'RSI': np.random.rand(100) * 100,
        'MACD': np.random.rand(100) * 10,
        'MACD_SIGNAL': np.random.rand(100) * 10,
        'ATR': np.random.rand(100) * 5,
        'BB_UPPER': np.random.rand(100) * 100 + 5,
        'BB_MIDDLE': np.random.rand(100) * 100,
        'BB_LOWER': np.random.rand(100) * 100 - 5,
        'STOCH_K': np.random.rand(100) * 100,
        'STOCH_D': np.random.rand(100) * 100,
        'SUPERTREND': np.random.rand(100) * 100,
        # 'ICHIMOKU_CONV': np.random.rand(100) * 100, # Пусть их не будет для теста
        # 'ICHIMOKU_BASE': np.random.rand(100) * 100,
        'SMA_200': np.random.rand(100) * 100,
        'OBV': np.random.rand(100) * 100000,
        'ADX': np.random.rand(100) * 100,
    }
    df_test = pd.DataFrame(data, index=dates)
    df_test = df_test.astype(float) # Убедимся, что типы правильные
    
    print("Исходный DataFrame для теста:")
    print(df_test.tail())
    
    df_enhanced = create_final_features(df_test)
    
    print("\nDataFrame с расширенными признаками (после create_final_features):")
    print(df_enhanced.tail())
    print(f"\nФорма до: {df_test.shape}, форма после: {df_enhanced.shape}")
    if not df_enhanced.empty:
        print(f"Новые колонки: {set(df_enhanced.columns) - set(df_test.columns)}")

def prepare_data_for_prediction(exchange, symbol, timeframe, limit, history_limit_for_features=100):
    """
    Подготовить данные для получения сигнала: загрузка, индикаторы, фичи.
    history_limit_for_features определяет, сколько свечей нужно для 
    корректного расчета лагов, окон и т.п. (например, 100 для 20-периодных скользящих средних и лагов).
    """
    # Импортируем внутри функции, чтобы избежать циклических импортов, если они возникнут
    # Или убедитесь, что эти импорты находятся вверху файла
    # from .indicators import fetch_and_analyze_symbol 
    # Если fetch_and_analyze_symbol находится в features/indicators.py, импортируйте её оттуда
    # В противном случае, скопируйте логику сюда или импортируйте соответствующие функции
    
    # Для примера, предположим, что fetch_and_analyze_symbol находится в features/indicators.py
    # Убедитесь, что путь импорта корректен для вашей структуры
    try:
        from .indicators import fetch_and_analyze_symbol
    except ImportError:
        # Если не удается импортировать, определяем локально (пример)
        # Вам нужно заменить это на правильную реализацию или импорт
        def fetch_and_analyze_symbol(exchange, symbol, timeframe, limit):
            import pandas as pd
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                df = add_classic_indicators(df)
                df = add_advanced_indicators(df)
                return df
            except Exception as e:
                print(f"Ошибка при загрузке/анализе данных для {symbol}: {e}")
                return pd.DataFrame() # Возвращаем пустой DataFrame в случае ошибки

    print(f"  -> Подготовка данных для {symbol}...")
    # 1. Получить данные с биржи (больше точек для расчета индикаторов)
    df_with_indicators = fetch_and_analyze_symbol(exchange, symbol, timeframe, limit=history_limit_for_features)
    
    if df_with_indicators.empty:
        print(f"     Пропущен {symbol}: нет данных.")
        return pd.DataFrame() # Возвращаем пустой DataFrame

    # 2. Создать расширенные признаки (лаги, дельты, окна и т.д.)
    df_final_features = create_final_features(df_with_indicators)
    
    if df_final_features.empty:
        print(f"     Пропущен {symbol}: не удалось создать признаки.")
        return pd.DataFrame()

    # 3. Вернуть последнюю строку (последняя свеча) для предсказания
    # reset_index, чтобы 'timestamp' стал колонкой, если это нужно для логов
    latest_data = df_final_features.iloc[[-1]].reset_index() 
    print(f"     Данные для {symbol} подготовлены. Форма: {latest_data.shape}")
    return latest_data

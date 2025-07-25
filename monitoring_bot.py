# monitoring_bot.py
"""
Скрипт для непрерывного мониторинга рынка и генерации сигналов с помощью обученной ML-модели.
Отслеживает активные сделки и логирует достижения TP/SL.
"""
import sys
import os

# Добавляем корневую директорию проекта в путь поиска модулей
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import json
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import glob

# --- Импорт нашего функционала ---
# Импортируем логгер
from utils.logger import monitoring_logger as logger

# Импорты из нашего проекта
from features.indicators import add_classic_indicators, add_advanced_indicators # Для TP/SL
from features.feature_engineering import prepare_data_for_prediction

# --- Управление состоянием ---
BOT_STATE_FILE = 'bot_state.json'
active_trades = {} # Словарь для хранения активных сделок в памяти

def load_bot_state():
    """Загружает состояние активных сделок из файла."""
    global active_trades
    if os.path.exists(BOT_STATE_FILE):
        try:
            with open(BOT_STATE_FILE, 'r') as f:
                active_trades = json.load(f)
            logger.info(f"Состояние бота загружено. Активных сделок: {len(active_trades)}")
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния бота: {e}. Начинаем с пустого состояния.")
            active_trades = {}
    else:
        logger.info("Файл состояния бота не найден. Начинаем с пустого состояния.")
        active_trades = {}

def save_bot_state():
    """Сохраняет состояние активных сделок в файл."""
    try:
        # Убедимся, что все ключи сериализуемы
        serializable_state = {}
        for symbol, trade_info in active_trades.items():
            safe_trade_info = trade_info.copy()
            serializable_state[symbol] = safe_trade_info
            
        with open(BOT_STATE_FILE, 'w') as f:
            json.dump(serializable_state, f, indent=4, default=str)
    except Exception as e:
        logger.error(f"Ошибка сохранения состояния бота: {e}")

# --- Загрузка модели ---
def load_latest_model(model_dir="models/ensemble"):
    """
    Ищет в папке 'models' файлы trade_model_*.pkl
    и возвращает модель + список фич из самого свежего.
    """
    import joblib
    pattern = "ensemble_model_*.pkl"
    files = glob.glob(os.path.join(model_dir, pattern))
    if not files:
        raise FileNotFoundError(f"В папке '{model_dir}' нет ни одного файла '{pattern}'")
    latest = max(files, key=os.path.getmtime)
    logger.info(f"Загрузка последней модели: {latest}")
    mdl = joblib.load(latest)
    return mdl["model"], mdl["features"]

# --- Расчет TP/SL ---
def calculate_dynamic_tp_sl(df_recent, side, sl_coef=2.0, tp_window=10):
    """
    Рассчитать TP1, TP2, TP3 и SL на основе последних данных.
    TP1, TP2, TP3 основаны на экстремальных значениях в окне tp_window.
    df_recent: DataFrame с последними данными, включая индикаторы.
    side: 'LONG' или 'SHORT'.
    sl_coef: Коэффициент для расчета SL (умножается на ATR).
    tp_window: Количество последних свечей для анализа TP.
    """
    if df_recent.empty:
        return None, None, None, None
        
    last_index = df_recent.index[-1]
    current_price = df_recent.at[last_index, 'close']
    
    # Определяем окно для анализа TP/SL (последние tp_window свечей, включая текущую)
    analysis_window = df_recent.tail(tp_window)
    
    # --- Расчет Stop Loss ---
    if side == 'LONG':
        # SL: минимум последних N свечей или уровень поддержки (например, нижняя полоса Боллинжера)
        sl_level = analysis_window['low'].min()
        # Используем ATR из последней свечи
        if 'ATR' in df_recent.columns and not pd.isna(df_recent.at[last_index, 'ATR']):
            sl_level_atr = current_price - (df_recent.at[last_index, 'ATR'] * sl_coef)
            # Берем более строгий (меньший) уровень между историческим минимумом и ATR
            sl_level = min(sl_level, sl_level_atr)
        else:
            # Если ATR недоступен, используем простую логику
            sl_level = current_price * 0.99 # Например, 1% от текущей цены
            
    elif side == 'SHORT':
        # SL: максимум последних N свечей или уровень сопротивления
        sl_level = analysis_window['high'].max()
        # Используем ATR из последней свечи
        if 'ATR' in df_recent.columns and not pd.isna(df_recent.at[last_index, 'ATR']):
            sl_level_atr = current_price + (df_recent.at[last_index, 'ATR'] * sl_coef)
            # Берем более строгий (больший) уровень между историческим максимумом и ATR
            sl_level = max(sl_level, sl_level_atr)
        else:
            # Если ATR недоступен, используем простую логику
            sl_level = current_price * 1.01 # Например, 1% от текущей цены
    else:
        return None, None, None, None

    # --- Расчет Take Profit ---
    if side == 'LONG':
        # TP1: Среднее значение High в окне анализа
        tp1_level = analysis_window['high'].mean()
        # TP2: 75-й процентиль High в окне анализа
        tp2_level = analysis_window['high'].quantile(0.75)
        # TP3: Максимум High в окне анализа (максимальный рост)
        tp3_level = analysis_window['high'].max()
        
    elif side == 'SHORT':
        # TP1: Среднее значение Low в окне анализа
        tp1_level = analysis_window['low'].mean()
        # TP2: 25-й процентиль Low в окне анализа
        tp2_level = analysis_window['low'].quantile(0.25)
        # TP3: Минимум Low в окне анализа (максимальное падение)
        tp3_level = analysis_window['low'].min()
    else:
        return None, None, None, None

    # --- Ограничения ---
    if side == 'LONG':
        sl_level = min(sl_level, current_price * 0.995)
        tp1_level = max(tp1_level, current_price * 1.001)
        tp2_level = max(tp2_level, tp1_level * 1.001)
        tp3_level = max(tp3_level, tp2_level * 1.001)
        
    else: # SHORT
        sl_level = max(sl_level, current_price * 1.005)
        tp1_level = min(tp1_level, current_price * 0.999)
        tp2_level = min(tp2_level, tp1_level * 0.999)
        tp3_level = min(tp3_level, tp2_level * 0.999)
        
    return (
        round(tp1_level, 8), 
        round(tp2_level, 8), 
        round(tp3_level, 8), 
        round(sl_level, 8)
    )

# --- Проверка активных сделок ---
def check_active_trades(exchange, timeframe):
    """
    Проверяет, достигнуты ли TP/SL для активных сделок.
    """
    global active_trades
    if not active_trades:
        return

    logger.info(f"-> Проверка {len(active_trades)} активных сделок...")
    trades_to_remove = []
    for symbol, trade_info in list(active_trades.items()):
        try:
            side = trade_info['side']
            entry_price = trade_info['entry_price']
            tp1 = trade_info['tp1']
            tp2 = trade_info['tp2']
            tp3 = trade_info['tp3']
            sl = trade_info['sl']
            entry_time = trade_info.get('entry_time', 'N/A')

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1)
            if not ohlcv:
                logger.warning(f"Не удалось получить данные для {symbol} при проверке сделки.")
                continue
            current_candle = ohlcv[-1]
            current_high = current_candle[2] # high
            current_low = current_candle[3] # low

            event_logged = False

            # --- Проверка достижений ---
            if side == 'LONG':
                if current_high >= tp3:
                    logger.info(f"🎉 [{symbol}] ДОСТИГНУТ TP3 ({tp3:.8f})! Сделка закрыта. Вход: {entry_price:.8f}")
                    trades_to_remove.append(symbol)
                    event_logged = True
                elif current_low <= sl:
                    logger.info(f"💔 [{symbol}] ДОСТИГНУТ SL ({sl:.8f})! Сделка закрыта. Вход: {entry_price:.8f}")
                    trades_to_remove.append(symbol)
                    event_logged = True
                if not event_logged and current_high >= tp2:
                    if not trade_info.get('tp2_hit', False):
                         logger.info(f"✅ [{symbol}] ДОСТИГНУТ TP2 ({tp2:.8f}). Вход: {entry_price:.8f}")
                         active_trades[symbol]['tp2_hit'] = True
                         event_logged = True
                if not event_logged and current_high >= tp1:
                    if not trade_info.get('tp1_hit', False) and not trade_info.get('tp2_hit', False):
                         logger.info(f"✅ [{symbol}] ДОСТИГНУТ TP1 ({tp1:.8f}). Вход: {entry_price:.8f}")
                         active_trades[symbol]['tp1_hit'] = True

            elif side == 'SHORT':
                if current_low <= tp3:
                    logger.info(f"🎉 [{symbol}] ДОСТИГНУТ TP3 ({tp3:.8f})! Сделка закрыта. Вход: {entry_price:.8f}")
                    trades_to_remove.append(symbol)
                    event_logged = True
                elif current_high >= sl:
                    logger.info(f"💔 [{symbol}] ДОСТИГНУТ SL ({sl:.8f})! Сделка закрыта. Вход: {entry_price:.8f}")
                    trades_to_remove.append(symbol)
                    event_logged = True
                if not event_logged and current_low <= tp2:
                    if not trade_info.get('tp2_hit', False):
                         logger.info(f"✅ [{symbol}] ДОСТИГНУТ TP2 ({tp2:.8f}). Вход: {entry_price:.8f}")
                         active_trades[symbol]['tp2_hit'] = True
                         event_logged = True
                if not event_logged and current_low <= tp1:
                    if not trade_info.get('tp1_hit', False) and not trade_info.get('tp2_hit', False):
                         logger.info(f"✅ [{symbol}] ДОСТИГНУТ TP1 ({tp1:.8f}). Вход: {entry_price:.8f}")
                         active_trades[symbol]['tp1_hit'] = True

        except Exception as e:
            logger.error(f"Ошибка проверки сделки для {symbol}: {e}")

    # --- Удаление закрытых сделок ---
    for symbol in trades_to_remove:
        active_trades.pop(symbol, None)
        
    # --- Сохранение состояния ---
    if trades_to_remove:
        save_bot_state()
        logger.info(f"-> Активных сделок после проверки: {len(active_trades)}")

# --- Загрузка конфигурации ---
def load_config(config_path='config.json'):
    """Загрузить конфигурацию."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.critical(f"Критическая ошибка загрузки {config_path}: {e}")
        raise

# --- Основной цикл ---
def main():
    """Основной цикл мониторинга."""
    logger.info("--- Запуск Monitoring Bot ---")
    
    # --- Загрузка состояния ---
    load_bot_state()
    
    # --- 1. Загрузка конфигурации ---
    try:
        config = load_config()
        symbols = config['symbols']
        timeframe = config['timeframe']
        monitoring_interval = config.get('monitoring_interval_seconds', 60)
        filter_threshold = config.get('filter_threshold', 0.5)
        exchange_config = config.get('exchange', {})
        logger.info(f"Конфиг загружен. Символов: {len(symbols)}, Интервал: {monitoring_interval}с, Порог: {filter_threshold}")
    except Exception as e:
        logger.critical(f"Критическая ошибка загрузки config.json: {e}")
        return

    # --- 2. Инициализация биржи ---
    try:
        exchange = ccxt.binance({
            'rateLimit': exchange_config.get('rateLimit', 1200),
            'enableRateLimit': exchange_config.get('enableRateLimit', True),
            'options': exchange_config.get('options', {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            }),
        })
        logger.info(f"Биржа {exchange.id} инициализирована.")
    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации биржи: {e}")
        return

    # --- 3. Загрузка модели ---
    try:
        model, features = load_latest_model()
        logger.info(f"Модель загружена. Признаков: {len(features)}")
    except Exception as e:
        logger.critical(f"Критическая ошибка загрузки модели: {e}")
        return

    logger.info("--- Начало цикла мониторинга ---")
    while True:
        cycle_start_time = time.time()
        logger.info(f">>> Начало итерации мониторинга: {datetime.now(timezone.utc).isoformat()}")
        
        # --- 1. Проверка активных сделок ---
        try:
            check_active_trades(exchange, timeframe)
        except Exception as e:
             logger.error(f"Ошибка при проверке активных сделок: {e}")

        # --- 2. Поиск новых сигналов ---
        for symbol in symbols:
            if symbol in active_trades:
                continue

            try:
                data_for_prediction = prepare_data_for_prediction(
                    exchange, symbol, timeframe, limit=1, history_limit_for_features=200
                )
                
                if data_for_prediction.empty:
                    continue

                missing_features = [f for f in features if f not in data_for_prediction.columns]
                if missing_features:
                    logger.debug(f"Пропущен {symbol}: отсутствуют признаки {missing_features[:5]}...")
                    continue

                timestamp = data_for_prediction['timestamp'].iloc[0]
                X_new = data_for_prediction[features].copy()

                # --- Предсказание ---
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_new)[0, 1]
                else:
                    try:
                        import numpy as np
                        predictions_proba = []
                        for name, m in model.items():
                            if hasattr(m, 'predict_proba'):
                                p = m.predict_proba(X_new)[:, 1]
                                predictions_proba.append(p)
                        if predictions_proba:
                             prob = np.mean(predictions_proba, axis=0)[0]
                        else:
                            logger.debug(f"Пропущен {symbol}: ни одна модель в ансамбле не может предсказать.")
                            continue
                    except Exception as e:
                         logger.error(f"Ошибка предсказания ансамблем для {symbol}: {e}")
                         continue
                
                signal_status = "✅ СИГНАЛ" if prob > filter_threshold else "❌ НЕТ СИГНАЛА"
                logger.info(f"{symbol} ({timestamp}) ▶ Вероятность = {prob:.4f} (Порог={filter_threshold:.4f}) {signal_status}")

                # --- Если сигнал найден ---
                if prob > filter_threshold:
                    df_with_indicators_raw = prepare_data_for_prediction(
                        exchange, symbol, timeframe, limit=1, history_limit_for_features=50
                    ).set_index('timestamp')
                    
                    if not df_with_indicators_raw.empty:
                        side = 'LONG' # Предположим LONG для примера, можно усложнить логику
                        tp1, tp2, tp3, sl = calculate_dynamic_tp_sl(df_with_indicators_raw, side, sl_coef=2.0, tp_window=10)
                        if tp1 is not None and tp2 is not None and tp3 is not None and sl is not None:
                            logger.info(f"📈 Сигнал LONG для {symbol}!")
                            logger.info(f"   Вход: ~{df_with_indicators_raw['close'].iloc[-1]:.8f}")
                            logger.info(f"   TP1: {tp1:.8f}")
                            logger.info(f"   TP2: {tp2:.8f}")
                            logger.info(f"   TP3: {tp3:.8f}")
                            logger.info(f"   SL: {sl:.8f}")
                            
                            # --- Открытие сделки (логическое) ---
                            entry_price = df_with_indicators_raw['close'].iloc[-1]
                            active_trades[symbol] = {
                                'side': side,
                                'entry_price': entry_price,
                                'tp1': tp1,
                                'tp2': tp2,
                                'tp3': tp3,
                                'sl': sl,
                                'entry_time': datetime.now(timezone.utc).isoformat(),
                                'tp1_hit': False,
                                'tp2_hit': False
                            }
                            logger.warning(f"🚨 [{symbol}] СДЕЛКА ОТКРЫТА и добавлена в список отслеживания.")
                            save_bot_state()
                        else:
                            logger.warning(f"⚠️ Сигнал для {symbol}, но не удалось рассчитать TP/SL.")
                        
            except Exception as e:
                logger.error(f"Ошибка обработки {symbol}: {e}")
        
        # --- 3. Ожидание до следующей итерации ---
        cycle_elapsed = time.time() - cycle_start_time
        sleep_time = max(monitoring_interval - cycle_elapsed, 0)
        logger.info(f"<<< Итерация завершена за {cycle_elapsed:.2f}с. Ожидание {sleep_time:.2f}с до следующей итерации.")
        if sleep_time > 0:
            time.sleep(sleep_time)

if __name__ == '__main__':
    # Логгер уже настроен в utils/logger.py
    main()
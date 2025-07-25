# signals/signal_generator.py
"""
Новый сигнальный генератор на основе ансамбля моделей.
"""
import sys
import os

# Добавляем корневую директорию проекта в путь поиска модулей
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names*",
    category=UserWarning
)

import glob
import os
import yaml
import pandas as pd
import joblib
import numpy as np
import ccxt
from datetime import datetime, timezone

# Импорты из наших модулей
from features.multi_timeframe import get_multi_timeframe_features
from features.feature_engineering import create_final_features

def load_config(config_path='config/config.yaml'):
    """Загрузить конфигурацию из YAML файла."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_latest_ensemble_model(model_dir="models/ensemble"):
    """
    Ищет в папке 'models/ensemble' файлы *_model_*.pkl
    и возвращает словарь моделей + список фич из самого свежего.
    """
    pattern = "*_model_*.pkl"
    files = glob.glob(os.path.join(model_dir, pattern))
    if not files:
        raise FileNotFoundError(f"В папке '{model_dir}' нет ни одного файла '{pattern}'")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading latest ensemble model: {latest}")
    mdl = joblib.load(latest)
    return mdl["model"], mdl["features"] # mdl["model"] is a dict of models

def get_current_features(exchange, symbol, timeframes, base_tf):
    """
    Собирает текущие фичи для последней свечи символа.
    Возвращает DataFrame с одной строкой.
    """
    try:
        # 1. Получить многотаймфреймовые данные (последние N свечей)
        multi_tf_data = get_multi_timeframe_features(exchange, symbol, timeframes, base_tf)
        
        if multi_tf_data.empty:
            print(f"Нет данных для {symbol}")
            return pd.DataFrame() # Возвращаем пустой DataFrame
            
        # 2. Добавить расширенные признаки
        df_with_features = create_final_features(multi_tf_data)
        
        if df_with_features.empty:
            print(f"Не удалось создать признаки для {symbol}")
            return pd.DataFrame()
            
        # 3. Взять последнюю строку (последняя свеча)
        latest_features = df_with_features.iloc[[-1]].copy() # Используем двойные скобки для DataFrame
        latest_features['symbol'] = symbol # Добавляем символ
        
        return latest_features
        
    except Exception as e:
        print(f"Ошибка при получении признаков для {symbol}: {e}")
        return pd.DataFrame() # Возвращаем пустой DataFrame в случае ошибки

def ensemble_predict_proba(models_dict, X):
    """
    Предсказать вероятность с помощью ансамбля.
    """
    predictions_proba = []
    for name, model in models_dict.items():
        try:
            proba = model.predict_proba(X)[:, 1]
            predictions_proba.append(proba)
        except Exception as e:
            print(f"Ошибка предсказания моделью {name}: {e}")
            # В случае ошибки добавляем 0.5 (нейтральная вероятность)
            predictions_proba.append(np.full(len(X), 0.5))
            
    # Среднее арифметическое вероятностей (soft voting)
    if predictions_proba:
        ensemble_proba = np.mean(predictions_proba, axis=0)
        return ensemble_proba
    else:
        return np.full(len(X), 0.5) # Если все модели дали сбой

def main():
    # 1) Загружаем конфигурацию
    config = load_config()
    symbols = config['symbols']
    timeframes = config['timeframes']['multi']
    base_tf = config['timeframes']['base']
    threshold = config['model']['filter_threshold']
    exchange_config = config['exchange']

    # 2) Инициализация биржи
    exchange = getattr(ccxt, exchange_config['name'])({
        'rateLimit': exchange_config['rateLimit'],
        'enableRateLimit': exchange_config['enableRateLimit'],
        'options': exchange_config['options'],
    })

    # 3) Загружаем ансамбль моделей и фичи
    try:
        models_dict, features = load_latest_ensemble_model()
        print(f"Загружены модели: {list(models_dict.keys())}")
        print(f"Количество признаков в модели: {len(features)}")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # 4) Для каждого символа получаем текущие признаки и делаем предсказание
    signals = []
    for symbol in symbols:
        print(f"→ Анализ {symbol}...")
        
        # Получаем текущие признаки
        current_data = get_current_features(exchange, symbol, timeframes, base_tf)
        
        if current_data.empty:
            print(f"   Пропущен {symbol}: нет данных.")
            continue
            
        # Проверяем, что все необходимые признаки присутствуют
        missing_features = [f for f in features if f not in current_data.columns]
        if missing_features:
            print(f"   Пропущен {symbol}: отсутствуют признаки {missing_features}")
            continue
            
        # Подготавливаем DataFrame с нужными признаками в правильном порядке
        X_new = current_data[features].copy()
        
        # Предсказываем вероятность
        try:
            prob = ensemble_predict_proba(models_dict, X_new)[0] # Берем первую (и единственную) вероятность
        except Exception as e:
            print(f"   Ошибка предсказания для {symbol}: {e}")
            continue
            
        now = datetime.now(timezone.utc).isoformat()
        signal_status = "✅ SIGNAL" if prob > threshold else "❌ NO SIGNAL"
        
        print(f"   {now} ▶ prob = {prob:.4f} (thr={threshold:.4f}) {signal_status}")
        
        # Сохраняем сигнал
        signals.append({
            'timestamp': now,
            'symbol': symbol,
            'probability': prob,
            'threshold': threshold,
            'signal': 1 if prob > threshold else 0
        })
        
    # 5) Выводим все сигналы
    if signals:
        signals_df = pd.DataFrame(signals)
        print("\n--- Все сигналы ---")
        print(signals_df.to_string(index=False))
        
        # Сохраняем сигналы в файл
        signals_file = f"signals/signals_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
        os.makedirs(os.path.dirname(signals_file), exist_ok=True)
        signals_df.to_csv(signals_file, index=False)
        print(f"\nСигналы сохранены в {signals_file}")
    else:
        print("\nНет сигналов для вывода.")

if __name__ == "__main__":
    main()
# models/train_ensemble_model.py
"""
Скрипт для обучения ансамбля моделей на расширенном датасете.
"""
import sys
import os

# Добавляем корневую директорию проекта в путь поиска модулей
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import warnings
import logging
from datetime import datetime, timezone

import pandas as pd
import joblib
import numpy as np
import yaml
import json

# Импорты моделей
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# LightGBM
import lightgbm as lgb

# XGBoost
import xgboost as xgb

# CatBoost
from catboost import CatBoostClassifier

# Подавляем предупреждения
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

def load_config(config_path='config/config.yaml'):
    """Загрузить конфигурацию из YAML файла."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(file_path):
    """Загрузить данные из CSV."""
    df = pd.read_csv(file_path)
    # Проверка и преобразование 'entry_time'
    if 'entry_time' not in df.columns:
        raise ValueError(f"Колонка 'entry_time' отсутствует в {file_path}. Она необходима для TimeSeriesSplit.")
    
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    df.sort_values(by=['symbol', 'entry_time'], inplace=True)
    print(f"Итоговые данные для обучения: {len(df)} строк.")
    print("Распределение меток в исходных данных: ")
    print(df['label'].value_counts())
    return df

def prepare_features_and_target(df):
    """Подготовить признаки (X) и целевую переменную (y)."""
    # Исключаем нечисловые колонки и целевую переменную 'label' из признаков X
    X_cols_to_drop = ['entry_time', 'symbol', 'side', 'label']
    X = df.drop(columns=[col for col in X_cols_to_drop if col in df.columns], errors='ignore')
    y = df['label']
    return X, y

def select_features(X, threshold=0.0):
    """Отбор признаков с помощью VarianceThreshold."""
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    selected_feature_names = X.columns[selector.get_support()].tolist()
    print(f"Выбрано {len(selected_feature_names)} признаков после VarianceThreshold.")
    # Преобразуем обратно в DataFrame
    X = pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)
    return X, selected_feature_names, selector

def train_single_model(model_name, model, X_train, y_train, X_test, y_test):
    """Обучить одну модель и вернуть предсказания и модель."""
    print(f"Обучение модели {model_name}...")
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return model, y_pred_proba, y_pred

def evaluate_model(y_test, y_pred, y_pred_proba, model_name):
    """Оценить модель и вернуть метрики."""
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n--- Оценка модели {model_name} ---")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    return {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def create_ensemble(models_dict, X_train, y_train, X_test, y_test):
    """Создать ансамбль моделей с мягким голосованием."""
    print("\n--- Создание ансамбля ---")
    
    # Обучаем все модели и собираем их предсказания
    trained_models = {}
    predictions_proba = {}
    
    for name, model in models_dict.items():
        print(f"Обучение {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        predictions_proba[name] = model.predict_proba(X_test)[:, 1]
        
    # Среднее арифметическое вероятностей (soft voting)
    ensemble_proba = np.mean(list(predictions_proba.values()), axis=0)
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    # Оценка ансамбля
    metrics = evaluate_model(y_test, ensemble_pred, ensemble_proba, "Ensemble (Soft Voting)")
    
    return trained_models, ensemble_proba, ensemble_pred, metrics

def find_optimal_threshold(y_test, y_pred_proba):
    """Найти оптимальный порог по метрике F1."""
    from sklearn.metrics import f1_score
    thresholds = np.arange(0.1, 1.0, 0.01)
    f1_scores = [f1_score(y_test, (y_pred_proba >= t).astype(int), zero_division=0) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Оптимальный порог (максимизация F1): {optimal_threshold:.4f} (F1={f1_scores[optimal_idx]:.4f})")
    return optimal_threshold

def save_model_and_config(model, features, threshold, model_name, config_path):
    """Сохранить модель, признаки и обновить конфиг."""
    # Создаем директорию для модели, если её нет
    model_dir = os.path.join('models', model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Имя файла модели
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    model_filename = os.path.join(model_dir, f"{model_name}_model_{timestamp}.pkl")
    
    # Сохраняем модель
    joblib.dump({
        "model": model,
        "features": features,
        "timestamp": timestamp
    }, model_filename)
    print(f"✅ Модель сохранена в {model_filename}")
    
    # Обновляем config.yaml
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        cfg = {} # Если файл не найден или ошибка, начинаем с пустого конфига

    cfg['model']['filter_threshold'] = float(f"{threshold:.4f}")
    
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"Обновлен filter_threshold в {config_path} до {cfg['model']['filter_threshold']}.")

def main():
    # 1) Загрузка конфигурации
    config = load_config()
    config_path = 'config/config.yaml'
    data_file = 'enhanced_training_dataset.csv'
    model_name = 'ensemble'
    
    # 2) Загрузка данных
    df = load_data(data_file)
    
    # 3) Подготовка признаков и целевой переменной
    X, y = prepare_features_and_target(df)
    
    # 4) Отбор признаков
    X, selected_features, selector = select_features(X)
    
    # 5) Подготовка кросс-валидации
    tscv = TimeSeriesSplit(n_splits=config['model']['cv_folds'])
    
    # 6) Определение моделей
    models = {
        'LightGBM': lgb.LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31, verbose=-1),
        'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=6, verbosity=0),
        'CatBoost': CatBoostClassifier(random_state=42, iterations=500, learning_rate=0.05, depth=6, verbose=False),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10),
    }
    
    # Для хранения метрик по фолдам
    all_fold_metrics = {name: [] for name in models.keys()}
    all_fold_metrics['Ensemble'] = []
    all_optimal_thresholds = []
    
    print("\n--- Начало кросс-валидации ---")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- ФОЛД {fold + 1} ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Размер обучающего набора: {len(X_train)} строк, размер тестового набора: {len(X_test)} строк.")
        
        # Обучаем и оцениваем каждую модель
        trained_models = {}
        model_predictions_proba = {}
        
        for name, model in models.items():
            trained_model, y_pred_proba, y_pred = train_single_model(name, model, X_train, y_train, X_test, y_test)
            trained_models[name] = trained_model
            model_predictions_proba[name] = y_pred_proba
            
            metrics = evaluate_model(y_test, y_pred, y_pred_proba, name)
            all_fold_metrics[name].append(metrics)
            
        # Создаем ансамбль для этого фолда
        ensemble_proba = np.mean(list(model_predictions_proba.values()), axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        ensemble_metrics = evaluate_model(y_test, ensemble_pred, ensemble_proba, f"Ensemble_Fold_{fold+1}")
        all_fold_metrics['Ensemble'].append(ensemble_metrics)
        
        # Находим оптимальный порог для ансамбля этого фолда
        opt_thr = find_optimal_threshold(y_test, ensemble_proba)
        all_optimal_thresholds.append(opt_thr)
        
    # 7) Агрегация метрик
    print("\n--- Агрегированные метрики по всем фолдам ---")
    avg_metrics = {}
    for model_name_key, metrics_list in all_fold_metrics.items():
        if metrics_list:
            avg_roc_auc = np.mean([m['roc_auc'] for m in metrics_list])
            avg_f1 = np.mean([m['f1'] for m in metrics_list])
            avg_precision = np.mean([m['precision'] for m in metrics_list])
            avg_recall = np.mean([m['recall'] for m in metrics_list])
            print(f"{model_name_key}:")
            print(f"  Средний ROC AUC: {avg_roc_auc:.4f}")
            print(f"  Средний F1-score: {avg_f1:.4f}")
            print(f"  Средний Precision: {avg_precision:.4f}")
            print(f"  Средний Recall: {avg_recall:.4f}")
            avg_metrics[model_name_key] = {
                'roc_auc': avg_roc_auc,
                'f1': avg_f1,
                'precision': avg_precision,
                'recall': avg_recall
            }
    
    # 8) Финальное обучение на всех данных
    print("\n--- Финальное обучение моделей на всех данных ---")
    final_trained_models = {}
    for name, model in models.items():
        print(f"Обучение финальной модели {name}...")
        model.fit(X, y)
        final_trained_models[name] = model
        
    # Создаем финальный ансамбль
    print("Создание финального ансамбля...")
    final_ensemble_proba = np.mean([model.predict_proba(X)[:, 1] for model in final_trained_models.values()], axis=0)
    final_ensemble_pred = (final_ensemble_proba > 0.5).astype(int)
    
    # Оценка финального ансамбля
    final_metrics = evaluate_model(y, final_ensemble_pred, final_ensemble_proba, "Final_Ensemble")
    
    # Найти финальный оптимальный порог
    final_optimal_threshold = find_optimal_threshold(y, final_ensemble_proba)
    
    # 9) Сохранение финального ансамбля (все модели)
    save_model_and_config(final_trained_models, selected_features, final_optimal_threshold, model_name, config_path)
    
    print("\n--- Обучение завершено ---")

if __name__ == "__main__":
    main()
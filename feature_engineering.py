# feature_engineering.py
"""
Адаптированный скрипт для создания обучающего датасета на основе новых, расширенных данных.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os

def main():
    # Пути к файлам
    expert_trades_file = 'expert_trades.csv' # Файл с идеальными сделками
    labeled_trades_file = 'labeled_trades.csv' # Файл с размеченными сделками бота
    enhanced_history_file = 'data/enhanced_full_history_with_indicators.csv' # Новый файл с расширенными данными
    output_training_file = 'enhanced_training_dataset.csv' # Выходной файл
    
    # 1) Загрузим размеченные сделки бота
    try:
        labels_bot = pd.read_csv(labeled_trades_file, dtype={'symbol': str})
        labels_bot['entry_time'] = pd.to_datetime(labels_bot['entry_time'], errors='coerce', utc=True)
        labels_bot['exit_time'] = pd.to_datetime(labels_bot['exit_time'], errors='coerce', utc=True)
        labels_bot['source'] = 'bot'
        print(f"Загружено {len(labels_bot)} размеченных сделок бота.")
    except Exception as e:
        print(f"Ошибка при загрузке {labeled_trades_file}: {e}")
        labels_bot = pd.DataFrame() # Пустой DataFrame если файл не найден или ошибка

    # 2) Загрузим экспертные сделки, если файл существует
    labels = labels_bot.copy()
    if os.path.exists(expert_trades_file) and os.path.getsize(expert_trades_file) > 0:
        try:
            labels_expert = pd.read_csv(expert_trades_file, dtype={'symbol': str})
            labels_expert['entry_time'] = pd.to_datetime(labels_expert['entry_time'], errors='coerce', utc=True)
            labels_expert['exit_time'] = pd.to_datetime(labels_expert['exit_time'], errors='coerce', utc=True)
            labels_expert['source'] = 'expert'
            # Приоритет у expert-меток: объединяем, оставляя expert в случае дубликатов
            labels = pd.concat([labels_bot, labels_expert]).drop_duplicates(subset=['symbol', 'entry_time', 'side'], keep='first')
            print(f"Загружено {len(labels_expert)} экспертных сделок. Итого после объединения: {len(labels)}")
        except Exception as e:
            print(f"Ошибка при загрузке {expert_trades_file}: {e}. Используем только метки бота.")
    else:
        print("Файл expert_trades.csv не найден или пуст. Используем только метки бота.")

    if labels.empty:
        print("Нет размеченных сделок для обработки. Выход.")
        # Создаем пустой файл
        pd.DataFrame(columns=[]).to_csv(output_training_file, index=False)
        return

    # 3) Загрузим полную историю с расширенными индикаторами
    try:
        full_history = pd.read_csv(enhanced_history_file, dtype={'symbol': str})
        full_history['timestamp'] = pd.to_datetime(full_history['timestamp'], errors='coerce', utc=True)
        full_history.dropna(subset=['timestamp'], inplace=True)
        full_history.set_index('timestamp', inplace=True)
        full_history.sort_index(inplace=True)
        print(f"Загружено {len(full_history)} строк полной истории с расширенными индикаторами.")
    except Exception as e:
        print(f"Ошибка при загрузке {enhanced_history_file}: {e}")
        print("Выход.")
        # Создаем пустой файл
        pd.DataFrame(columns=[]).to_csv(output_training_file, index=False)
        return

    print("Объединение размеченных сделок с историческими данными...")
    
    # Список для хранения результатов
    training_data_rows = []
    
    # 4) Для каждой размеченной сделки извлекаем данные индикаторов
    all_symbols = labels['symbol'].unique()
    for sym in all_symbols:
        labels_sym = labels[labels['symbol'] == sym].sort_values('entry_time')
        history_sym = full_history[full_history['symbol'] == sym].sort_index()
        
        if history_sym.empty:
            print(f"   Пропущен символ {sym}: нет исторических данных.")
            continue
            
        # Используем merge_asof для поиска ближайшей свечи перед entry_time
        # join_time - это timestamp свечи, которую мы хотим использовать
        labels_sym_sorted = labels_sym.sort_values('entry_time')
        history_sym_sorted = history_sym.sort_index()
        
        # merge_asof требует, чтобы оба DataFrame были отсортированы по ключевой колонке
        merged_df = pd.merge_asof(
            labels_sym_sorted,
            history_sym_sorted,
            left_on='entry_time',
            right_index=True,
            by='symbol',
            direction='backward', # Ищем свечу, которая закончилась до entry_time
            tolerance=pd.Timedelta('10s') # Небольшой допуск
        )
        
        # Отфильтровываем случаи, когда не удалось найти соответствующую свечу
        merged_df.dropna(subset=history_sym_sorted.columns, how='all', inplace=True)
        
        # Добавляем данные в список
        if not merged_df.empty:
            training_data_rows.append(merged_df)
            
    if not training_data_rows:
        print("Не удалось объединить ни одну размеченную сделку с историческими данными. Выход.")
        # Создаем пустой файл с заголовками
        # Определяем все возможные колонки, которые должны быть в training_dataset.csv
        all_cols = list(full_history.columns) + list(labels.columns) 
        pd.DataFrame(columns=all_cols).to_csv(output_training_file, index=False)
        return
        
    # 5) Объединяем все результаты
    combined_df = pd.concat(training_data_rows, ignore_index=True)
    print(f"Объединено {len(combined_df)} строк для обучения.")
    
    # Определяем колонки, которые не нужны для признаков
    columns_to_exclude = [
        'open', 'high', 'low', 'close', 'volume', # Исключаем OHLCV, если уже есть индикаторы
        'symbol', 'join_time', 'side', 'entry_time', 'exit_time', 
        'entry_price', 'exit_price', 'reason', 'pnl', 'source'
    ]
    
    # Сохраняем entry_time, symbol и side отдельно для последующего объединения с X и y
    entry_times_to_save = combined_df['entry_time']
    symbols_to_save = combined_df['symbol']
    sides_to_save = combined_df['side']
    
    # Удаляем колонки, которые не нужны для признаков, а также целевую переменную 'label'
    X = combined_df.drop(columns=[col for col in columns_to_exclude if col in combined_df.columns] + ['label'], errors='ignore')
    y = combined_df['label']
    
    # 6) Обрабатываем бесконечные значения и большие числа
    print(f"До обработки inf/NaN: {len(X)} строк.")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 7) Удаляем признаки с большим количеством NaN (если необходимо)
    initial_features_count = X.shape[1]
    nan_threshold = 0.8 # Если более 80% значений NaN
    cols_to_drop_nan = X.columns[X.isnull().sum() / len(X) > nan_threshold].tolist()
    if cols_to_drop_nan:
        X.drop(columns=cols_to_drop_nan, inplace=True)
        print(f"Удалены признаки с более чем {nan_threshold*100}% NaN: {cols_to_drop_nan}")
    print(f"После удаления признаков с большим количеством NaN: {X.shape[1]} признаков.")
    
    # 8) Заполняем оставшиеся NaN медианой
    # Важно: медиана должна быть рассчитана на обучающем наборе, но для простоты здесь заполняем глобально
    # Для продакшена лучше использовать SimpleImputer в пайплайне
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            if not pd.isna(median_val): # Проверяем, что медиана не NaN
                X[col].fillna(median_val, inplace=True)
            else:
                # Если колонка полностью NaN, заполняем 0 или удаляем
                print(f"Предупреждение: Колонка '{col}' полностью NaN после фильтрации. Заполнение 0.")
                X[col].fillna(0, inplace=True) # Заполняем 0, если вся колонка NaN
                
    print(f"После заполнения NaN: {len(X)} строк.")
    
    # 9) Финальная очистка от любых оставшихся NaN/inf (должны быть уже обработаны)
    before_dropna_final = len(X)
    # Создаем DataFrame для final dropna, включая y
    combined_for_dropna = pd.concat([X, y], axis=1) # Используем y как целевую переменную
    combined_for_dropna = combined_for_dropna.dropna()
    # Разделяем обратно
    X = combined_for_dropna.drop(columns=['label'])
    y = combined_for_dropna['label']
    after_dropna_final = len(X)
    print(f"Финальная очистка NaN/inf: удалено {before_dropna_final - after_dropna_final} строк.")
    
    # Объединяем X, y, entry_time, symbol и side в один DataFrame для сохранения
    training_df = pd.concat([X, y, entry_times_to_save, symbols_to_save, sides_to_save], axis=1)
    
    # 10) Сохраняем итоговый файл
    if not training_df.empty:
        training_df.to_csv(output_training_file, index=False)
        print(f"Обучающий датасет сохранен в {output_training_file}. Всего: {len(training_df)} строк.")
        print(f"Распределение меток:\n{training_df['label'].value_counts()}")
        print(f"Использованные признаки (X): {X.columns.tolist()}")
    else:
        # Если DataFrame пуст, создаем пустой файл с заголовками
        print(f"Обучающий датасет пуст. Создаю пустой файл {output_training_file}.")
        # Определяем все возможные колонки, которые должны быть в training_dataset.csv
        all_cols = X.columns.tolist() + ['label', 'entry_time', 'symbol', 'side']
        pd.DataFrame(columns=all_cols).to_csv(output_training_file, index=False)
        
if __name__ == '__main__':
    main()
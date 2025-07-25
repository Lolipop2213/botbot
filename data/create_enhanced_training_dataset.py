# data/create_enhanced_training_dataset.py
"""
Адаптированный скрипт для создания обучающего датасета на основе новых, расширенных данных.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os # Добавлен импорт

def main():
    # Пути к файлам
    expert_trades_file = 'expert_trades.csv' # Файл с идеальными сделками
    labeled_trades_file = 'labeled_trades.csv' # Файл с размеченными сделками бота
    # ИСПРАВЛЕНО: Путь к файлу с историей должен соответствовать тому, куда его сохраняет generate_enhanced_history.py
    enhanced_history_file = 'data/enhanced/enhanced_full_history_with_indicators.csv' # Новый файл с расширенными данными
    # ИСПРАВЛЕНО: Путь к выходному файлу с './' в начале, чтобы dirname не был пустым
    output_training_file = './enhanced_training_dataset.csv' # Выходной файл
    
    print(f"Попытка загрузить {labeled_trades_file}...")
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
            print(f"Попытка загрузить {expert_trades_file}...")
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
    print(f"Попытка загрузить {enhanced_history_file}...")
    try:
        # Убедимся, что файл существует
        if not os.path.exists(enhanced_history_file):
            raise FileNotFoundError(f"Файл {enhanced_history_file} не найден.")
            
        full_history = pd.read_csv(enhanced_history_file, dtype={'symbol': str})
        full_history['timestamp'] = pd.to_datetime(full_history['timestamp'], errors='coerce', utc=True)
        # Удаляем строки с некорректными датами
        full_history.dropna(subset=['timestamp'], inplace=True) 
        # Устанавливаем timestamp как индекс для merge_asof
        full_history.set_index('timestamp', inplace=True) 
        full_history.sort_index(inplace=True)
        print(f"Загружено {len(full_history)} строк полной истории с расширенными индикаторами.")
        print(f"  Диапазон времени истории: {full_history.index.min()} - {full_history.index.max()}")
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
    print(f"Найдено {len(all_symbols)} уникальных символов в размеченных сделках.")
    
    for i, sym in enumerate(all_symbols):
        print(f"  Обработка символа {i+1}/{len(all_symbols)}: {sym}")
        labels_sym = labels[labels['symbol'] == sym].sort_values('entry_time')
        history_sym = full_history[full_history['symbol'] == sym].sort_index()
        
        if history_sym.empty:
            print(f"    Пропущен символ {sym}: нет исторических данных.")
            continue
            
        print(f"    Найдено {len(labels_sym)} сделок и {len(history_sym)} свечей для {sym}.")
        print(f"    Диапазон времени сделок: {labels_sym['entry_time'].min()} - {labels_sym['entry_time'].max()}")
        print(f"    Диапазон времени истории: {history_sym.index.min()} - {history_sym.index.max()}")
        
        # Проверим, попадают ли сделки в диапазон истории
        earliest_trade = labels_sym['entry_time'].min()
        latest_history = history_sym.index.max()
        if earliest_trade > latest_history:
            print(f"    ⚠️  ВНИМАНИЕ: Самая ранняя сделка ({earliest_trade}) позже последней свечи истории ({latest_history}) для {sym}.")
            
        # Используем merge_asof для поиска ближайшей свечи перед entry_time
        labels_sym_sorted = labels_sym.sort_values('entry_time').reset_index(drop=True)
        history_sym_sorted = history_sym.sort_index()
        
        # --- ОТЛАДКА: Проверим форматы времени ---
        print(f"    Формат времени первой сделки: {type(labels_sym_sorted['entry_time'].iloc[0])}")
        print(f"    Формат времени первой свечи: {type(history_sym_sorted.index[0])}")
        # --- КОНЕЦ ОТЛАДКИ ---
        
        # merge_asof требует, чтобы оба DataFrame были отсортированы по ключевой колонке
        try:
            merged_df = pd.merge_asof(
                labels_sym_sorted,
                history_sym_sorted,
                left_on='entry_time',
                right_index=True, # Правый DataFrame уже имеет timestamp в индексе
                by='symbol',
                direction='backward', # Ищем свечу, которая закончилась до entry_time
                tolerance=pd.Timedelta('1min') # Увеличиваем допуск до 1 минуты
            )
            print(f"    merge_asof завершен. Результат: {merged_df.shape}")
        except Exception as e:
            print(f"    ❌ Ошибка merge_asof для {sym}: {e}")
            continue
            
        # Отфильтровываем случаи, когда не удалось найти соответствующую свечу
        # Проверяем, есть ли в объединенном DataFrame данные из истории (не только из labels)
        # Мы можем проверить любую колонку из истории, например, 'close'
        initial_rows = len(merged_df)
        if 'close' in merged_df.columns:
            # Удаляем строки, где 'close' (признак из истории) является NaN
            merged_df.dropna(subset=['close'], inplace=True) 
            rows_after_dropna = len(merged_df)
            print(f"    Удалено {initial_rows - rows_after_dropna} строк из-за отсутствия данных в истории (NaN в 'close'). Осталось: {rows_after_dropna}")
        else:
            print(f"    ⚠️  ВНИМАНИЕ: Колонка 'close' не найдена в результате merge_asof для {sym}.")
            merged_df = pd.DataFrame() # Если нет ключевых колонок, считаем результат пустым
            
        # Добавляем данные в список
        if not merged_df.empty:
            print(f"    ✅ Добавлено {len(merged_df)} строк для обучения по {sym}.")
            training_data_rows.append(merged_df)
        else:
            print(f"    ❌ Не удалось добавить данные для {sym}.")
            
    if not training_data_rows:
        print("Не удалось объединить ни одну размеченную сделку с историческими данными. Выход.")
        # Создаем пустой файл с заголовками
        # Определяем все возможные колонки, которые должны быть в training_dataset.csv
        # Берем колонки из full_history и labels как пример
        all_cols_from_history = list(full_history.columns) if not full_history.empty else []
        all_cols_from_labels = list(labels.columns) if not labels.empty else []
        all_cols = list(set(all_cols_from_history + all_cols_from_labels))
        pd.DataFrame(columns=all_cols).to_csv(output_training_file, index=False)
        return
        
    # 5) Объединяем все результаты
    combined_df = pd.concat(training_data_rows, ignore_index=True)
    print(f"Объединено {len(combined_df)} строк для обучения.")
    
    if combined_df.empty:
         print("Итоговый DataFrame пуст. Выход.")
         pd.DataFrame(columns=[]).to_csv(output_training_file, index=False)
         return
         
    # Определяем колонки, которые не нужны для признаков
    # ИСПРАВЛЕНО: Уточнен список исключений
    columns_to_exclude = [
        'open', 'high', 'low', 'close', 'volume', # Исключаем базовые OHLCV, если уже есть индикаторы
        'symbol', 'side', 'entry_time', 'exit_time', 
        'entry_price', 'exit_price', 'reason', 'pnl', 'source'
        # 'join_time' убран, так как его нет в новых данных
    ]
    
    # Сохраняем entry_time, symbol и side отдельно для последующего объединения с X и y
    # ИСПРАВЛЕНО: Делаем копию до любых dropna
    meta_data_to_save = combined_df[['entry_time', 'symbol', 'side']].copy()
    
    # Удаляем колонки, которые не нужны для признаков, а также целевую переменную 'label'
    X = combined_df.drop(columns=[col for col in columns_to_exclude if col in combined_df.columns] + ['label'], errors='ignore')
    y = combined_df['label']
    
    print(f"Форма признаков (X) до обработки NaN: {X.shape}")
    print(f"Распределение меток (y):")
    print(y.value_counts(dropna=False))
    
    # 6) Обрабатываем бесконечные значения и большие числа
    print(f"До обработки inf/NaN: {len(X)} строк.")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 7) Удаляем признаки с большим количеством NaN (если необходимо)
    initial_features_count = X.shape[1]
    nan_threshold = 0.95 # Если более 95% значений NaN
    cols_to_drop_nan = X.columns[X.isnull().sum() / len(X) > nan_threshold].tolist()
    if cols_to_drop_nan:
        X.drop(columns=cols_to_drop_nan, inplace=True)
        print(f"Удалены признаки с более чем {nan_threshold*100}% NaN: {cols_to_drop_nan}")
    print(f"После удаления признаков с большим количеством NaN: {X.shape[1]} признаков.")
    
    # 8) Заполняем оставшиеся NaN медианой
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            if not pd.isna(median_val): # Проверяем, что медиана не NaN
                # ИСПРАВЛЕНО: Используем .loc для заполнения
                X[col].fillna(median_val, inplace=True)
            else:
                # Если колонка полностью NaN, заполняем 0
                print(f"Предупреждение: Колонка '{col}' полностью NaN. Заполнение 0.")
                X[col].fillna(0, inplace=True) # Заполняем 0, если вся колонка NaN
                
    print(f"После заполнения NaN: {len(X)} строк.")
    
    # 9) Финальная очистка от любых оставшихся NaN/inf
    before_dropna_final = len(X)
    # Создаем DataFrame для final dropna, включая y
    combined_for_dropna = pd.concat([X, y], axis=1) 
    combined_for_dropna = combined_for_dropna.dropna()
    # Разделяем обратно
    X = combined_for_dropna.drop(columns=['label'])
    y = combined_for_dropna['label']
    after_dropna_final = len(X)
    print(f"Финальная очистка NaN/inf: удалено {before_dropna_final - after_dropna_final} строк.")
    
    # Проверка на пустоту после финальной очистки
    if X.empty or y.empty:
         print("X или y стали пустыми после финальной очистки. Выход.")
         pd.DataFrame(columns=[]).to_csv(output_training_file, index=False)
         return
         
    # Объединяем X, y и мета-данные в один DataFrame для сохранения
    # ИСПРАВЛЕНО: Используем meta_data_to_save, отфильтрованное по индексу финального X
    training_df = pd.concat([X, y, meta_data_to_save.loc[X.index]], axis=1)
    
    # 10) Сохраняем итоговый файл
    if not training_df.empty:
        # ИСПРАВЛЕНО: Проверка перед созданием директории
        output_dir = os.path.dirname(output_training_file)
        if output_dir: # Если путь не пустой (например, './file.csv' -> dirname='./')
            os.makedirs(output_dir, exist_ok=True)
        # Если путь пустой (например, 'file.csv'), os.makedirs не вызывается
        
        training_df.to_csv(output_training_file, index=False)
        print(f"Обучающий датасет сохранен в {output_training_file}. Всего: {len(training_df)} строк.")
        print(f"Распределение меток:\n{training_df['label'].value_counts()}")
        print(f"Использованные признаки (X): {len(X.columns)} колонок")
        # print(f"Использованные признаки (X): {X.columns.tolist()}") # Опционально, для детального списка
    else:
        # Если DataFrame пуст, создаем пустой файл с заголовками
        print(f"Обучающий датасет пуст. Создаю пустой файл {output_training_file}.")
        # Определяем все возможные колонки
        all_cols = X.columns.tolist() + ['label', 'entry_time', 'symbol', 'side'] 
        pd.DataFrame(columns=all_cols).to_csv(output_training_file, index=False)
        
if __name__ == '__main__':
    main()

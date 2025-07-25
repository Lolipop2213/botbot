# run_full_pipeline.py
"""
Основной скрипт для запуска всего пайплайна: от генерации данных до получения сигналов.
"""
import subprocess
import sys
import os
from datetime import datetime

# Импортируем логгер для пайплайна
from utils.logger import pipeline_logger as logger

def run_script(script_name):
    """Запустить скрипт и дождаться его завершения."""
    logger.info(f"{datetime.now().strftime('%H:%M:%S')} → Запуск {script_name}")
    try:
        # Используем sys.executable для совместимости
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        logger.info(f"✓ {script_name} завершен успешно.")
        # Логируем stdout скрипта на уровне DEBUG
        if result.stdout.strip():
            logger.debug(f"Вывод {script_name}:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Ошибка в {script_name}: код возврата {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ Неожиданная ошибка при запуске {script_name}: {e}")
        return False

def main():
    logger.info("🚀 Запуск полного пайплайна Super Trading Bot")
    logger.info("=" * 50)
    
    # Определяем порядок выполнения скриптов
    # Предполагаем, что они находятся в подпапках
    pipeline_steps = [
        "data/generate_enhanced_history.py",              # 1. Генерация расширенных исторических данных
        "data_generation/generate_ideal_labels.py",       # 2. Генерация идеальных меток
        "data_generation/label_trades.py",                # 3. Парсинг логов бота
        "data/create_enhanced_training_dataset.py",       # 4. Создание обучающего датасета
        "models/train_ensemble_model.py",                 # 5. Обучение ансамбля моделей
        # "monitoring_bot.py"                             # 6. Запуск мониторинга (обычно запускается отдельно)
    ]
    
    # Выполняем каждый шаг
    for step in pipeline_steps:
        if not os.path.exists(step):
            logger.error(f"✗ Файл {step} не найден. Пропуск.")
            continue
            
        success = run_script(step)
        if not success:
            logger.critical("❌ Пайплайн остановлен из-за ошибки.")
            sys.exit(1)
        logger.info("-" * 30)
        
    logger.info(f"\n✅ {datetime.now().strftime('%H:%M:%S')} Весь пайплайн выполнен успешно!")

if __name__ == "__main__":
    main()
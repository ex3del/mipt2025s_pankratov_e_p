# Отчёт о проделанной работе

## 🔹 Основные выполненные задачи

### 1. Анализ пайплайна классификации и детекции
- Разобран пример реализации пайплайна из репозитория:  
  [mipt2025s-5-modern-cv/detector](https://github.com/dvpsun/mipt2025s-5-modern-cv/tree/main/detector)
- Изучены ключевые компоненты:
  - Подготовка данных
  - Конфигурация модели
  - Процесс обучения и валидации

### 2. Создание Docker-образов
| Образ | Назначение | Особенности |
|-------|------------|-------------|
| **notebook-image** | Для работы с .ipynb тетрадками | Поддержка Jupyter Lab, CUDA |
| **barcode_detector_classifier-image** | Для выполнения .py скриптов | Оптимизирован для production, CUDA |

### 3. Эксперименты с датасетом DOTA8
- Результаты сохранены в:  
  [train_dota](https://github.com/dvpsun/mipt2025s-5-modern-cv/tree/main/runs/obb/train_dota)

### 4. Работа с нашим датасетом
- Результаты сохранены в:  
  [train12_our_data](https://github.com/dvpsun/mipt2025s-5-modern-cv/tree/main/runs/obb/train12_our_data)
- Достигнутые показатели:
  ```python
  mAP50: 0.995
  Class Accuracy:
    ean13: 99.5%
    c39: 98.7%
    qr: 97.3%
    ```
  
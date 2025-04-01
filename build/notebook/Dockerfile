# Используем базовый образ с поддержкой CUDA для работы с видеокартой
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python пакеты
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install numpy opencv-python pyzbar ultralytics inference-sdk
# Устанавливаем Jupyter и ipykernel
RUN pip3 install jupyterlab ipykernel

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app

# Создаем ядро для Jupyter
RUN python3 -m ipykernel install --user --name=myenv

# Открываем порт для Jupyter (обычно 8888)
EXPOSE 8888

# Команда по умолчанию при запуске контейнера
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
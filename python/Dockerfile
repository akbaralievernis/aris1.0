# ===========================================
# ARIS Neuro v3.0 - Python ML Dockerfile
# С поддержкой GPU и оптимизированный
# ===========================================

# Используем базовый образ с Python и CUDA для GPU
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    sox \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем виртуальное окружение
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Обновляем pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Копируем requirements
WORKDIR /app
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем PyTorch с поддержкой CUDA
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Устанавливаем Whisper
RUN pip install --no-cache-dir openai-whisper

# Устанавливаем Coqui TTS для синтеза речи
RUN pip install --no-cache-dir TTS

# Копируем исходный код Python
COPY . .

# Создаем директории для моделей и временных файлов
RUN mkdir -p /app/models /app/temp /app/logs

# Скачиваем предобученные модели
RUN python -c "
import whisper
print('Загрузка модели Whisper...')
model = whisper.load_model('medium')
"

RUN python -c "
from TTS.api import TTS
print('Загрузка модели TTS...')
tts = TTS(model_name='tts_models/ru/ru_ruslan')
"

# Создаем пользователя для безопасности
RUN useradd -m -u 1000 -s /bin/bash mluser
RUN chown -R mluser:mluser /app
USER mluser

# Health check для Python сервиса
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Команда запуска
CMD ["python", "voice_processor.py"]
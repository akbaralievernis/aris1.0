"""
Сервис синтеза речи на базе Coqui TTS и других моделей
Поддержка русского и английского языков, различных голосов
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import torch
import torchaudio
from TTS.api import TTS
import soundfile as sf
import io
import base64
from redis import Redis
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TTSConfig:
    """Конфигурация TTS сервиса"""
    # Модели по умолчанию
    DEFAULT_MODELS = {
        'ru': 'tts_models/ru/ru_ruslan',  # Русский, мужской голос
        'en': 'tts_models/en/vctk/vits',  # Английский, много голосов
        'en-male': 'tts_models/en/vctk/vits',
        'en-female': 'tts_models/en/vctk/vits',
    }
    
    # Поддерживаемые языки
    SUPPORTED_LANGUAGES = ['ru', 'en', 'de', 'fr', 'es', 'it']
    
    # Каталоги
    MODELS_DIR = Path("/app/models/tts")
    TEMP_DIR = Path("/app/temp/tts")
    CACHE_DIR = Path("/app/cache/tts")
    
    # Настройки производительности
    MAX_TEXT_LENGTH = 5000  # Максимальная длина текста в символах
    BATCH_SIZE = 10  # Размер батча для пакетной обработки
    CACHE_TTL = 3600  # Время жизни кэша в секундах
    
    # Настройки аудио
    SAMPLE_RATE = 22050
    AUDIO_FORMAT = "wav"
    BITRATE = "192k"
    
    # Настройки голосов
    VOICES = {
        'ru': {
            'male': ['ruslan'],
            'female': ['natasha', 'tatyana'],
            'neutral': ['aidar']
        },
        'en': {
            'male': ['p225', 'p226', 'p227', 'p228', 'p229'],
            'female': ['p230', 'p231', 'p232', 'p233', 'p234'],
            'neutral': ['p240', 'p241', 'p242']
        }
    }

class TTSService:
    """Сервис синтеза речи"""
    
    def __init__(self, config: TTSConfig = None, use_gpu: bool = True):
        self.config = config or TTSConfig()
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Создаем директории
        self._create_directories()
        
        # Загруженные модели
        self.models: Dict[str, TTS] = {}
        # Кэш синтезированных аудио
        self.audio_cache = {}
        # Пул потоков для обработки
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Redis клиент для распределенного кэширования
        self.redis_client = None
        self._init_redis()
        
        # Статистика
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'characters_processed': 0,
            'total_processing_time': 0,
            'errors': 0
        }
        
        logger.info(f"Инициализация TTSService на устройстве: {self.device}")
        
    def _create_directories(self):
        """Создание необходимых директорий"""
        directories = [self.config.MODELS_DIR, self.config.TEMP_DIR, self.config.CACHE_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Создана директория: {directory}")
    
    def _init_redis(self):
        """Инициализация Redis для кэширования"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = Redis.from_url(redis_url, decode_responses=False)
            # Тестируем подключение
            self.redis_client.ping()
            logger.info("✅ Redis подключен для кэширования TTS")
        except Exception as e:
            logger.warning(f"❌ Не удалось подключиться к Redis: {e}")
            self.redis_client = None
    
    def load_model(self, model_name: str, force_reload: bool = False) -> TTS:
        """
        Загрузка TTS модели
        
        Args:
            model_name: Имя модели в формате TTS
            force_reload: Принудительная перезагрузка модели
            
        Returns:
            Загруженная модель TTS
        """
        if model_name in self.models and not force_reload:
            return self.models[model_name]
        
        try:
            logger.info(f"Загрузка модели: {model_name}")
            
            # Проверяем, есть ли модель локально
            model_path = self.config.MODELS_DIR / model_name.replace("/", "_")
            
            tts = TTS(
                model_name=model_name,
                progress_bar=False,
                gpu=self.use_gpu
            )
            
            self.models[model_name] = tts
            logger.info(f"✅ Модель загружена: {model_name}")
            return tts
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели {model_name}: {e}")
            raise
    
    def get_model_for_language(self, language: str, voice_type: str = "neutral") -> TTS:
        """
        Получение модели для заданного языка и типа голоса
        
        Args:
            language: Код языка (ru, en и т.д.)
            voice_type: Тип голоса (male, female, neutral)
            
        Returns:
            Модель TTS
        """
        if language not in self.config.SUPPORTED_LANGUAGES:
            language = "en"  # Fallback на английский
        
        model_key = self.config.DEFAULT_MODELS.get(language, self.config.DEFAULT_MODELS['en'])
        
        # Для английского выбираем конкретную модель с голосом
        if language == "en" and voice_type in ["male", "female"]:
            model_key = self.config.DEFAULT_MODELS[f'en-{voice_type}']
        
        return self.load_model(model_key)
    
    def _generate_cache_key(self, text: str, language: str, voice: str, 
                           speed: float, pitch: float) -> str:
        """Генерация ключа для кэша"""
        import hashlib
        content = f"{text}_{language}_{voice}_{speed}_{pitch}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """
        Получение аудио из кэша
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            Аудио данные или None
        """
        # Сначала проверяем in-memory кэш
        if cache_key in self.audio_cache:
            cached_data = self.audio_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.config.CACHE_TTL:
                self.stats['cache_hits'] += 1
                return cached_data['audio']
        
        # Затем проверяем Redis
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"tts:{cache_key}")
                if cached:
                    self.stats['cache_hits'] += 1
                    # Сохраняем в in-memory кэш
                    self.audio_cache[cache_key] = {
                        'audio': cached,
                        'timestamp': time.time()
                    }
                    return cached
            except Exception as e:
                logger.warning(f"Ошибка чтения из Redis: {e}")
        
        return None
    
    def _cache_audio(self, cache_key: str, audio_data: bytes):
        """
        Кэширование аудио
        
        Args:
            cache_key: Ключ кэша
            audio_data: Аудио данные
        """
        # Сохраняем в in-memory кэш
        self.audio_cache[cache_key] = {
            'audio': audio_data,
            'timestamp': time.time()
        }
        
        # Сохраняем в Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"tts:{cache_key}",
                    self.config.CACHE_TTL,
                    audio_data
                )
            except Exception as e:
                logger.warning(f"Ошибка записи в Redis: {e}")
        
        # Очищаем старые записи из in-memory кэша
        if len(self.audio_cache) > 1000:
            oldest_key = min(self.audio_cache.keys(), 
                           key=lambda k: self.audio_cache[k]['timestamp'])
            del self.audio_cache[oldest_key]
    
    def preprocess_text(self, text: str, language: str = "ru") -> str:
        """
        Предобработка текста для синтеза
        
        Args:
            text: Исходный текст
            language: Язык текста
            
        Returns:
            Обработанный текст
        """
        # Удаляем лишние пробелы и переносы
        text = ' '.join(text.split())
        
        # Замены для русского языка
        if language == "ru":
            # Заменяем английские кавычки на русские
            text = text.replace('"', '«').replace('"', '»')
            # Исправляем common issues
            text = text.replace("ё", "е")  # Некоторые модели плохо работают с ё
        
        # Ограничиваем длину
        if len(text) > self.config.MAX_TEXT_LENGTH:
            text = text[:self.config.MAX_TEXT_LENGTH] + "..."
            logger.warning(f"Текст обрезан до {self.config.MAX_TEXT_LENGTH} символов")
        
        return text
    
    def synthesize(self, text: str, language: str = "ru", 
                  voice: str = "neutral", speed: float = 1.0, 
                  pitch: float = 1.0, emotion: str = "neutral") -> Dict:
        """
        Синтез речи из текста
        
        Args:
            text: Текст для синтеза
            language: Язык текста
            voice: Голос (male, female, neutral или конкретное имя)
            speed: Скорость речи (0.5-2.0)
            pitch: Высота тона (0.5-2.0)
            emotion: Эмоция (neutral, happy, sad, angry)
            
        Returns:
            Словарь с результатом синтеза
        """
        start_time = time.time()
        self.stats['requests'] += 1
        self.stats['characters_processed'] += len(text)
        
        try:
            # Предобработка текста
            processed_text = self.preprocess_text(text, language)
            
            # Генерация ключа кэша
            cache_key = self._generate_cache_key(
                processed_text, language, voice, speed, pitch
            )
            
            # Проверка кэша
            cached_audio = self._get_cached_audio(cache_key)
            if cached_audio:
                logger.debug(f"Использован кэшированный результат для: {cache_key[:16]}...")
                processing_time = time.time() - start_time
                
                return {
                    'success': True,
                    'text': text,
                    'processed_text': processed_text,
                    'audio': cached_audio,
                    'audio_format': self.config.AUDIO_FORMAT,
                    'sample_rate': self.config.SAMPLE_RATE,
                    'duration': len(cached_audio) / (self.config.SAMPLE_RATE * 2),  # Приблизительно
                    'language': language,
                    'voice': voice,
                    'speed': speed,
                    'pitch': pitch,
                    'emotion': emotion,
                    'cached': True,
                    'processing_time': processing_time,
                    'cache_key': cache_key,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Получение модели
            tts_model = self.get_model_for_language(language, voice)
            
            # Выбор конкретного голоса для английского
            speaker = None
            if language == "en" and voice in ["male", "female"]:
                # Выбираем случайный голос из доступных
                import random
                available_voices = self.config.VOICES['en'][voice]
                speaker = random.choice(available_voices)
            
            # Синтез речи
            logger.info(f"Синтез речи: {len(processed_text)} символов, язык: {language}")
            
            # Генерируем аудио
            audio_path = self.config.TEMP_DIR / f"temp_{cache_key}.wav"
            
            tts_model.tts_to_file(
                text=processed_text,
                speaker=speaker,
                language=language if language != "ru" else None,  # Для русской модели язык не указываем
                file_path=str(audio_path),
                speed=speed
            )
            
            # Читаем сгенерированный файл
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Удаляем временный файл
            audio_path.unlink(missing_ok=True)
            
            # Кэшируем результат
            self._cache_audio(cache_key, audio_data)
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"✅ Синтез завершен за {processing_time:.2f} секунд")
            
            return {
                'success': True,
                'text': text,
                'processed_text': processed_text,
                'audio': audio_data,
                'audio_format': self.config.AUDIO_FORMAT,
                'sample_rate': self.config.SAMPLE_RATE,
                'duration': len(audio_data) / (self.config.SAMPLE_RATE * 2),
                'language': language,
                'voice': voice,
                'speed': speed,
                'pitch': pitch,
                'emotion': emotion,
                'cached': False,
                'processing_time': processing_time,
                'cache_key': cache_key,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Ошибка синтеза речи: {e}", exc_info=True)
            
            return {
                'success': False,
                'error': str(e),
                'text': text,
                'language': language,
                'voice': voice,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def synthesize_async(self, text: str, language: str = "ru", 
                             voice: str = "neutral", speed: float = 1.0, 
                             pitch: float = 1.0, emotion: str = "neutral") -> Dict:
        """
        Асинхронный синтез речи
        
        Args:
            text: Текст для синтеза
            language: Язык текста
            voice: Голос
            speed: Скорость речи
            pitch: Высота тона
            emotion: Эмоция
            
        Returns:
            Словарь с результатом синтеза
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.synthesize,
            text, language, voice, speed, pitch, emotion
        )
    
    def batch_synthesize(self, texts: List[str], language: str = "ru", 
                        voice: str = "neutral", speed: float = 1.0, 
                        pitch: float = 1.0) -> List[Dict]:
        """
        Пакетный синтез речи
        
        Args:
            texts: Список текстов для синтеза
            language: Язык текстов
            voice: Голос
            speed: Скорость речи
            pitch: Высота тона
            
        Returns:
            Список результатов синтеза
        """
        results = []
        
        for i in range(0, len(texts), self.config.BATCH_SIZE):
            batch = texts[i:i + self.config.BATCH_SIZE]
            logger.info(f"Обработка батча {i//self.config.BATCH_SIZE + 1}: {len(batch)} текстов")
            
            for text in batch:
                result = self.synthesize(
                    text=text,
                    language=language,
                    voice=voice,
                    speed=speed,
                    pitch=pitch
                )
                results.append(result)
            
            # Небольшая пауза между батчами
            if i + self.config.BATCH_SIZE < len(texts):
                time.sleep(0.1)
        
        return results
    
    def audio_to_base64(self, audio_data: bytes) -> str:
        """
        Конвертация аудио в base64
        
        Args:
            audio_data: Бинарные аудио данные
            
        Returns:
            Base64 строка
        """
        return base64.b64encode(audio_data).decode('utf-8')
    
    def base64_to_audio(self, base64_string: str) -> bytes:
        """
        Конвертация base64 в аудио
        
        Args:
            base64_string: Base64 строка
            
        Returns:
            Бинарные аудио данные
        """
        return base64.b64decode(base64_string)
    
    def get_audio_info(self, audio_data: bytes) -> Dict:
        """
        Получение информации об аудио
        
        Args:
            audio_data: Бинарные аудио данные
            
        Returns:
            Информация об аудио
        """
        try:
            # Используем soundfile для чтения метаданных
            with io.BytesIO(audio_data) as buffer:
                with sf.SoundFile(buffer) as audio_file:
                    return {
                        'sample_rate': audio_file.samplerate,
                        'channels': audio_file.channels,
                        'duration': len(audio_file) / audio_file.samplerate,
                        'format': audio_file.format,
                        'subtype': audio_file.subtype,
                        'frames': len(audio_file)
                    }
        except Exception as e:
            logger.error(f"Ошибка чтения информации об аудио: {e}")
            return {}
    
    def convert_format(self, audio_data: bytes, target_format: str = "mp3", 
                      bitrate: str = "192k") -> bytes:
        """
        Конвертация аудио в другой формат
        
        Args:
            audio_data: Исходные аудио данные
            target_format: Целевой формат (mp3, ogg, flac)
            bitrate: Битрейт для сжатия
            
        Returns:
            Конвертированные аудио данные
        """
        try:
            # Сохраняем во временный файл
            temp_input = self.config.TEMP_DIR / "temp_input.wav"
            temp_output = self.config.TEMP_DIR / f"temp_output.{target_format}"
            
            with open(temp_input, 'wb') as f:
                f.write(audio_data)
            
            # Используем ffmpeg для конвертации
            import subprocess
            
            cmd = [
                'ffmpeg', '-y', '-i', str(temp_input),
                '-codec:a', 'libmp3lame' if target_format == 'mp3' else 'libvorbis',
                '-b:a', bitrate,
                str(temp_output)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Читаем конвертированный файл
            with open(temp_output, 'rb') as f:
                converted_data = f.read()
            
            # Удаляем временные файлы
            temp_input.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)
            
            return converted_data
            
        except Exception as e:
            logger.error(f"Ошибка конвертации аудио: {e}")
            # Возвращаем оригинальные данные в случае ошибки
            return audio_data
    
    def list_available_voices(self, language: str = None) -> Dict:
        """
        Список доступных голосов
        
        Args:
            language: Язык для фильтрации
            
        Returns:
            Словарь с доступными голосами
        """
        voices = {}
        
        if language:
            if language in self.config.VOICES:
                voices[language] = self.config.VOICES[language]
        else:
            voices = self.config.VOICES.copy()
        
        return voices
    
    def get_health_status(self) -> Dict:
        """
        Получение статуса здоровья сервиса
        
        Returns:
            Словарь со статусом
        """
        gpu_available = torch.cuda.is_available()
        gpu_info = {}
        
        if gpu_available:
            gpu_info = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,  # GB
                'memory_reserved': torch.cuda.memory_reserved(0) / 1024**3,  # GB
            }
        
        return {
            'status': 'healthy',
            'gpu_available': gpu_available,
            'gpu_info': gpu_info if gpu_available else None,
            'loaded_models': list(self.models.keys()),
            'cache_size': len(self.audio_cache),
            'redis_connected': self.redis_client is not None,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.executor.shutdown(wait=True)
        
        # Очищаем модели
        for model_name in list(self.models.keys()):
            del self.models[model_name]
        
        logger.info("✅ TTSService остановлен")

class TTSAPI:
    """REST API для TTS сервиса"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5001):
        self.host = host
        self.port = port
        self.tts_service = TTSService()
        
        # Инициализация Flask приложения
        from flask import Flask, request, jsonify, send_file
        from flask_cors import CORS
        
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Регистрация маршрутов
        self._register_routes()
        
    def _register_routes(self):
        """Регистрация API маршрутов"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            status = self.tts_service.get_health_status()
            return jsonify(status)
        
        @self.app.route('/api/v1/tts/synthesize', methods=['POST'])
        def synthesize():
            """Синтез речи из текста"""
            data = request.json
            
            text = data.get('text', '')
            language = data.get('language', 'ru')
            voice = data.get('voice', 'neutral')
            speed = data.get('speed', 1.0)
            pitch = data.get('pitch', 1.0)
            emotion = data.get('emotion', 'neutral')
            format = data.get('format', 'wav')
            return_base64 = data.get('return_base64', False)
            
            if not text:
                return jsonify({
                    'success': False,
                    'error': 'Текст обязателен'
                }), 400
            
            # Синтез речи
            result = self.tts_service.synthesize(
                text=text,
                language=language,
                voice=voice,
                speed=speed,
                pitch=pitch,
                emotion=emotion
            )
            
            if not result['success']:
                return jsonify(result), 500
            
            # Конвертация формата если нужно
            if format != 'wav':
                audio_data = self.tts_service.convert_format(
                    result['audio'],
                    target_format=format
                )
                result['audio_format'] = format
            else:
                audio_data = result['audio']
            
            # Возвращаем результат
            if return_base64:
                result['audio'] = self.tts_service.audio_to_base64(audio_data)
                return jsonify(result)
            else:
                # Возвращаем файл
                from io import BytesIO
                audio_io = BytesIO(audio_data)
                audio_io.seek(0)
                
                filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
                return send_file(
                    audio_io,
                    mimetype=f'audio/{format}',
                    as_attachment=True,
                    download_name=filename
                )
        
        @self.app.route('/api/v1/tts/batch', methods=['POST'])
        def batch_synthesize():
            """Пакетный синтез речи"""
            data = request.json
            
            texts = data.get('texts', [])
            language = data.get('language', 'ru')
            voice = data.get('voice', 'neutral')
            speed = data.get('speed', 1.0)
            pitch = data.get('pitch', 1.0)
            
            if not texts or not isinstance(texts, list):
                return jsonify({
                    'success': False,
                    'error': 'Список текстов обязателен'
                }), 400
            
            # Пакетный синтез
            results = self.tts_service.batch_synthesize(
                texts=texts,
                language=language,
                voice=voice,
                speed=speed,
                pitch=pitch
            )
            
            return jsonify({
                'success': True,
                'results': results,
                'total': len(results),
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/v1/tts/voices', methods=['GET'])
        def list_voices():
            """Список доступных голосов"""
            language = request.args.get('language')
            
            voices = self.tts_service.list_available_voices(language)
            
            return jsonify({
                'success': True,
                'voices': voices,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/v1/tts/info', methods=['POST'])
        def audio_info():
            """Информация об аудио"""
            data = request.json
            
            audio_base64 = data.get('audio')
            if not audio_base64:
                return jsonify({
                    'success': False,
                    'error': 'Аудио данные обязательны'
                }), 400
            
            try:
                audio_data = self.tts_service.base64_to_audio(audio_base64)
                info = self.tts_service.get_audio_info(audio_data)
                
                return jsonify({
                    'success': True,
                    'info': info,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/tts/convert', methods=['POST'])
        def convert_audio():
            """Конвертация аудио формата"""
            data = request.json
            
            audio_base64 = data.get('audio')
            target_format = data.get('format', 'mp3')
            bitrate = data.get('bitrate', '192k')
            
            if not audio_base64:
                return jsonify({
                    'success': False,
                    'error': 'Аудио данные обязательны'
                }), 400
            
            try:
                audio_data = self.tts_service.base64_to_audio(audio_base64)
                converted_data = self.tts_service.convert_format(
                    audio_data,
                    target_format=target_format,
                    bitrate=bitrate
                )
                
                return jsonify({
                    'success': True,
                    'audio': self.tts_service.audio_to_base64(converted_data),
                    'format': target_format,
                    'bitrate': bitrate,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def run(self, debug: bool = False):
        """
        Запуск API сервера
        
        Args:
            debug: Режим отладки
        """
        logger.info(f"Запуск TTS API на {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)
    
    def stop(self):
        """Остановка сервиса"""
        self.tts_service.cleanup()
        logger.info("✅ TTS API остановлен")

# FastAPI версия (альтернатива Flask)
class FastTTSAPI:
    """FastAPI версия TTS сервиса"""
    
    def __init__(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, StreamingResponse
        import io
        
        self.app = FastAPI(title="ARIS Neuro TTS API", version="3.0.0")
        self.tts_service = TTSService()
        
        # Настройка CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Регистрация маршрутов
        self._register_routes()
    
    def _register_routes(self):
        """Регистрация FastAPI маршрутов"""
        from fastapi import APIRouter, Query, Body
        from pydantic import BaseModel
        from typing import List, Optional
        
        router = APIRouter(prefix="/api/v1/tts", tags=["TTS"])
        
        # Модели запросов
        class SynthesizeRequest(BaseModel):
            text: str
            language: str = "ru"
            voice: str = "neutral"
            speed: float = 1.0
            pitch: float = 1.0
            emotion: str = "neutral"
            format: str = "wav"
            return_base64: bool = False
        
        class BatchRequest(BaseModel):
            texts: List[str]
            language: str = "ru"
            voice: str = "neutral"
            speed: float = 1.0
            pitch: float = 1.0
        
        class ConvertRequest(BaseModel):
            audio: str  # base64
            format: str = "mp3"
            bitrate: str = "192k"
        
        @router.get("/health")
        async def health():
            return self.tts_service.get_health_status()
        
        @router.post("/synthesize")
        async def synthesize(request: SynthesizeRequest):
            result = self.tts_service.synthesize(
                text=request.text,
                language=request.language,
                voice=request.voice,
                speed=request.speed,
                pitch=request.pitch,
                emotion=request.emotion
            )
            
            if not result['success']:
                raise HTTPException(status_code=500, detail=result['error'])
            
            # Конвертация формата если нужно
            if request.format != 'wav':
                audio_data = self.tts_service.convert_format(
                    result['audio'],
                    target_format=request.format
                )
                result['audio_format'] = request.format
            else:
                audio_data = result['audio']
            
            if request.return_base64:
                result['audio'] = self.tts_service.audio_to_base64(audio_data)
                return result
            else:
                audio_io = io.BytesIO(audio_data)
                audio_io.seek(0)
                
                filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{request.format}"
                
                return StreamingResponse(
                    audio_io,
                    media_type=f'audio/{request.format}',
                    headers={
                        'Content-Disposition': f'attachment; filename="{filename}"'
                    }
                )
        
        @router.post("/batch")
        async def batch_synthesize(request: BatchRequest):
            results = self.tts_service.batch_synthesize(
                texts=request.texts,
                language=request.language,
                voice=request.voice,
                speed=request.speed,
                pitch=request.pitch
            )
            
            return {
                'success': True,
                'results': results,
                'total': len(results),
                'timestamp': datetime.now().isoformat()
            }
        
        @router.get("/voices")
        async def list_voices(language: Optional[str] = None):
            voices = self.tts_service.list_available_voices(language)
            return {
                'success': True,
                'voices': voices,
                'timestamp': datetime.now().isoformat()
            }
        
        @router.post("/convert")
        async def convert_audio(request: ConvertRequest):
            try:
                audio_data = self.tts_service.base64_to_audio(request.audio)
                converted_data = self.tts_service.convert_format(
                    audio_data,
                    target_format=request.format,
                    bitrate=request.bitrate
                )
                
                return {
                    'success': True,
                    'audio': self.tts_service.audio_to_base64(converted_data),
                    'format': request.format,
                    'bitrate': request.bitrate,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Регистрируем роутер
        self.app.include_router(router)
    
    def run(self, host: str = "0.0.0.0", port: int = 5001):
        """
        Запуск FastAPI сервера
        
        Args:
            host: Хост
            port: Порт
        """
        import uvicorn
        logger.info(f"Запуск FastAPI TTS API на {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)
    
    def stop(self):
        """Остановка сервиса"""
        self.tts_service.cleanup()
        logger.info("✅ FastAPI TTS API остановлен")

# Основной скрипт запуска
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ARIS Neuro TTS Service")
    parser.add_argument("--api", choices=["flask", "fastapi"], default="fastapi",
                       help="Тип API (flask или fastapi)")
    parser.add_argument("--host", default="0.0.0.0", help="Хост для API")
    parser.add_argument("--port", type=int, default=5001, help="Порт для API")
    parser.add_argument("--debug", action="store_true", help="Режим отладки")
    
    args = parser.parse_args()
    
    try:
        if args.api == "flask":
            api = TTSAPI(host=args.host, port=args.port)
            api.run(debug=args.debug)
        else:
            api = FastTTSAPI()
            api.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания...")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
    finally:
        if 'api' in locals():
            api.stop()
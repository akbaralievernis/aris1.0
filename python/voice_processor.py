# Voice processor placeholder
"""
Основной Python ML сервис для обработки голоса
Интеграция Whisper, VAD, денойзинга, анализа эмоций
"""

import os
import sys
import json
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import time
import traceback

# Импорт ML библиотек
try:
    import torch
    import torchaudio
    import whisper
    import librosa
    import soundfile as sf
    import noisereduce as nr
    from scipy import signal
    import speech_recognition as sr
    from pyannote.audio import Pipeline
    from transformers import pipeline as hf_pipeline
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras
except ImportError as e:
    print(f"❌ Не удалось импортировать ML библиотеки: {e}")
    print("Установите зависимости: pip install -r requirements.txt")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/voice_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VoiceProcessorConfig:
    """Конфигурация процессора голоса"""
    
    # Пути
    MODELS_DIR = Path("/app/models")
    TEMP_DIR = Path("/app/temp/voice")
    CACHE_DIR = Path("/app/cache/voice")
    
    # Настройки аудио
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 0.1  # 100ms chunks for real-time
    MAX_RECORDING_TIME = 30  # Максимальная запись в секундах
    AUDIO_FORMAT = "wav"
    
    # Whisper настройки
    WHISPER_MODEL = "medium"
    WHISPER_LANGUAGE = "ru"
    WHISPER_TEMPERATURE = 0.0
    WHISPER_BEAM_SIZE = 5
    
    # VAD (Voice Activity Detection)
    VAD_THRESHOLD = 0.5
    VAD_FRAME_DURATION = 0.03  # 30ms
    VAD_PADDING_DURATION = 0.1  # 100ms
    
    # Денойзинг
    NOISE_REDUCTION_PROFILE_DURATION = 1.0  # 1 секунда для профиля шума
    NOISE_REDUCTION_STATIONARY = True
    
    # Эмоциональный анализ
    EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    EMOTION_THRESHOLD = 0.6
    
    # Диаризация (разделение спикеров)
    DIARIZATION_MODEL = "pyannote/speaker-diarization"
    
    # Wake word detection
    WAKE_WORD_MODEL = "wakeword_model.pth"
    WAKE_WORD_THRESHOLD = 0.7
    
    # Производительность
    USE_GPU = True
    GPU_MEMORY_LIMIT = 0.5  # Ограничение памяти GPU в GB
    BATCH_SIZE = 16
    MAX_WORKERS = 4
    
    # Кэширование
    CACHE_ENABLED = True
    CACHE_TTL = 3600  # 1 час

class VoiceProcessor:
    """Основной процессор голосовых данных"""
    
    def __init__(self, config: VoiceProcessorConfig = None):
        self.config = config or VoiceProcessorConfig()
        self.device = self._get_device()
        
        # Создание директорий
        self._create_directories()
        
        # Загрузка моделей
        self.models = {}
        self._load_models()
        
        # Инициализация кэша
        self.cache = {}
        
        # Статистика
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0,
            'audio_duration_processed': 0,
            'cache_hits': 0
        }
        
        logger.info(f"Инициализация VoiceProcessor на устройстве: {self.device}")
        logger.info(f"Конфигурация: {json.dumps(asdict(self.config), indent=2)}")
    
    def _get_device(self) -> str:
        """Определение устройства для вычислений"""
        if self.config.USE_GPU and torch.cuda.is_available():
            device = "cuda"
            # Устанавливаем лимит памяти GPU
            torch.cuda.set_per_process_memory_fraction(self.config.GPU_MEMORY_LIMIT)
            logger.info(f"✅ Используется GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Ограничение памяти GPU: {self.config.GPU_MEMORY_LIMIT * 100}%")
        else:
            device = "cpu"
            logger.info("ℹ️ Используется CPU")
        
        return device
    
    def _create_directories(self):
        """Создание необходимых директорий"""
        directories = [
            self.config.MODELS_DIR,
            self.config.TEMP_DIR,
            self.config.CACHE_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Создана директория: {directory}")
    
    def _load_models(self):
        """Загрузка всех ML моделей"""
        logger.info("Загрузка ML моделей...")
        
        try:
            # Загрузка Whisper для транскрипции
            logger.info(f"Загрузка Whisper модели: {self.config.WHISPER_MODEL}")
            self.models['whisper'] = whisper.load_model(
                self.config.WHISPER_MODEL,
                device=self.device,
                download_root=str(self.config.MODELS_DIR / "whisper")
            )
            logger.info("✅ Whisper модель загружена")
            
            # Загрузка модели для эмоционального анализа
            logger.info("Загрузка модели анализа эмоций...")
            self.models['emotion'] = hf_pipeline(
                "audio-classification",
                model=self.config.EMOTION_MODEL,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✅ Модель анализа эмоций загружена")
            
            # Загрузка модели для диаризации
            logger.info("Загрузка модели диаризации...")
            try:
                self.models['diarization'] = Pipeline.from_pretrained(
                    self.config.DIARIZATION_MODEL,
                    use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
                ).to(torch.device(self.device))
                logger.info("✅ Модель диаризации загружена")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель диаризации: {e}")
                self.models['diarization'] = None
            
            # Загрузка модели wake word detection
            logger.info("Загрузка модели wake word detection...")
            wake_word_path = self.config.MODELS_DIR / self.config.WAKE_WORD_MODEL
            if wake_word_path.exists():
                try:
                    self.models['wake_word'] = torch.jit.load(
                        str(wake_word_path),
                        map_location=self.device
                    )
                    self.models['wake_word'].eval()
                    logger.info("✅ Модель wake word detection загружена")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить модель wake word: {e}")
                    self.models['wake_word'] = None
            else:
                logger.warning(f"Модель wake word не найдена: {wake_word_path}")
                self.models['wake_word'] = None
            
            # Загрузка модели для детекции языка
            logger.info("Загрузка модели детекции языка...")
            try:
                from langdetect import detect
                self.models['language_detection'] = detect
                logger.info("✅ Модель детекции языка загружена")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель детекции языка: {e}")
                self.models['language_detection'] = None
            
            logger.info(f"✅ Всего загружено моделей: {len(self.models)}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки моделей: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Предобработка аудио: ресемплинг, нормализация, денойзинг
        
        Args:
            audio_data: Аудио данные
            sample_rate: Исходная частота дискретизации
            
        Returns:
            Словарь с обработанным аудио и метаданными
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Предобработка аудио: {len(audio_data)} samples, SR: {sample_rate}")
            
            # Конвертация в моно если нужно
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
                logger.debug("Аудио конвертировано в моно")
            
            # Ресемплинг до целевой частоты
            if sample_rate != self.config.SAMPLE_RATE:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=self.config.SAMPLE_RATE
                )
                sample_rate = self.config.SAMPLE_RATE
                logger.debug(f"Ресемплинг до {sample_rate}Hz")
            
            # Нормализация
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            
            # Денойзинг
            if len(audio_data) > int(self.config.NOISE_REDUCTION_PROFILE_DURATION * sample_rate):
                # Используем первые N секунд как профиль шума
                noise_profile_duration = int(
                    self.config.NOISE_REDUCTION_PROFILE_DURATION * sample_rate
                )
                noise_profile = audio_data[:noise_profile_duration]
                
                audio_data = nr.reduce_noise(
                    y=audio_data,
                    sr=sample_rate,
                    y_noise=noise_profile,
                    stationary=self.config.NOISE_REDUCTION_STATIONARY,
                    prop_decrease=0.8
                )
                logger.debug("Денойзинг применен")
            
            # Обрезка тишины
            audio_data = self._trim_silence(audio_data, sample_rate)
            
            processing_time = time.time() - start_time
            
            return {
                'audio': audio_data,
                'sample_rate': sample_rate,
                'duration': len(audio_data) / sample_rate,
                'processing_time': processing_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Ошибка предобработки аудио: {e}")
            return {
                'audio': audio_data,
                'sample_rate': sample_rate,
                'duration': len(audio_data) / sample_rate,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _trim_silence(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Обрезка тишины с начала и конца
        
        Args:
            audio: Аудио данные
            sample_rate: Частота дискретизации
            
        Returns:
            Аудио без тишины
        """
        # Используем энергию сигнала для детекции тишины
        frame_length = int(0.025 * sample_rate)  # 25ms
        hop_length = int(0.010 * sample_rate)    # 10ms
        
        # Вычисляем энергию
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2)
            for i in range(0, len(audio) - frame_length, hop_length)
        ])
        
        if len(energy) == 0:
            return audio
        
        # Порог для тишины
        threshold = np.percentile(energy, 20)
        
        # Находим не тихие фреймы
        non_silent_frames = np.where(energy > threshold)[0]
        
        if len(non_silent_frames) == 0:
            return audio
        
        # Конвертируем в samples
        start_frame = max(0, non_silent_frames[0] - 5)  # Добавляем padding
        end_frame = min(len(energy) - 1, non_silent_frames[-1] + 5)
        
        start_sample = start_frame * hop_length
        end_sample = min(len(audio), (end_frame * hop_length) + frame_length)
        
        return audio[start_sample:end_sample]
    
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """
        Детекция активности речи (VAD)
        
        Args:
            audio: Аудио данные
            sample_rate: Частота дискретизации
            
        Returns:
            Список сегментов с речью
        """
        try:
            frame_duration = self.config.VAD_FRAME_DURATION
            frame_samples = int(frame_duration * sample_rate)
            
            segments = []
            current_segment = None
            
            for i in range(0, len(audio), frame_samples):
                frame = audio[i:i+frame_samples]
                
                if len(frame) < frame_samples:
                    break
                
                # Простая детекция по энергии
                energy = np.mean(frame**2)
                is_speech = energy > self.config.VAD_THRESHOLD * np.mean(audio**2)
                
                timestamp = i / sample_rate
                
                if is_speech:
                    if current_segment is None:
                        current_segment = {
                            'start': timestamp,
                            'end': timestamp + frame_duration,
                            'confidence': float(energy)
                        }
                    else:
                        current_segment['end'] = timestamp + frame_duration
                        current_segment['confidence'] = max(
                            current_segment['confidence'],
                            float(energy)
                        )
                else:
                    if current_segment is not None:
                        # Добавляем padding
                        current_segment['start'] = max(0, current_segment['start'] - 
                                                     self.config.VAD_PADDING_DURATION)
                        current_segment['end'] = current_segment['end'] + \
                                                self.config.VAD_PADDING_DURATION
                        segments.append(current_segment)
                        current_segment = None
            
            # Добавляем последний сегмент если есть
            if current_segment is not None:
                current_segment['start'] = max(0, current_segment['start'] - 
                                             self.config.VAD_PADDING_DURATION)
                current_segment['end'] = current_segment['end'] + \
                                        self.config.VAD_PADDING_DURATION
                segments.append(current_segment)
            
            # Объединяем близкие сегменты
            segments = self._merge_segments(segments, max_gap=0.3)
            
            logger.debug(f"Найдено сегментов речи: {len(segments)}")
            
            return segments
            
        except Exception as e:
            logger.error(f"Ошибка VAD: {e}")
            return []
    
    def _merge_segments(self, segments: List[Dict], max_gap: float = 0.3) -> List[Dict]:
        """Объединение близких сегментов"""
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            if segment['start'] - current['end'] <= max_gap:
                # Объединяем сегменты
                current['end'] = segment['end']
                current['confidence'] = max(current['confidence'], segment['confidence'])
            else:
                merged.append(current)
                current = segment.copy()
        
        merged.append(current)
        return merged
    
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int, 
                        language: str = None) -> Dict:
        """
        Транскрипция аудио с использованием Whisper
        
        Args:
            audio: Аудио данные
            sample_rate: Частота дискретизации
            language: Язык (опционально)
            
        Returns:
            Результат транскрипции
        """
        start_time = time.time()
        
        try:
            # Проверяем кэш
            cache_key = f"transcribe_{hash(audio.tobytes())}_{language}"
            if self.config.CACHE_ENABLED and cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.config.CACHE_TTL:
                    self.stats['cache_hits'] += 1
                    return cached_result['result']
            
            # Подготовка аудио для Whisper
            audio_whisper = whisper.pad_or_trim(audio)
            
            # Создание лог-мел спектрограммы
            mel = whisper.log_mel_spectrogram(audio_whisper).to(self.device)
            
            # Детекция языка если не указан
            if language is None:
                _, probs = self.models['whisper'].detect_language(mel)
                language = max(probs, key=probs.get)
                logger.debug(f"Определен язык: {language} (вероятность: {probs[language]:.2f})")
            
            # Опции декодирования
            options = whisper.DecodingOptions(
                language=language,
                temperature=self.config.WHISPER_TEMPERATURE,
                beam_size=self.config.WHISPER_BEAM_SIZE,
                fp16=(self.device == "cuda")
            )
            
            # Декодирование
            result = whisper.decode(self.models['whisper'], mel, options)
            
            # Форматирование результата
            transcription_result = {
                'text': result.text,
                'language': language,
                'confidence': np.exp(result.avg_logprob),
                'tokens': len(result.tokens),
                'no_speech_prob': result.no_speech_prob,
                'processing_time': time.time() - start_time,
                'success': True
            }
            
            # Кэширование результата
            if self.config.CACHE_ENABLED:
                self.cache[cache_key] = {
                    'result': transcription_result,
                    'timestamp': time.time()
                }
            
            logger.info(f"Транскрипция завершена: {len(result.text)} символов")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Ошибка транскрипции: {e}")
            return {
                'text': '',
                'language': language or 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }
    
    def analyze_emotions(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Анализ эмоций в голосе
        
        Args:
            audio: Аудио данные
            sample_rate: Частота дискретизация
            
        Returns:
            Результат анализа эмоций
        """
        start_time = time.time()
        
        try:
            # Проверяем кэш
            cache_key = f"emotion_{hash(audio.tobytes())}"
            if self.config.CACHE_ENABLED and cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.config.CACHE_TTL:
                    self.stats['cache_hits'] += 1
                    return cached_result['result']
            
            # Подготовка аудио для модели эмоций
            # Модель ожидает массив numpy с частотой 16000
            if sample_rate != 16000:
                audio_emotion = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            else:
                audio_emotion = audio.copy()
            
            # Анализ эмоций
            emotion_result = self.models['emotion'](
                audio_emotion,
                sampling_rate=16000,
                top_k=5
            )
            
            # Форматирование результата
            emotions = []
            dominant_emotion = None
            
            for item in emotion_result:
                emotion_data = {
                    'emotion': item['label'],
                    'confidence': float(item['score']),
                    'normalized_confidence': float(item['score'])
                }
                emotions.append(emotion_data)
                
                if dominant_emotion is None or item['score'] > dominant_emotion['confidence']:
                    dominant_emotion = emotion_data
            
            # Дополнительный анализ: валентность и активация
            valence, activation = self._analyze_voice_characteristics(audio, sample_rate)
            
            result = {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'valence': valence,  # От негативного к позитивному
                'activation': activation,  # От спокойного к активному
                'processing_time': time.time() - start_time,
                'success': True
            }
            
            # Кэширование результата
            if self.config.CACHE_ENABLED:
                self.cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
            
            logger.debug(f"Анализ эмоций: {dominant_emotion['emotion']} "
                        f"(confidence: {dominant_emotion['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа эмоций: {e}")
            return {
                'emotions': [],
                'dominant_emotion': {'emotion': 'neutral', 'confidence': 0.5},
                'valence': 0.5,
                'activation': 0.5,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }
    
    def _analyze_voice_characteristics(self, audio: np.ndarray, sample_rate: int) -> Tuple[float, float]:
        """Анализ характеристик голоса: валентность и активация"""
        try:
            # Извлечение аудио фич
            features = {}
            
            # Энергия (активация)
            features['energy'] = np.mean(audio**2)
            
            # Частота основного тона (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate
            )
            f0 = f0[~np.isnan(f0)]
            if len(f0) > 0:
                features['pitch_mean'] = np.mean(f0)
                features['pitch_std'] = np.std(f0)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
            
            # Спектральные центроиды
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=sample_rate
            )[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # MFCC (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            for i in range(min(5, mfccs.shape[0])):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Нормализация фич
            # Простая rule-based логика для определения валентности и активации
            
            # Активация: на основе энергии и вариативности тона
            activation = min(1.0, features['energy'] * 10) * 0.6 + \
                        min(1.0, features['pitch_std'] / 100) * 0.4
            
            # Валентность: на основе среднего тона и спектрального центроида
            valence = 0.5
            if features['pitch_mean'] > 200:  # Высокий тон - обычно позитивно
                valence += 0.2
            if features['spectral_centroid_mean'] > 2000:  # Яркий тембр
                valence += 0.1
            
            # Ограничиваем значения
            activation = max(0.0, min(1.0, activation))
            valence = max(0.0, min(1.0, valence))
            
            return float(valence), float(activation)
            
        except Exception as e:
            logger.warning(f"Ошибка анализа характеристик голоса: {e}")
            return 0.5, 0.5
    
    def detect_wake_word(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Детекция ключевого слова "ARIS"
        
        Args:
            audio: Аудио данные
            sample_rate: Частота дискретизации
            
        Returns:
            Результат детекции
        """
        start_time = time.time()
        
        try:
            if self.models.get('wake_word') is None:
                # Fallback на энергетическую детекцию
                energy = np.mean(audio**2)
                detected = energy > 0.01
                
                return {
                    'detected': bool(detected),
                    'confidence': float(min(1.0, energy * 10)),
                    'model': 'energy_based',
                    'processing_time': time.time() - start_time,
                    'success': True
                }
            
            # Подготовка аудио для модели
            # Модель ожидает определенный формат
            target_sr = 16000
            if sample_rate != target_sr:
                audio_wake = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
            else:
                audio_wake = audio.copy()
            
            # Нормализация
            audio_wake = audio_wake / (np.max(np.abs(audio_wake)) + 1e-8)
            
            # Конвертация в tensor
            audio_tensor = torch.FloatTensor(audio_wake).unsqueeze(0)
            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()
            
            # Предикт
            with torch.no_grad():
                output = self.models['wake_word'](audio_tensor)
                confidence = torch.sigmoid(output).item()
            
            detected = confidence > self.config.WAKE_WORD_THRESHOLD
            
            result = {
                'detected': bool(detected),
                'confidence': float(confidence),
                'threshold': self.config.WAKE_WORD_THRESHOLD,
                'model': 'neural_network',
                'processing_time': time.time() - start_time,
                'success': True
            }
            
            if detected:
                logger.info(f"✅ Wake word detected! Confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка детекции wake word: {e}")
            return {
                'detected': False,
                'confidence': 0.0,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }
    
    def diarize_speakers(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Диаризация (разделение спикеров)
        
        Args:
            audio: Аудио данные
            sample_rate: Частота дискретизации
            
        Returns:
            Результат диаризации
        """
        start_time = time.time()
        
        try:
            if self.models.get('diarization') is None:
                return {
                    'speakers': [],
                    'segments': [],
                    'error': 'Diarization model not loaded',
                    'processing_time': time.time() - start_time,
                    'success': False
                }
            
            # Сохраняем аудио во временный файл
            temp_file = self.config.TEMP_DIR / f"diarize_{int(time.time())}.wav"
            sf.write(str(temp_file), audio, sample_rate)
            
            # Диаризация
            diarization = self.models['diarization'](str(temp_file))
            
            # Обработка результата
            speakers = set()
            segments = []
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                segments.append({
                    'speaker': speaker,
                    'start': float(turn.start),
                    'end': float(turn.end),
                    'duration': float(turn.end - turn.start)
                })
            
            # Удаляем временный файл
            temp_file.unlink(missing_ok=True)
            
            result = {
                'speakers': list(speakers),
                'segments': segments,
                'total_speakers': len(speakers),
                'total_segments': len(segments),
                'processing_time': time.time() - start_time,
                'success': True
            }
            
            logger.info(f"Диаризация: найдено {len(speakers)} спикеров, {len(segments)} сегментов")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка диаризации: {e}")
            return {
                'speakers': [],
                'segments': [],
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }
    
    def extract_audio_features(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Извлечение аудио фич для аналитики
        
        Args:
            audio: Аудио данные
            sample_rate: Частота дискретизации
            
        Returns:
            Извлеченные фичи
        """
        start_time = time.time()
        
        try:
            features = {}
            
            # Базовые фичи
            features['duration'] = len(audio) / sample_rate
            features['samples'] = len(audio)
            features['sample_rate'] = sample_rate
            
            # Энергия
            features['energy'] = float(np.mean(audio**2))
            features['energy_std'] = float(np.std(audio**2))
            
            # Zero Crossing Rate
            features['zcr'] = float(librosa.feature.zero_crossing_rate(audio)[0, 0])
            
            # Спектральные фичи
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_std'] = float(np.std(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            for i in range(mfccs.shape[0]):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            # Pitch
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate
            )
            f0 = f0[~np.isnan(f0)]
            if len(f0) > 0:
                features['pitch_mean'] = float(np.mean(f0))
                features['pitch_std'] = float(np.std(f0))
                features['pitch_range'] = float(np.max(f0) - np.min(f0))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0
            
            # RMS energy
            features['rms'] = float(librosa.feature.rms(y=audio)[0, 0])
            
            # Tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
                features['tempo'] = float(tempo[0] if len(tempo) > 0 else 0)
            except:
                features['tempo'] = 0.0
            
            features['processing_time'] = time.time() - start_time
            features['success'] = True
            
            return features
            
        except Exception as e:
            logger.error(f"Ошибка извлечения фич: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }
    
    def process_audio(self, audio_data: np.ndarray, sample_rate: int, 
                     language: str = None, full_analysis: bool = True) -> Dict:
        """
        Полная обработка аудио
        
        Args:
            audio_data: Аудио данные
            sample_rate: Частота дискретизации
            language: Язык (опционально)
            full_analysis: Полный анализ или только транскрипция
            
        Returns:
            Полный результат обработки
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Предобработка аудио
            preprocess_result = self.preprocess_audio(audio_data, sample_rate)
            
            if not preprocess_result['success']:
                raise Exception(f"Ошибка предобработки: {preprocess_result.get('error')}")
            
            processed_audio = preprocess_result['audio']
            processed_sample_rate = preprocess_result['sample_rate']
            
            # Транскрипция
            transcription_result = self.transcribe_audio(
                processed_audio, 
                processed_sample_rate, 
                language
            )
            
            result = {
                'preprocessing': preprocess_result,
                'transcription': transcription_result,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            # Полный анализ если требуется
            if full_analysis and transcription_result['success']:
                # VAD
                vad_segments = self.detect_voice_activity(processed_audio, processed_sample_rate)
                result['vad'] = {
                    'segments': vad_segments,
                    'total_segments': len(vad_segments)
                }
                
                # Анализ эмоций
                emotion_result = self.analyze_emotions(processed_audio, processed_sample_rate)
                result['emotion_analysis'] = emotion_result
                
                # Wake word detection
                wake_word_result = self.detect_wake_word(processed_audio, processed_sample_rate)
                result['wake_word'] = wake_word_result
                
                # Извлечение фич
                features = self.extract_audio_features(processed_audio, processed_sample_rate)
                result['audio_features'] = features
                
                # Диаризация (если аудио достаточно длинное)
                if preprocess_result['duration'] > 5.0:  # Минимум 5 секунд
                    diarization_result = self.diarize_speakers(processed_audio, processed_sample_rate)
                    result['diarization'] = diarization_result
            
            # Обновление статистики
            self.stats['successful_requests'] += 1
            self.stats['total_processing_time'] += result['processing_time']
            self.stats['audio_duration_processed'] += preprocess_result['duration']
            
            logger.info(f"✅ Аудио обработано за {result['processing_time']:.2f} секунд")
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"❌ Ошибка обработки аудио: {e}")
            
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
    
    def process_audio_file(self, file_path: str, language: str = None, 
                          full_analysis: bool = True) -> Dict:
        """
        Обработка аудио файла
        
        Args:
            file_path: Путь к аудио файлу
            language: Язык (опционально)
            full_analysis: Полный анализ
            
        Returns:
            Результат обработки
        """
        try:
            # Загрузка аудио файла
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            
            # Обработка
            result = self.process_audio(audio_data, sample_rate, language, full_analysis)
            result['file_path'] = file_path
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {e}")
            return {
                'error': str(e),
                'file_path': file_path,
                'success': False
            }
    
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
        
        models_loaded = {
            name: model is not None 
            for name, model in self.models.items()
        }
        
        return {
            'status': 'healthy',
            'service': 'voice_processor',
            'version': '3.0.0',
            'gpu_available': gpu_available,
            'gpu_info': gpu_info if gpu_available else None,
            'models_loaded': models_loaded,
            'cache_size': len(self.cache),
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup_cache(self, max_age_seconds: int = None):
        """Очистка кэша от старых записей"""
        if max_age_seconds is None:
            max_age_seconds = self.config.CACHE_TTL
        
        current_time = time.time()
        keys_to_delete = []
        
        for key, value in self.cache.items():
            if current_time - value['timestamp'] > max_age_seconds:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.cache[key]
        
        logger.info(f"Очищено записей из кэша: {len(keys_to_delete)}")
    
    def cleanup(self):
        """Очистка ресурсов"""
        logger.info("Очистка ресурсов VoiceProcessor...")
        
        # Очищаем модели
        for name in list(self.models.keys()):
            if hasattr(self.models[name], 'to'):
                self.models[name].to('cpu')
            del self.models[name]
        
        # Очищаем кэш
        self.cache.clear()
        
        # Очищаем GPU память если используется
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ VoiceProcessor остановлен")

# REST API для Voice Processor
class VoiceProcessorAPI:
    """REST API для Voice Processor"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        from flask import Flask, request, jsonify, send_file
        from flask_cors import CORS
        
        self.host = host
        self.port = port
        self.processor = VoiceProcessor()
        
        # Инициализация Flask
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Увеличение максимального размера файла
        self.app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
        
        # Регистрация маршрутов
        self._register_routes()
        
        logger.info(f"VoiceProcessorAPI инициализирован на {host}:{port}")
    
    def _register_routes(self):
        """Регистрация API маршрутов"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            status = self.processor.get_health_status()
            return jsonify(status)
        
        @self.app.route('/api/v1/voice/process', methods=['POST'])
        def process_voice():
            """Обработка голосового аудио"""
            start_time = time.time()
            
            try:
                # Проверяем наличие файла или данных
                if 'file' in request.files:
                    # Загрузка файла
                    audio_file = request.files['file']
                    file_ext = os.path.splitext(audio_file.filename)[1].lower()
                    
                    # Сохраняем временный файл
                    temp_file = self.processor.config.TEMP_DIR / f"upload_{int(time.time())}{file_ext}"
                    audio_file.save(str(temp_file))
                    
                    # Обработка файла
                    language = request.form.get('language')
                    full_analysis = request.form.get('full_analysis', 'true').lower() == 'true'
                    
                    result = self.processor.process_audio_file(
                        str(temp_file),
                        language=language,
                        full_analysis=full_analysis
                    )
                    
                    # Удаляем временный файл
                    temp_file.unlink(missing_ok=True)
                    
                elif 'audio' in request.json:
                    # Аудио данные в base64 или массив
                    audio_data = request.json['audio']
                    sample_rate = request.json.get('sample_rate', 16000)
                    language = request.json.get('language')
                    full_analysis = request.json.get('full_analysis', True)
                    
                    # Конвертация base64 если нужно
                    if isinstance(audio_data, str) and audio_data.startswith('data:'):
                        import base64
                        # Извлекаем base64 часть
                        audio_b64 = audio_data.split(',')[1]
                        audio_bytes = base64.b64decode(audio_b64)
                        
                        # Конвертируем в numpy array
                        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    elif isinstance(audio_data, list):
                        # Массив чисел
                        audio_np = np.array(audio_data, dtype=np.float32)
                    else:
                        return jsonify({
                            'error': 'Invalid audio format',
                            'success': False
                        }), 400
                    
                    # Обработка
                    result = self.processor.process_audio(
                        audio_np,
                        sample_rate,
                        language=language,
                        full_analysis=full_analysis
                    )
                    
                else:
                    return jsonify({
                        'error': 'No audio data provided',
                        'success': False
                    }), 400
                
                # Добавляем время обработки API
                result['api_processing_time'] = time.time() - start_time
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
                return jsonify({
                    'error': str(e),
                    'api_processing_time': time.time() - start_time,
                    'success': False
                }), 500
        
        @self.app.route('/api/v1/voice/transcribe', methods=['POST'])
        def transcribe():
            """Только транскрипция аудио"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                audio_file = request.files['file']
                language = request.form.get('language')
                
                # Сохраняем временный файл
                temp_file = self.processor.config.TEMP_DIR / f"transcribe_{int(time.time())}.wav"
                audio_file.save(str(temp_file))
                
                # Загрузка аудио
                audio_data, sample_rate = librosa.load(str(temp_file), sr=None)
                
                # Транскрипция
                result = self.processor.transcribe_audio(audio_data, sample_rate, language)
                
                # Удаляем временный файл
                temp_file.unlink(missing_ok=True)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Ошибка транскрипции: {e}")
                return jsonify({'error': str(e), 'success': False}), 500
        
        @self.app.route('/api/v1/voice/detect-wakeword', methods=['POST'])
        def detect_wakeword():
            """Детекция ключевого слова"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                audio_file = request.files['file']
                
                # Сохраняем временный файл
                temp_file = self.processor.config.TEMP_DIR / f"wakeword_{int(time.time())}.wav"
                audio_file.save(str(temp_file))
                
                # Загрузка аудио
                audio_data, sample_rate = librosa.load(str(temp_file), sr=None)
                
                # Детекция
                result = self.processor.detect_wake_word(audio_data, sample_rate)
                
                # Удаляем временный файл
                temp_file.unlink(missing_ok=True)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Ошибка детекции wake word: {e}")
                return jsonify({'error': str(e), 'success': False}), 500
        
        @self.app.route('/api/v1/voice/analyze-emotions', methods=['POST'])
        def analyze_emotions():
            """Анализ эмоций в голосе"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                audio_file = request.files['file']
                
                # Сохраняем временный файл
                temp_file = self.processor.config.TEMP_DIR / f"emotions_{int(time.time())}.wav"
                audio_file.save(str(temp_file))
                
                # Загрузка аудио
                audio_data, sample_rate = librosa.load(str(temp_file), sr=None)
                
                # Анализ эмоций
                result = self.processor.analyze_emotions(audio_data, sample_rate)
                
                # Удаляем временный файл
                temp_file.unlink(missing_ok=True)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Ошибка анализа эмоций: {e}")
                return jsonify({'error': str(e), 'success': False}), 500
        
        @self.app.route('/api/v1/voice/stats', methods=['GET'])
        def get_stats():
            """Получение статистики сервиса"""
            return jsonify(self.processor.stats)
        
        @self.app.route('/api/v1/voice/cleanup-cache', methods=['POST'])
        def cleanup_cache():
            """Очистка кэша"""
            max_age = request.json.get('max_age_seconds')
            self.processor.cleanup_cache(max_age)
            
            return jsonify({
                'success': True,
                'message': 'Cache cleaned',
                'timestamp': datetime.now().isoformat()
            })
    
    def run(self, debug: bool = False):
        """
        Запуск API сервера
        
        Args:
            debug: Режим отладки
        """
        logger.info(f"Запуск VoiceProcessorAPI на {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)
    
    def stop(self):
        """Остановка сервиса"""
        self.processor.cleanup()
        logger.info("✅ VoiceProcessorAPI остановлен")

# FastAPI версия
class FastVoiceProcessorAPI:
    """FastAPI версия Voice Processor"""
    
    def __init__(self):
        from fastapi import FastAPI, File, UploadFile, Form, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        import tempfile
        
        self.app = FastAPI(
            title="ARIS Neuro Voice Processor API",
            version="3.0.0",
            description="API для обработки голосовых данных с использованием ML"
        )
        
        self.processor = VoiceProcessor()
        
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
        
        logger.info("FastVoiceProcessorAPI инициализирован")
    
    def _register_routes(self):
        """Регистрация FastAPI маршрутов"""
        from fastapi import APIRouter
        from pydantic import BaseModel
        from typing import Optional, List
        
        router = APIRouter(prefix="/api/v1/voice", tags=["Voice"])
        
        # Модели запросов
        class ProcessRequest(BaseModel):
            audio: Optional[List[float]] = None
            sample_rate: int = 16000
            language: Optional[str] = None
            full_analysis: bool = True
        
        @router.get("/health")
        async def health():
            return self.processor.get_health_status()
        
        @router.post("/process")
        async def process_voice(
            file: Optional[UploadFile] = File(None),
            language: Optional[str] = Form(None),
            full_analysis: bool = Form(True)
        ):
            try:
                if file:
                    # Создаем временный файл
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        content = await file.read()
                        tmp.write(content)
                        tmp_path = tmp.name
                    
                    try:
                        result = self.processor.process_audio_file(
                            tmp_path,
                            language=language,
                            full_analysis=full_analysis
                        )
                        return result
                    finally:
                        # Удаляем временный файл
                        import os
                        os.unlink(tmp_path)
                else:
                    raise HTTPException(status_code=400, detail="No audio file provided")
                    
            except Exception as e:
                logger.error(f"Ошибка обработки: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.post("/transcribe")
        async def transcribe(
            file: UploadFile = File(...),
            language: Optional[str] = Form(None)
        ):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name
                
                try:
                    audio_data, sample_rate = librosa.load(tmp_path, sr=None)
                    result = self.processor.transcribe_audio(
                        audio_data, sample_rate, language
                    )
                    return result
                finally:
                    import os
                    os.unlink(tmp_path)
                    
            except Exception as e:
                logger.error(f"Ошибка транскрипции: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/stats")
        async def get_stats():
            return self.processor.stats
        
        # Регистрируем роутер
        self.app.include_router(router)
    
    def run(self, host: str = "0.0.0.0", port: int = 5000):
        """
        Запуск FastAPI сервера
        
        Args:
            host: Хост
            port: Порт
        """
        import uvicorn
        logger.info(f"Запуск FastVoiceProcessorAPI на {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)
    
    def stop(self):
        """Остановка сервиса"""
        self.processor.cleanup()
        logger.info("✅ FastVoiceProcessorAPI остановлен")

# Точка входа
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ARIS Neuro Voice Processor")
    parser.add_argument("--api", choices=["flask", "fastapi"], default="fastapi",
                       help="Тип API (flask или fastapi)")
    parser.add_argument("--host", default="0.0.0.0", help="Хост для API")
    parser.add_argument("--port", type=int, default=5000, help="Порт для API")
    parser.add_argument("--debug", action="store_true", help="Режим отладки")
    
    args = parser.parse_args()
    
    try:
        if args.api == "flask":
            api = VoiceProcessorAPI(host=args.host, port=args.port)
            api.run(debug=args.debug)
        else:
            api = FastVoiceProcessorAPI()
            api.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания...")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
    finally:
        if 'api' in locals():
            api.stop()
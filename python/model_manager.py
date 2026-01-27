"""
Менеджер ML моделей для ARIS Neuro
Управление загрузкой, кэшированием и версионированием моделей
"""

import os
import sys
import json
import logging
import hashlib
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import pickle

try:
    import torch
    import torch.nn as nn
    import numpy as np
    import whisper
    from transformers import AutoModel, AutoTokenizer
    import onnxruntime as ort
except ImportError as e:
    print(f"⚠️  Некоторые ML библиотеки не установлены: {e}")
    print("Установите зависимости: pip install -r requirements.txt")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Информация о модели"""
    name: str
    type: str  # whisper, tts, wakeword, emotion, diarization
    version: str
    path: str
    size_mb: float
    loaded: bool = False
    device: str = "cpu"
    memory_mb: float = 0.0
    load_time: float = 0.0
    last_used: Optional[datetime] = None
    metadata: Dict = None

class ModelManager:
    """Менеджер для управления ML моделями"""
    
    def __init__(self, models_dir: str = None, cache_dir: str = None):
        """
        Инициализация менеджера моделей
        
        Args:
            models_dir: Директория с моделями
            cache_dir: Директория для кэша
        """
        # Определяем директории
        self.models_dir = Path(models_dir or os.getenv('MODELS_DIR', '/app/models'))
        self.cache_dir = Path(cache_dir or os.getenv('CACHE_DIR', '/app/cache/models'))
        
        # Создаем директории
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Загруженные модели
        self.loaded_models: Dict[str, Any] = {}
        
        # Информация о моделях
        self.models_info: Dict[str, ModelInfo] = {}
        
        # Блокировка для потокобезопасности
        self.lock = threading.RLock()
        
        # Конфигурация
        self.config = {
            'max_loaded_models': 10,
            'auto_unload_timeout': 3600,  # 1 час
            'cache_enabled': True,
            'gpu_memory_limit': 0.8,  # 80% GPU памяти
            'preload_models': ['whisper-medium', 'tts-default']
        }
        
        # Загружаем информацию о моделях
        self._load_models_info()
        
        logger.info(f"ModelManager инициализирован. Модели: {self.models_dir}, Кэш: {self.cache_dir}")
    
    def _load_models_info(self):
        """Загрузка информации о доступных моделях"""
        info_file = self.cache_dir / 'models_info.json'
        
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    data = json.load(f)
                    for name, info_dict in data.items():
                        self.models_info[name] = ModelInfo(**info_dict)
                logger.info(f"Загружена информация о {len(self.models_info)} моделях")
            except Exception as e:
                logger.warning(f"Не удалось загрузить информацию о моделях: {e}")
    
    def _save_models_info(self):
        """Сохранение информации о моделях"""
        info_file = self.cache_dir / 'models_info.json'
        
        try:
            data = {
                name: asdict(info) 
                for name, info in self.models_info.items()
            }
            
            # Конвертируем datetime в строки
            for name, info_dict in data.items():
                if info_dict.get('last_used'):
                    info_dict['last_used'] = info_dict['last_used'].isoformat()
            
            with open(info_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Не удалось сохранить информацию о моделях: {e}")
    
    def register_model(
        self,
        name: str,
        model_type: str,
        path: str,
        version: str = "1.0.0",
        metadata: Dict = None
    ) -> ModelInfo:
        """
        Регистрация новой модели
        
        Args:
            name: Имя модели
            model_type: Тип модели
            path: Путь к модели
            version: Версия модели
            metadata: Дополнительные метаданные
            
        Returns:
            Информация о модели
        """
        with self.lock:
            # Проверяем существование файла
            model_path = Path(path)
            if not model_path.exists():
                raise FileNotFoundError(f"Модель не найдена: {path}")
            
            # Вычисляем размер
            size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # Создаем информацию о модели
            model_info = ModelInfo(
                name=name,
                type=model_type,
                version=version,
                path=str(model_path.absolute()),
                size_mb=size_mb,
                metadata=metadata or {}
            )
            
            # Сохраняем
            self.models_info[name] = model_info
            self._save_models_info()
            
            logger.info(f"✅ Модель зарегистрирована: {name} ({model_type}, {size_mb:.2f} MB)")
            
            return model_info
    
    def load_model(
        self,
        name: str,
        device: str = None,
        force_reload: bool = False
    ) -> Any:
        """
        Загрузка модели
        
        Args:
            name: Имя модели
            device: Устройство (cuda, cpu, auto)
            force_reload: Принудительная перезагрузка
            
        Returns:
            Загруженная модель
        """
        with self.lock:
            # Проверяем, не загружена ли уже модель
            if name in self.loaded_models and not force_reload:
                model_info = self.models_info.get(name)
                if model_info:
                    model_info.last_used = datetime.now()
                logger.debug(f"Модель {name} уже загружена")
                return self.loaded_models[name]
            
            # Проверяем наличие информации о модели
            if name not in self.models_info:
                raise ValueError(f"Модель {name} не зарегистрирована")
            
            model_info = self.models_info[name]
            
            # Определяем устройство
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Проверяем лимит загруженных моделей
            if len(self.loaded_models) >= self.config['max_loaded_models']:
                self._unload_oldest_model()
            
            start_time = time.time()
            
            try:
                # Загружаем модель в зависимости от типа
                model = self._load_model_by_type(model_info, device)
                
                # Обновляем информацию
                model_info.loaded = True
                model_info.device = device
                model_info.load_time = time.time() - start_time
                model_info.last_used = datetime.now()
                
                # Вычисляем использование памяти
                if device == "cuda" and torch.cuda.is_available():
                    model_info.memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                
                # Сохраняем модель
                self.loaded_models[name] = model
                self._save_models_info()
                
                logger.info(
                    f"✅ Модель {name} загружена на {device} "
                    f"за {model_info.load_time:.2f} сек"
                )
                
                return model
                
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки модели {name}: {e}")
                raise
    
    def _load_model_by_type(self, model_info: ModelInfo, device: str) -> Any:
        """Загрузка модели по типу"""
        model_type = model_info.type.lower()
        model_path = Path(model_info.path)
        
        if model_type == "whisper":
            # Загрузка Whisper модели
            model_name = model_info.metadata.get('whisper_model', 'medium')
            model = whisper.load_model(
                name=model_name,
                device=device,
                download_root=str(self.models_dir / "whisper")
            )
            
        elif model_type == "tts":
            # Загрузка TTS модели
            # Здесь может быть Coqui TTS, VITS и т.д.
            if model_path.suffix in ['.pt', '.pth']:
                model = torch.load(model_path, map_location=device)
            else:
                # Используем transformers для TTS
                from TTS.api import TTS
                model = TTS(model_name=model_info.metadata.get('tts_model', 'tts_models/ru/ru_ruslan'))
            
        elif model_type == "wakeword":
            # Загрузка модели wake word detection
            if model_path.suffix in ['.pt', '.pth']:
                model = torch.jit.load(str(model_path), map_location=device)
                model.eval()
            elif model_path.suffix == '.onnx':
                model = ort.InferenceSession(str(model_path))
            elif model_path.suffix == '.tflite':
                import tflite_runtime.interpreter as tflite
                model = tflite.Interpreter(model_path=str(model_path))
                model.allocate_tensors()
            else:
                raise ValueError(f"Неподдерживаемый формат модели: {model_path.suffix}")
            
        elif model_type == "emotion":
            # Загрузка модели анализа эмоций
            from transformers import pipeline
            model = pipeline(
                "audio-classification",
                model=model_info.metadata.get('emotion_model', 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'),
                device=0 if device == "cuda" else -1
            )
            
        elif model_type == "diarization":
            # Загрузка модели диаризации
            from pyannote.audio import Pipeline
            model = Pipeline.from_pretrained(
                model_info.metadata.get('diarization_model', 'pyannote/speaker-diarization'),
                use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
            ).to(torch.device(device))
            
        elif model_type == "pytorch":
            # Общая PyTorch модель
            model = torch.load(model_path, map_location=device)
            if hasattr(model, 'eval'):
                model.eval()
                
        elif model_type == "onnx":
            # ONNX модель
            model = ort.InferenceSession(
                str(model_path),
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
            )
            
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        return model
    
    def unload_model(self, name: str) -> bool:
        """
        Выгрузка модели из памяти
        
        Args:
            name: Имя модели
            
        Returns:
            True если модель была выгружена
        """
        with self.lock:
            if name not in self.loaded_models:
                logger.warning(f"Модель {name} не загружена")
                return False
            
            try:
                model = self.loaded_models[name]
                
                # Очищаем память GPU если используется
                if self.models_info[name].device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Удаляем модель
                del self.loaded_models[name]
                
                # Обновляем информацию
                if name in self.models_info:
                    self.models_info[name].loaded = False
                    self.models_info[name].memory_mb = 0.0
                
                self._save_models_info()
                
                logger.info(f"✅ Модель {name} выгружена из памяти")
                return True
                
            except Exception as e:
                logger.error(f"❌ Ошибка выгрузки модели {name}: {e}")
                return False
    
    def _unload_oldest_model(self):
        """Выгрузка самой старой неиспользуемой модели"""
        if not self.loaded_models:
            return
        
        # Находим модель с самым старым временем использования
        oldest_name = None
        oldest_time = None
        
        for name, model_info in self.models_info.items():
            if name in self.loaded_models and model_info.last_used:
                if oldest_time is None or model_info.last_used < oldest_time:
                    oldest_time = model_info.last_used
                    oldest_name = name
        
        if oldest_name:
            logger.info(f"Автоматическая выгрузка старой модели: {oldest_name}")
            self.unload_model(oldest_name)
    
    def get_model(self, name: str) -> Optional[Any]:
        """
        Получение загруженной модели
        
        Args:
            name: Имя модели
            
        Returns:
            Модель или None
        """
        with self.lock:
            return self.loaded_models.get(name)
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """
        Получение информации о модели
        
        Args:
            name: Имя модели
            
        Returns:
            Информация о модели или None
        """
        return self.models_info.get(name)
    
    def list_models(self, model_type: str = None) -> List[ModelInfo]:
        """
        Список всех моделей
        
        Args:
            model_type: Фильтр по типу модели
            
        Returns:
            Список информации о моделях
        """
        models = list(self.models_info.values())
        
        if model_type:
            models = [m for m in models if m.type == model_type]
        
        return models
    
    def list_loaded_models(self) -> List[str]:
        """Список загруженных моделей"""
        return list(self.loaded_models.keys())
    
    def get_stats(self) -> Dict:
        """Получение статистики менеджера моделей"""
        total_models = len(self.models_info)
        loaded_models = len(self.loaded_models)
        
        total_size = sum(m.size_mb for m in self.models_info.values())
        loaded_size = sum(
            m.memory_mb for m in self.models_info.values() 
            if m.name in self.loaded_models
        )
        
        models_by_type = {}
        for model_info in self.models_info.values():
            model_type = model_info.type
            models_by_type[model_type] = models_by_type.get(model_type, 0) + 1
        
        return {
            'total_models': total_models,
            'loaded_models': loaded_models,
            'total_size_mb': total_size,
            'loaded_size_mb': loaded_size,
            'models_by_type': models_by_type,
            'cache_enabled': self.config['cache_enabled'],
            'max_loaded_models': self.config['max_loaded_models']
        }
    
    def cleanup_unused_models(self, max_age_seconds: int = None):
        """
        Очистка неиспользуемых моделей
        
        Args:
            max_age_seconds: Максимальный возраст неиспользования
        """
        if max_age_seconds is None:
            max_age_seconds = self.config['auto_unload_timeout']
        
        current_time = datetime.now()
        unloaded = 0
        
        with self.lock:
            for name, model_info in self.models_info.items():
                if name in self.loaded_models and model_info.last_used:
                    age = (current_time - model_info.last_used).total_seconds()
                    if age > max_age_seconds:
                        self.unload_model(name)
                        unloaded += 1
        
        if unloaded > 0:
            logger.info(f"Очищено неиспользуемых моделей: {unloaded}")
    
    def preload_models(self, model_names: List[str] = None):
        """
        Предзагрузка моделей
        
        Args:
            model_names: Список имен моделей для предзагрузки
        """
        if model_names is None:
            model_names = self.config['preload_models']
        
        logger.info(f"Предзагрузка моделей: {model_names}")
        
        for name in model_names:
            try:
                if name in self.models_info:
                    self.load_model(name)
                else:
                    logger.warning(f"Модель {name} не найдена для предзагрузки")
            except Exception as e:
                logger.error(f"Ошибка предзагрузки модели {name}: {e}")
    
    def cleanup(self):
        """Очистка всех ресурсов"""
        logger.info("Очистка ModelManager...")
        
        with self.lock:
            # Выгружаем все модели
            for name in list(self.loaded_models.keys()):
                self.unload_model(name)
            
            # Сохраняем информацию
            self._save_models_info()
        
        logger.info("✅ ModelManager очищен")

# Глобальный экземпляр менеджера
_manager_instance = None

def get_manager() -> ModelManager:
    """Получение глобального экземпляра менеджера"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelManager()
    return _manager_instance

# Точка входа для тестирования
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Manager для ARIS Neuro")
    parser.add_argument("--action", choices=["list", "load", "unload", "stats", "register"], default="list")
    parser.add_argument("--name", help="Имя модели")
    parser.add_argument("--type", help="Тип модели")
    parser.add_argument("--path", help="Путь к модели")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    
    args = parser.parse_args()
    
    manager = get_manager()
    
    if args.action == "list":
        models = manager.list_models()
        print(f"\n📋 Доступные модели ({len(models)}):")
        for model in models:
            status = "✅ Загружена" if model.loaded else "⏸️  Не загружена"
            print(f"  - {model.name} ({model.type}) - {status}")
    
    elif args.action == "load":
        if not args.name:
            print("❌ Укажите имя модели: --name MODEL_NAME")
            sys.exit(1)
        try:
            model = manager.load_model(args.name, args.device)
            print(f"✅ Модель {args.name} загружена")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            sys.exit(1)
    
    elif args.action == "unload":
        if not args.name:
            print("❌ Укажите имя модели: --name MODEL_NAME")
            sys.exit(1)
        success = manager.unload_model(args.name)
        if success:
            print(f"✅ Модель {args.name} выгружена")
        else:
            print(f"❌ Не удалось выгрузить модель {args.name}")
            sys.exit(1)
    
    elif args.action == "stats":
        stats = manager.get_stats()
        print("\n📊 Статистика ModelManager:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.action == "register":
        if not all([args.name, args.type, args.path]):
            print("❌ Укажите --name, --type и --path")
            sys.exit(1)
        try:
            model_info = manager.register_model(args.name, args.type, args.path)
            print(f"✅ Модель зарегистрирована: {model_info.name}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            sys.exit(1)

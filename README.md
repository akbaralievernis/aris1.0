# ARIS Neuro v3.0 🧠

**Интеллектуальный голосовой ассистент нового поколения**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![Docker](https://img.shields.io/badge/Docker-✓-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-✓-326CE5.svg)](https://kubernetes.io/)

## 🌟 Возможности

- **🎤 Голосовое взаимодействие** - Распознавание и синтез речи в реальном времени
- **🧠 Мульти-провайдерный AI** - OpenAI, Mistral, Anthropic, локальные модели
- **💾 Контекстуальная память** - Долгосрочное хранение и извлечение контекста
- **⚡ Real-time коммуникация** - WebSocket для мгновенных ответов
- **📊 Мониторинг и аналитика** - Полный стек observability
- **🔒 Безопасность** - JWT аутентификация, rate limiting, шифрование
- **🐳 Масштабируемость** - Docker, Kubernetes, горизонтальное масштабирование

## 🚀 Быстрый старт

### Предварительные требования

- Node.js 18+
- Docker & Docker Compose
- Python 3.8+ (для ML моделей)
- MongoDB 6.0+
- Redis 7.0+

### Установка для разработки

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/aris-neuro.git
cd aris-neuro

# Установка зависимостей
cd backend/node
npm install

# Настройка окружения
cp .env.example .env.development
# Отредактируйте .env.development файл

# Запуск в development режиме
npm run dev
#!/bin/bash

# ARIS Neuro v3.0 - Скрипт настройки окружения
# Автоматическая настройка окружения для разработки

set -e

echo "🚀 ARIS Neuro v3.0 - Настройка окружения"
echo "========================================"

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Проверка зависимостей
check_dependency() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✅ $1 установлен${NC}"
        return 0
    else
        echo -e "${RED}❌ $1 не установлен${NC}"
        return 1
    fi
}

echo ""
echo "Проверка зависимостей..."
MISSING_DEPS=0

check_dependency node || MISSING_DEPS=1
check_dependency npm || MISSING_DEPS=1
check_dependency python3 || MISSING_DEPS=1
check_dependency pip3 || MISSING_DEPS=1
check_dependency docker || MISSING_DEPS=1
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✅ docker-compose установлен${NC}"
elif docker compose version &> /dev/null; then
    echo -e "${GREEN}✅ docker compose доступен${NC}"
else
    echo -e "${RED}❌ docker-compose или docker compose не установлен${NC}"
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${RED}Установите недостающие зависимости перед продолжением${NC}"
    exit 1
fi

# Проверка версий
echo ""
echo "Проверка версий..."
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}Требуется Node.js 18 или выше${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Требуется Python 3.10 или выше${NC}"
    exit 1
fi

# Создание директорий
echo ""
echo "Создание директорий..."
mkdir -p logs
mkdir -p uploads
mkdir -p temp
mkdir -p cache
mkdir -p backend/python/ml_models
mkdir -p backend/python/temp
mkdir -p backend/python/cache

# Установка Node.js зависимостей
echo ""
echo "Установка Node.js зависимостей..."
cd backend/node
if [ ! -d "node_modules" ]; then
    npm install
    echo -e "${GREEN}✅ Node.js зависимости установлены${NC}"
else
    echo -e "${YELLOW}⚠️  node_modules уже существует, пропускаем...${NC}"
fi
cd ../..

# Установка Python зависимостей
echo ""
echo "Установка Python зависимостей..."
cd backend/python
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Python зависимости установлены${NC}"
else
    echo -e "${YELLOW}⚠️  venv уже существует, активируем...${NC}"
    source venv/bin/activate
fi
cd ../..

# Создание .env файлов
echo ""
echo "Настройка переменных окружения..."

if [ ! -f ".env.development" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env.development
        echo -e "${GREEN}✅ Создан .env.development из .env.example${NC}"
        echo -e "${YELLOW}⚠️  Не забудьте отредактировать .env.development с вашими настройками${NC}"
    else
        echo -e "${YELLOW}⚠️  .env.example не найден, создайте .env.development вручную${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  .env.development уже существует${NC}"
fi

# Проверка MongoDB и Redis
echo ""
echo "Проверка подключения к базам данных..."

# MongoDB
if command -v mongosh &> /dev/null; then
    if mongosh --eval "db.version()" --quiet &> /dev/null; then
        echo -e "${GREEN}✅ MongoDB доступен${NC}"
    else
        echo -e "${YELLOW}⚠️  MongoDB не запущен или недоступен${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  mongosh не установлен, пропускаем проверку MongoDB${NC}"
fi

# Redis
if redis-cli ping &> /dev/null; then
    echo -e "${GREEN}✅ Redis доступен${NC}"
else
    echo -e "${YELLOW}⚠️  Redis не запущен или недоступен${NC}"
    echo -e "${YELLOW}   Запустите Redis: redis-server${NC}"
fi

# Итоги
echo ""
echo "========================================"
echo -e "${GREEN}✅ Настройка окружения завершена!${NC}"
echo ""
echo "Следующие шаги:"
echo "1. Отредактируйте .env.development с вашими настройками"
echo "2. Убедитесь, что MongoDB и Redis запущены"
echo "3. Запустите приложение:"
echo "   - Node.js: cd backend/node && npm start"
echo "   - Python: cd backend/python && source venv/bin/activate && python voice_processor.py"
echo "   - Или используйте Docker: docker compose up (или docker-compose up)"
echo ""

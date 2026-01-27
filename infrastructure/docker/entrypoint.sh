#!/bin/bash
# Entrypoint скрипт для ARIS Neuro Node.js сервиса

set -e

echo "🚀 Запуск ARIS Neuro v3.0..."

# Загрузка переменных окружения из файла если существует
if [ -f "/app/.env" ]; then
    echo "📄 Загружаю переменные окружения из .env файла..."
    export $(cat /app/.env | grep -v '^#' | xargs)
fi

# Проверка обязательных переменных окружения
required_vars=("JWT_SECRET" "MONGODB_URI")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Ошибка: переменная окружения $var не установлена"
        exit 1
    fi
done

# Создание директорий если не существуют
mkdir -p /app/logs /app/temp /app/uploads /app/backups

# Очистка старых временных файлов
find /app/temp -type f -mtime +1 -delete 2>/dev/null || true

# Настройка лимитов для Node.js
if [ -n "$MAX_MEMORY" ]; then
    export NODE_OPTIONS="--max-old-space-size=$MAX_MEMORY $NODE_OPTIONS"
fi

# Настройка количества воркеров для кластера
if [ "$NODE_ENV" = "production" ]; then
    export CLUSTER_WORKERS=${CLUSTER_WORKERS:-$(nproc)}
    echo "🖥️  Режим кластера: $CLUSTER_WORKERS воркеров"
else
    echo "🔧 Development режим: кластеризация отключена"
fi

# Ожидание подключения к MongoDB
if [ "$WAIT_FOR_DB" = "true" ]; then
    echo "⏳ Ожидание подключения к MongoDB..."
    timeout=60
    counter=0
    
    until nc -z $(echo $MONGODB_URI | sed -e 's|^[^/]*//||' -e 's|/.*$||' | cut -d: -f1) $(echo $MONGODB_URI | sed -e 's|^[^/]*//||' -e 's|/.*$||' | cut -d: -f2) 2>/dev/null
    do
        sleep 1
        counter=$((counter + 1))
        if [ $counter -ge $timeout ]; then
            echo "❌ Таймаут подключения к MongoDB"
            exit 1
        fi
    done
    echo "✅ MongoDB доступна"
fi

# Ожидание подключения к Redis
if [ "$WAIT_FOR_REDIS" = "true" ] && [ -n "$REDIS_URL" ]; then
    echo "⏳ Ожидание подключения к Redis..."
    timeout=30
    counter=0
    
    redis_host=$(echo $REDIS_URL | sed -e 's|^[^/]*//||' -e 's|/.*$||' | cut -d: -f1)
    redis_port=$(echo $REDIS_URL | sed -e 's|^[^/]*//||' -e 's|/.*$||' | cut -d: -f2)
    
    until nc -z $redis_host $redis_port 2>/dev/null
    do
        sleep 1
        counter=$((counter + 1))
        if [ $counter -ge $timeout ]; then
            echo "⚠️  Redis недоступен, продолжаем без кэша"
            break
        fi
    done
    echo "✅ Redis доступен"
fi

# Запуск миграций базы данных если необходимо
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "🔄 Выполнение миграций базы данных..."
    node backend/node/scripts/migrate.js
fi

# Предзагрузка моделей AI если необходимо
if [ "$PRELOAD_AI_MODELS" = "true" ]; then
    echo "🧠 Предзагрузка AI моделей..."
    node backend/node/scripts/preload-models.js &
fi

# Логирование информации о системе
echo "📊 Информация о системе:"
echo "   Node.js: $(node --version)"
echo "   NPM: $(npm --version)"
echo "   Память: $(free -h | awk '/^Mem:/ {print $2}')"
echo "   CPU: $(nproc) ядер"
echo "   Режим: $NODE_ENV"

# Запуск приложения
echo "🚀 Запуск основного приложения..."
exec "$@"
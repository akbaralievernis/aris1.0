#!/bin/bash
# Healthcheck скрипт для ARIS Neuro

# URL для health check
HEALTH_URL="http://localhost:${PORT:-3000}/health"

# Пытаемся получить health status
response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$HEALTH_URL")

if [ "$response" = "200" ]; then
    # Дополнительная проверка состояния сервисов
    health_data=$(curl -s --connect-timeout 2 "$HEALTH_URL")
    
    # Парсим JSON ответ
    status=$(echo "$health_data" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    uptime=$(echo "$health_data" | grep -o '"uptime":[^,]*' | cut -d':' -f2)
    
    if [ "$status" = "healthy" ] && [ "${uptime%.*}" -gt 10 ]; then
        echo "✅ Health check пройден: status=$status, uptime=$uptime"
        exit 0
    else
        echo "⚠️  Health check: сервис работает но состояние не healthy"
        exit 1
    fi
else
    echo "❌ Health check не пройден: HTTP $response"
    exit 1
fiч
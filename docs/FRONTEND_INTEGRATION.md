# 🔗 Интеграция фронтенда с бэкендом

## ✅ Проверка интеграции

### 1. **API Endpoints**

Фронтенд настроен на использование следующих endpoints:

```typescript
// Базовый URL API
VITE_API_URL=http://localhost:3000/api/v1

// WebSocket URL (Socket.IO)
VITE_WS_URL=http://localhost:3000
```

### 2. **CORS Настройки**

Бэкенд настроен для работы с фронтендом:

```javascript
// backend/node/server.js
app.use(cors({
  origin: config.security.CORS_ORIGIN === '*' ? '*' : config.security.CORS_ORIGIN,
  credentials: config.security.CORS_CREDENTIALS,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));
```

**Важно:** Убедитесь, что в `.env.development` указан правильный `FRONTEND_URL`:

```env
FRONTEND_URL=http://localhost:5173  # или ваш порт фронтенда
CORS_ORIGIN=http://localhost:5173   # или * для разработки
CORS_CREDENTIALS=true
```

### 3. **WebSocket (Socket.IO)**

Фронтенд и бэкенд используют Socket.IO:

**Фронтенд:**
```typescript
// frontend/src/api/websocket.ts
const wsUrl = 'http://localhost:3000';
this.socket = io(wsUrl, {
  path: '/api/v1/ws',
  transports: ['websocket', 'polling'],
  auth: { token }
});
```

**Бэкенд:**
```javascript
// backend/node/websocket/server.js
this.io = new Server(httpServer, {
  path: '/api/v1/ws',
  cors: {
    origin: process.env.CORS_ORIGIN || '*',
    credentials: true
  }
});
```

### 4. **Аутентификация**

Фронтенд автоматически добавляет JWT токен в заголовки:

```typescript
// frontend/src/api/client.ts
config.headers.Authorization = `Bearer ${token}`;
```

Бэкенд проверяет токен через middleware:

```javascript
// backend/node/server.js
apiRouter.use(authenticate); // Для защищенных роутов
```

## 📋 Доступные API Endpoints

### Аутентификация
- `POST /api/v1/auth/register` - Регистрация
- `POST /api/v1/auth/login` - Вход
- `POST /api/v1/auth/refresh` - Обновление токена
- `GET /api/v1/auth/me` - Получение текущего пользователя
- `PUT /api/v1/auth/profile` - Обновление профиля
- `POST /api/v1/auth/logout` - Выход

### AI
- `POST /api/v1/ai/chat` - Чат с AI
- `POST /api/v1/ai/complete` - Завершение текста
- `GET /api/v1/ai/providers` - Список провайдеров
- `GET /api/v1/ai/models` - Список моделей

### Голос
- `POST /api/v1/voice/transcribe` - Транскрипция
- `POST /api/v1/voice/synthesize` - Синтез речи
- `POST /api/v1/voice/detect-wakeword` - Детекция wake word
- `GET /api/v1/voice/voices` - Список голосов

### Память
- `GET /api/v1/memory` - Получение памяти
- `POST /api/v1/memory` - Сохранение памяти
- `DELETE /api/v1/memory/:memoryId` - Удаление памяти
- `GET /api/v1/memory/stats` - Статистика памяти

### Проекты
- `GET /api/v1/projects` - Список проектов
- `GET /api/v1/projects/:projectId` - Получение проекта
- `POST /api/v1/projects` - Создание проекта
- `PUT /api/v1/projects/:projectId` - Обновление проекта
- `DELETE /api/v1/projects/:projectId` - Удаление проекта
- `POST /api/v1/projects/:projectId/restore` - Восстановление проекта
- `GET /api/v1/projects/:projectId/stats` - Статистика проекта
- `POST /api/v1/projects/:projectId/duplicate` - Дублирование проекта
- `GET /api/v1/projects/:projectId/export` - Экспорт проекта

### Настройки
- `GET /api/v1/settings` - Получение настроек
- `PUT /api/v1/settings` - Обновление настроек
- `PATCH /api/v1/settings/:key` - Обновление конкретной настройки
- `POST /api/v1/settings/reset` - Сброс настроек
- `GET /api/v1/settings/export` - Экспорт настроек
- `POST /api/v1/settings/import` - Импорт настроек

### Мониторинг
- `GET /api/v1/monitoring/stats` - Статистика системы
- `GET /api/v1/monitoring/health` - Health check
- `GET /api/v1/monitoring/metrics` - Метрики

## 🔌 WebSocket Events

### Отправка (Frontend → Backend)

```typescript
// AI чат
websocketClient.sendChatMessage('Привет, ARIS!', 'gpt-4', 0.7);

// Голосовой поток
websocketClient.sendVoiceStream(audioData, 'wav', 16000);

// Детекция wake word
websocketClient.sendWakeWordDetection(audioData, 0.7);

// Сохранение памяти
websocketClient.saveMemory('note', 'Важная заметка', ['важное']);

// Поиск в памяти
websocketClient.searchMemory('запрос', 'note', 10);

// Комнаты
websocketClient.joinRoom('project-123');
websocketClient.leaveRoom('project-123');
```

### Получение (Backend → Frontend)

```typescript
// Подписка на события
websocketClient.on('ai:response', (data) => {
  console.log('AI ответ:', data);
});

websocketClient.on('voice:processed', (data) => {
  console.log('Обработанный голос:', data);
});

websocketClient.on('wakeword:detected', (data) => {
  console.log('Wake word обнаружен:', data);
});

websocketClient.on('memory:saved', (data) => {
  console.log('Память сохранена:', data);
});

websocketClient.on('memory:results', (data) => {
  console.log('Результаты поиска:', data);
});

websocketClient.on('room:joined', (data) => {
  console.log('Присоединились к комнате:', data);
});

websocketClient.on('notification', (data) => {
  console.log('Уведомление:', data);
});
```

## 🚀 Запуск для разработки

### 1. Запуск бэкенда

```bash
cd backend/node
npm install
npm run dev
```

Бэкенд будет доступен на `http://localhost:3000`

### 2. Запуск фронтенда

```bash
cd frontend
npm install
npm run dev
```

Фронтенд будет доступен на `http://localhost:5173` (или другой порт Vite)

### 3. Настройка переменных окружения

**Фронтенд (.env):**
```env
VITE_API_URL=http://localhost:3000/api/v1
VITE_WS_URL=http://localhost:3000
VITE_APP_NAME=ARIS Neuro
VITE_APP_VERSION=3.0.0
```

**Бэкенд (.env.development):**
```env
FRONTEND_URL=http://localhost:5173
CORS_ORIGIN=http://localhost:5173
CORS_CREDENTIALS=true
```

## ✅ Чеклист интеграции

- [x] API клиент настроен на правильный URL
- [x] WebSocket клиент использует Socket.IO
- [x] CORS настроен для фронтенда
- [x] JWT токены передаются в заголовках
- [x] WebSocket аутентификация через токен
- [x] Все endpoints доступны
- [x] Обработка ошибок настроена
- [x] Автоматическое обновление токенов

## 🔍 Отладка

### Проверка подключения API

```bash
curl http://localhost:3000/health
```

### Проверка WebSocket

Откройте консоль браузера и проверьте:
```javascript
// В консоли браузера
websocketClient.connect();
websocketClient.onConnectionChange((connected) => {
  console.log('WebSocket connected:', connected);
});
```

### Проверка CORS

Если возникают CORS ошибки:
1. Убедитесь, что `FRONTEND_URL` в бэкенде совпадает с URL фронтенда
2. Проверьте, что `CORS_ORIGIN` включает URL фронтенда
3. Убедитесь, что `CORS_CREDENTIALS=true`

## 📝 Примечания

- Socket.IO автоматически использует WebSocket или polling в зависимости от поддержки браузером
- Токены обновляются автоматически при истечении через interceptor
- Все запросы логируются на бэкенде (если включено)
- WebSocket переподключается автоматически при разрыве соединения

---

**Интеграция готова!** 🎉


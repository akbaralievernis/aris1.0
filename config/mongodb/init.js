/**
 * Скрипт инициализации MongoDB
 * Создает пользователей, роли и индексы
 */

print('🔧 Инициализация MongoDB для ARIS Neuro v3.0...');

// Создаем административную базу данных
db = db.getSiblingDB('admin');

// Создаем пользователя администратора (если не существует)
const adminExists = db.getUsers({user: 'admin'}).users?.length > 0;
if (!adminExists) {
    print('👤 Создание пользователя администратора...');
    db.createUser({
        user: 'admin',
        pwd: 'aris_admin_password_123',
        roles: [
            { role: 'userAdminAnyDatabase', db: 'admin' },
            { role: 'readWriteAnyDatabase', db: 'admin' },
            { role: 'dbAdminAnyDatabase', db: 'admin' },
            { role: 'clusterAdmin', db: 'admin' }
        ]
    });
    print('✅ Пользователь администратора создан');
}

// Создаем базу данных приложения
db = db.getSiblingDB('aris_neuro');

// Создаем пользователя приложения
const appUserExists = db.getUsers({user: 'aris_app'}).users?.length > 0;
if (!appUserExists) {
    print('👤 Создание пользователя приложения...');
    db.createUser({
        user: 'aris_app',
        pwd: 'aris_app_password_123',
        roles: [
            { role: 'readWrite', db: 'aris_neuro' },
            { role: 'dbAdmin', db: 'aris_neuro' }
        ]
    });
    print('✅ Пользователь приложения создан');
}

// Создаем пользователя для мониторинга
const monitorUserExists = db.getUsers({user: 'monitor'}).users?.length > 0;
if (!monitorUserExists) {
    print('📊 Создание пользователя мониторинга...');
    db.createUser({
        user: 'monitor',
        pwd: 'monitor_password_123',
        roles: [
            { role: 'clusterMonitor', db: 'admin' },
            { role: 'read', db: 'aris_neuro' }
        ]
    });
    print('✅ Пользователь мониторинга создан');
}

// Создаем коллекции и индексы
print('📊 Создание коллекций и индексов...');

// Коллекция users
db.createCollection('users', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['email', 'passwordHash', 'createdAt'],
            properties: {
                email: {
                    bsonType: 'string',
                    description: 'Email пользователя (уникальный)'
                },
                username: {
                    bsonType: 'string',
                    description: 'Имя пользователя (уникальное)'
                },
                passwordHash: {
                    bsonType: 'string',
                    description: 'Хэш пароля'
                },
                status: {
                    bsonType: 'string',
                    enum: ['active', 'inactive', 'suspended', 'banned', 'deleted'],
                    description: 'Статус аккаунта'
                },
                roles: {
                    bsonType: 'array',
                    items: { bsonType: 'string' },
                    description: 'Роли пользователя'
                },
                permissions: {
                    bsonType: 'array',
                    items: { bsonType: 'string' },
                    description: 'Разрешения пользователя'
                },
                emailVerified: {
                    bsonType: 'bool',
                    description: 'Подтвержден ли email'
                },
                twoFactorEnabled: {
                    bsonType: 'bool',
                    description: 'Включена ли 2FA'
                },
                createdAt: {
                    bsonType: 'date',
                    description: 'Дата создания'
                },
                updatedAt: {
                    bsonType: 'date',
                    description: 'Дата обновления'
                }
            }
        }
    }
});

// Индексы для users
db.users.createIndex({ email: 1 }, { unique: true, name: 'email_unique' });
db.users.createIndex({ username: 1 }, { unique: true, sparse: true, name: 'username_unique' });
db.users.createIndex({ createdAt: -1 }, { name: 'created_at_desc' });
db.users.createIndex({ status: 1 }, { name: 'status_idx' });

// Коллекция api_keys
db.createCollection('api_keys', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['userId', 'key', 'name', 'createdAt'],
            properties: {
                userId: { bsonType: 'objectId' },
                key: { bsonType: 'string' },
                name: { bsonType: 'string' },
                description: { bsonType: 'string' },
                permissions: {
                    bsonType: 'array',
                    items: { bsonType: 'string' }
                },
                rateLimit: { bsonType: 'int' },
                expiresAt: { bsonType: 'date' },
                isActive: { bsonType: 'bool' },
                usageCount: { bsonType: 'int' },
                lastUsedAt: { bsonType: 'date' },
                createdAt: { bsonType: 'date' },
                revokedAt: { bsonType: 'date' }
            }
        }
    }
});

db.api_keys.createIndex({ key: 1 }, { unique: true, name: 'key_unique' });
db.api_keys.createIndex({ userId: 1 }, { name: 'user_api_keys_idx' });
db.api_keys.createIndex({ expiresAt: 1 }, { name: 'expires_at_idx' });
db.api_keys.createIndex({ isActive: 1 }, { name: 'active_idx' });

// Коллекция refresh_tokens
db.createCollection('refresh_tokens', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['userId', 'token', 'expiresAt', 'createdAt'],
            properties: {
                userId: { bsonType: 'objectId' },
                token: { bsonType: 'string' },
                userAgent: { bsonType: 'string' },
                ip: { bsonType: 'string' },
                expiresAt: { bsonType: 'date' },
                createdAt: { bsonType: 'date' }
            }
        }
    }
});

db.refresh_tokens.createIndex({ token: 1 }, { unique: true, name: 'token_unique' });
db.refresh_tokens.createIndex({ userId: 1 }, { name: 'user_tokens_idx' });
db.refresh_tokens.createIndex({ expiresAt: 1 }, { expireAfterSeconds: 0, name: 'expires_at_idx' });

// Коллекция conversations
db.createCollection('conversations', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['userId', 'userMessage', 'aiResponse', 'timestamp'],
            properties: {
                userId: { bsonType: 'objectId' },
                sessionId: { bsonType: 'string' },
                userMessage: { bsonType: 'string' },
                aiResponse: { bsonType: 'string' },
                model: { bsonType: 'string' },
                provider: { bsonType: 'string' },
                tokens: { bsonType: 'int' },
                timestamp: { bsonType: 'date' },
                metadata: { bsonType: 'object' }
            }
        }
    }
});

// Коллекция long_term_memories
db.createCollection('long_term_memories', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['userId', 'type', 'content', 'createdAt'],
            properties: {
                userId: { bsonType: 'objectId' },
                type: {
                    bsonType: 'string',
                    enum: ['memory', 'conversation', 'note', 'reminder', 'fact', 'preference']
                },
                content: { bsonType: 'string' },
                embedding: { bsonType: 'array', items: { bsonType: 'double' } },
                tags: {
                    bsonType: 'array',
                    items: { bsonType: 'string' }
                },
                importance: { bsonType: 'double', minimum: 0, maximum: 1 },
                accessCount: { bsonType: 'int' },
                createdAt: { bsonType: 'date' },
                updatedAt: { bsonType: 'date' },
                metadata: { bsonType: 'object' }
            }
        }
    }
});

// Коллекция voice_logs
db.createCollection('voice_logs', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['userId', 'timestamp'],
            properties: {
                userId: { bsonType: 'objectId' },
                clientId: { bsonType: 'string' },
                sessionId: { bsonType: 'string' },
                requestId: { bsonType: 'string' },
                duration: { bsonType: 'double' },
                transcription: { bsonType: 'object' },
                emotions: { bsonType: 'object' },
                features: { bsonType: 'object' },
                timestamp: { bsonType: 'date' },
                metadata: { bsonType: 'object' }
            }
        }
    }
});

// Коллекция ai_logs
db.createCollection('ai_logs', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['userId', 'provider', 'model', 'timestamp'],
            properties: {
                userId: { bsonType: 'objectId' },
                provider: {
                    bsonType: 'string',
                    enum: ['openai', 'mistral', 'anthropic', 'local']
                },
                model: { bsonType: 'string' },
                inputTokens: { bsonType: 'int' },
                outputTokens: { bsonType: 'int' },
                processingTime: { bsonType: 'int' },
                cost: { bsonType: 'double' },
                timestamp: { bsonType: 'date' },
                metadata: { bsonType: 'object' }
            }
        }
    }
});

// Коллекция request_logs
db.createCollection('request_logs', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['method', 'path', 'status', 'duration', 'timestamp'],
            properties: {
                method: {
                    bsonType: 'string',
                    enum: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD']
                },
                path: { bsonType: 'string' },
                status: { bsonType: 'int' },
                duration: { bsonType: 'int' },
                ip: { bsonType: 'string' },
                userId: { bsonType: 'objectId' },
                userAgent: { bsonType: 'string' },
                requestId: { bsonType: 'string' },
                timestamp: { bsonType: 'date' },
                metadata: { bsonType: 'object' }
            }
        }
    }
});

// Коллекция error_logs
db.createCollection('error_logs', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['error', 'timestamp'],
            properties: {
                error: { bsonType: 'string' },
                stack: { bsonType: 'string' },
                type: { bsonType: 'string' },
                status: { bsonType: 'int' },
                path: { bsonType: 'string' },
                method: { bsonType: 'string' },
                userId: { bsonType: 'objectId' },
                ip: { bsonType: 'string' },
                timestamp: { bsonType: 'date' }
            }
        }
    }
});

// Коллекция migrations
db.createCollection('migrations', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['name', 'executedAt', 'status'],
            properties: {
                name: { bsonType: 'string' },
                executedAt: { bsonType: 'date' },
                status: {
                    bsonType: 'string',
                    enum: ['pending', 'running', 'completed', 'failed', 'rolled_back']
                },
                error: { bsonType: 'string' },
                duration: { bsonType: 'int' }
            }
        }
    }
});

db.migrations.createIndex({ name: 1 }, { unique: true, name: 'migration_name_unique' });

// Коллекция health_check (для health checks)
db.createCollection('health_check', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            properties: {
                timestamp: { bsonType: 'date' }
            }
        }
    }
});

// Вставляем тестовую запись для health check
db.health_check.insertOne({ timestamp: new Date() });

print('🎉 Инициализация MongoDB завершена успешно!');
print('============================================');
print('Доступные базы данных:');
const dbs = db.adminCommand('listDatabases');
dbs.databases.forEach(dbInfo => {
    print(`  • ${dbInfo.name} (${dbInfo.sizeOnDisk} байт)`);
});
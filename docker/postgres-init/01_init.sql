-- =============================================================
-- Legal Agent 数据库初始化
-- 容器首次启动时自动执行此脚本
-- =============================================================

-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================
-- 用户表
-- =============================================================
CREATE TABLE IF NOT EXISTS users (
    id           BIGSERIAL PRIMARY KEY,
    external_id  TEXT UNIQUE NOT NULL,        -- 外部用户 ID(微信/手机号等)
    display_name TEXT,
    metadata     JSONB DEFAULT '{}',          -- 灵活扩展字段
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_external_id ON users(external_id);

-- =============================================================
-- 会话表
-- =============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     BIGINT REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);

-- =============================================================
-- 消息表
-- 注:简化版,M2 暂不用分区表,M9 流量起来再改
-- =============================================================
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

-- =============================================================
-- 向量记忆表(为 M3 RAG 预留)
-- =============================================================
CREATE TABLE IF NOT EXISTS embeddings (
    id          BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,                -- 'document' | 'memory' | 'kb'
    source_id   TEXT NOT NULL,                -- 来源标识
    content     TEXT NOT NULL,                -- 原始文本
    embedding   VECTOR(1024),                 -- BGE-M3 是 1024 维
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source_type, source_id);

-- HNSW 索引(M3 真正灌数据时再建,数据量小时建反而慢)
-- CREATE INDEX idx_embeddings_vector ON embeddings
--   USING hnsw (embedding vector_cosine_ops);

-- =============================================================
-- 自动更新 updated_at 字段的触发器
-- =============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;
CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================
-- 完成提示
-- =============================================================
DO $$
BEGIN
    RAISE NOTICE '✅ Legal Agent database initialized successfully';
END $$;

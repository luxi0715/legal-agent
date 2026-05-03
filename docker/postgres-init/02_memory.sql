-- =============================================================
-- M8 三层记忆架构
-- 容器首次启动时与 01_init.sql 一起执行(字典序)
-- 现有容器需手动 psql 执行一次
-- =============================================================

-- =============================================================
-- Hard Memory(用户身份事实持久化)
-- KV 设计:任意键值对,演化能力强
-- =============================================================
CREATE TABLE IF NOT EXISTS user_facts (
    id            BIGSERIAL PRIMARY KEY,
    user_id       UUID NOT NULL,                  -- 抽象用户标识(M8 起步 = session_id)
    key           VARCHAR(128) NOT NULL,          -- 例:location / occupation
    value         TEXT NOT NULL,                   -- 例:北京 / 程序员
    confidence    REAL NOT NULL DEFAULT 1.0,      -- LLM 抽取置信度 0~1
    extracted_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, key)                          -- 同一用户同一键唯一,新值覆盖
);

CREATE INDEX IF NOT EXISTS idx_user_facts_user_id ON user_facts(user_id);

-- =============================================================
-- Summary Memory(会话级长程摘要)
-- LLM 滚动压缩,Buffer 满 7 轮触发
-- =============================================================
CREATE TABLE IF NOT EXISTS session_summaries (
    session_id    UUID PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID NOT NULL,
    summary       TEXT NOT NULL DEFAULT '',
    last_updated  TIMESTAMPTZ DEFAULT NOW(),
    turn_count    INT NOT NULL DEFAULT 0           -- 累计轮数,触发摘要更新用
);

CREATE INDEX IF NOT EXISTS idx_session_summaries_user_id ON session_summaries(user_id);

-- =============================================================
-- 完成提示
-- =============================================================
DO $$
BEGIN
    RAISE NOTICE '✅ M8 memory tables initialized successfully';
END $$;

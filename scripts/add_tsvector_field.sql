-- M4.2.1: 给 embeddings 表加 tsvector 字段(基于 jieba 预分词)
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS content_tsv tsvector;

-- 创建 GIN 索引(适合全文检索)
CREATE INDEX IF NOT EXISTS idx_embeddings_content_tsv
    ON embeddings USING GIN(content_tsv);

-- 验证
SELECT
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'embeddings' AND column_name = 'content_tsv';

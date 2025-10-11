-- Initialize database with required extensions

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create a comment to document the schema
COMMENT ON DATABASE llm_monitoring IS 'LLM Operations Monitoring Database';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE llm_monitoring TO monitoring;

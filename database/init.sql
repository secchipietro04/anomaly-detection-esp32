-- timescaleDB extension init
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- relational tables

-- sequence for auto-incrementing model and ensemble IDs
CREATE SEQUENCE IF NOT EXISTS model_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS ensemble_id_seq START 1;

-- nodes table for registered sensors
CREATE TABLE IF NOT EXISTS nodes (
    node_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128),
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(32) NOT NULL DEFAULT 'registered',
    current_config JSONB DEFAULT '{}'::jsonb,
    last_trained_segment_id BIGINT NOT NULL DEFAULT 0
);

-- node capabilities reported via v1/+/info/caps
CREATE TABLE IF NOT EXISTS node_capabilities (
    node_id VARCHAR(64) PRIMARY KEY REFERENCES nodes(node_id) ON DELETE CASCADE,
    accel_freqs DOUBLE PRECISION[] NOT NULL DEFAULT '{}',
    gyro_freqs DOUBLE PRECISION[] NOT NULL DEFAULT '{}',
    enabled_ops TEXT[] NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- node health history reported via v1/+/info/health
CREATE TABLE IF NOT EXISTS node_health (
    id BIGSERIAL PRIMARY KEY,
    node_id VARCHAR(64) NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ram_free BIGINT NOT NULL DEFAULT 0,
    sd_status INTEGER NOT NULL DEFAULT 0,
    cached_models INTEGER[] NOT NULL DEFAULT '{}',
    last_segment_id BIGINT NOT NULL DEFAULT 0,
    status VARCHAR(64) NOT NULL DEFAULT 'idle',
    ips DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

-- models table for trained/deployed TFLite artifacts
CREATE TABLE IF NOT EXISTS models (
    id BIGINT PRIMARY KEY DEFAULT nextval('model_id_seq'),
    node_id VARCHAR(64) REFERENCES nodes(node_id) ON DELETE CASCADE,
    tag INTEGER,
    model_type INTEGER NOT NULL,
    tflite_binary BYTEA NOT NULL,
    size_bytes INTEGER NOT NULL,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ensembles table for routing configurations
CREATE TABLE IF NOT EXISTS ensembles (
    id BIGINT PRIMARY KEY DEFAULT nextval('ensemble_id_seq'),
    node_id VARCHAR(64) REFERENCES nodes(node_id) ON DELETE SET NULL,
    router_model_id BIGINT REFERENCES models(id) ON DELETE CASCADE,
    memory_model_id BIGINT REFERENCES models(id) ON DELETE SET NULL,
    warmup INTEGER NOT NULL DEFAULT 0,
    routes JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_size_bytes INTEGER NOT NULL DEFAULT 0,
    deployed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- time-series tables

-- raw telemetry segments and chunks
CREATE TABLE IF NOT EXISTS raw_telemetry (
    id BIGSERIAL,
    node_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    segment_id BIGINT NOT NULL,
    chunk_id INTEGER NOT NULL DEFAULT 1,
    sample_rate DOUBLE PRECISION NOT NULL,
    emit_reason INTEGER NOT NULL,
    accel_x REAL[] NOT NULL,
    accel_y REAL[] NOT NULL,
    accel_z REAL[] NOT NULL,
    gyro_x REAL[] NOT NULL,
    gyro_y REAL[] NOT NULL,
    gyro_z REAL[] NOT NULL,
    raw_bytes_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (node_id, timestamp, segment_id, chunk_id)
);

-- inference results with anomaly scores
CREATE TABLE IF NOT EXISTS inference_results (
    id BIGSERIAL,
    node_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    segment_id BIGINT NOT NULL,
    emit_reason INTEGER NOT NULL,
    router_model_id BIGINT,
    autoencoder_model_id BIGINT,
    mse DOUBLE PRECISION NOT NULL,
    anomaly BOOLEAN NOT NULL,
    inference_time_ms INTEGER,
    is_recalculated BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (node_id, timestamp, segment_id)
);

-- convert time-series tables to hypertables
SELECT create_hypertable('raw_telemetry', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('inference_results', 'timestamp', if_not_exists => TRUE);

-- create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_node_health_node_ts ON node_health (node_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_telemetry_node_ts ON raw_telemetry (node_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_telemetry_segment ON raw_telemetry (segment_id, chunk_id);
CREATE INDEX IF NOT EXISTS idx_raw_telemetry_node_segment ON raw_telemetry (node_id, segment_id);
CREATE INDEX IF NOT EXISTS idx_inference_results_node_ts ON inference_results (node_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_inference_results_anomaly ON inference_results (node_id, anomaly, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_inference_results_segment ON inference_results (segment_id);
CREATE INDEX IF NOT EXISTS idx_ensembles_node ON ensembles (node_id, deployed_at DESC);

-- enable timescaledb extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- nodes registry
CREATE TABLE IF NOT EXISTS nodes (
    node_id                 VARCHAR(64)  PRIMARY KEY,
    name                    VARCHAR(128),
    registered_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_seen               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    status                  VARCHAR(32)  NOT NULL DEFAULT 'registered',
    current_config          JSONB,
    last_trained_segment_id BIGINT       NOT NULL DEFAULT 0
);

-- node capabilities
CREATE TABLE IF NOT EXISTS node_capabilities (
    node_id     VARCHAR(64) PRIMARY KEY REFERENCES nodes(node_id) ON DELETE CASCADE,
    accel_freqs DOUBLE PRECISION[] NOT NULL DEFAULT '{}',
    gyro_freqs  DOUBLE PRECISION[] NOT NULL DEFAULT '{}',
    enabled_ops TEXT[]             NOT NULL DEFAULT '{}',
    updated_at  TIMESTAMPTZ        NOT NULL DEFAULT NOW()
);

-- health reports
CREATE TABLE IF NOT EXISTS node_health (
    id              BIGSERIAL PRIMARY KEY,
    node_id         VARCHAR(64)  NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    ram_free        BIGINT       NOT NULL DEFAULT 0,
    sd_status       INTEGER      NOT NULL DEFAULT 0,
    cached_models   INTEGER[]    NOT NULL DEFAULT '{}',
    last_segment_id BIGINT       NOT NULL DEFAULT 0,
    status          VARCHAR(32)  NOT NULL DEFAULT 'idle',
    ips             DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

-- raw vibration telemetry
CREATE TABLE IF NOT EXISTS raw_telemetry (
    id              BIGSERIAL,
    node_id         VARCHAR(64)  NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    segment_id      BIGINT       NOT NULL,
    chunk_id        INTEGER      NOT NULL DEFAULT 1,
    sample_rate     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    emit_reason     INTEGER      NOT NULL DEFAULT 0,
    accel_x         REAL[]       NOT NULL DEFAULT '{}',
    accel_y         REAL[]       NOT NULL DEFAULT '{}',
    accel_z         REAL[]       NOT NULL DEFAULT '{}',
    gyro_x          REAL[]       NOT NULL DEFAULT '{}',
    gyro_y          REAL[]       NOT NULL DEFAULT '{}',
    gyro_z          REAL[]       NOT NULL DEFAULT '{}',
    raw_bytes_count INTEGER      NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (node_id, timestamp, segment_id, chunk_id)
);

SELECT create_hypertable('raw_telemetry', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_raw_telemetry_node_seg ON raw_telemetry (node_id, segment_id);

-- inference results
CREATE TABLE IF NOT EXISTS inference_results (
    id                    BIGSERIAL,
    node_id               VARCHAR(64)  NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    timestamp             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    segment_id            BIGINT       NOT NULL,
    emit_reason           INTEGER      NOT NULL DEFAULT 0,
    router_model_id       BIGINT,
    autoencoder_model_id  BIGINT,
    mse                   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    anomaly               BOOLEAN      NOT NULL DEFAULT FALSE,
    is_recalculated       BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (node_id, timestamp, segment_id)
);

SELECT create_hypertable('inference_results', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_inference_node_seg ON inference_results (node_id, segment_id);

-- model binaries
CREATE TABLE IF NOT EXISTS models (
    id            BIGINT PRIMARY KEY,
    node_id       VARCHAR(64),
    tag           INTEGER,
    model_type    INTEGER NOT NULL,
    tflite_binary BYTEA   NOT NULL,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    config        JSONB   NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ensemble configurations
CREATE TABLE IF NOT EXISTS ensembles (
    id               BIGINT PRIMARY KEY,
    node_id          VARCHAR(64) REFERENCES nodes(node_id) ON DELETE SET NULL,
    router_model_id  BIGINT REFERENCES models(id) ON DELETE SET NULL,
    memory_model_id  BIGINT REFERENCES models(id) ON DELETE SET NULL,
    warmup           INTEGER NOT NULL DEFAULT 0,
    routes           JSONB   NOT NULL DEFAULT '[]',
    total_size_bytes INTEGER NOT NULL DEFAULT 0,
    deployed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

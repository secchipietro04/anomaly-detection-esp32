CREATE TABLE IF NOT EXISTS "sensor" (
  "id" INT PRIMARY KEY,
  "settings" VARCHAR(1024)
);

CREATE TABLE IF NOT EXISTS "datapoint_schema" (
  "schema_name" VARCHAR(255) PRIMARY KEY,
  "description" TEXT,
  "definition" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS "datapoint" (
  "id" INT PRIMARY KEY,
  "from_time" TIMESTAMP,
  "to_time" TIMESTAMP,
  "sensor_id" INT NOT NULL,
  "datapoint_schema" VARCHAR(255) NOT NULL,
  "data" TEXT
);

CREATE TABLE IF NOT EXISTS "ensemble" (
  "id" INT PRIMARY KEY,
  "timestamp_training_date" TIMESTAMP,
  "type" VARCHAR(100),
  "sensibility" REAL
);

CREATE TABLE IF NOT EXISTS "model" (
  "id" INT PRIMARY KEY,
  "model_data" BYTEA,
  "info" TEXT
);

CREATE TABLE IF NOT EXISTS "result" (
  "timestamp" TIMESTAMP,
  "score" NUMERIC,
  "alert" BOOLEAN,
  "sensor_id" INT NOT NULL,
  "model_id" INT NOT NULL,
  PRIMARY KEY ("sensor_id", "model_id", "timestamp")
);

ALTER TABLE "datapoint" ADD FOREIGN KEY ("sensor_id") REFERENCES "sensor" ("id") ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "datapoint" ADD FOREIGN KEY ("datapoint_schema") REFERENCES "datapoint_schema" ("schema_name") ON DELETE RESTRICT DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "model" ADD FOREIGN KEY ("id") REFERENCES "ensemble" ("id") ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "result" ADD FOREIGN KEY ("sensor_id") REFERENCES "sensor" ("id") ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "result" ADD FOREIGN KEY ("model_id") REFERENCES "ensemble" ("id") ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE;

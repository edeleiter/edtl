-- =============================================================================
-- 05_tasks.sql
-- Run as: ETL_DEPLOYER
-- Scheduled pipeline execution. Run after deploy.py creates the stored procs.
-- =============================================================================

USE ROLE ETL_DEPLOYER;
USE DATABASE UNIFIED_ETL;
USE SCHEMA PUBLIC;

-- Daily feature engineering pipeline (6 AM Mountain Time)
CREATE OR REPLACE TASK feature_pipeline_daily
    WAREHOUSE = 'COMPUTE_WH'
    SCHEDULE  = 'USING CRON 0 6 * * * America/Denver'
    COMMENT   = 'Daily feature engineering pipeline'
AS
    CALL run_feature_pipeline('raw_events', 'features_latest');

-- Batch inference preprocessing (triggered manually or by upstream task)
CREATE OR REPLACE TASK batch_inference
    WAREHOUSE = 'COMPUTE_WH'
    COMMENT   = 'Batch inference preprocessing'
AS
    CALL run_feature_pipeline('inference_input', 'inference_features');

-- Tasks are created in SUSPENDED state. To enable:
-- ALTER TASK feature_pipeline_daily RESUME;
-- ALTER TASK batch_inference RESUME;

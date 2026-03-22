-- =============================================================================
-- 04_stages.sql
-- Run as: ETL_DEPLOYER
-- Creates internal stages for reference data and Python packages.
-- =============================================================================

USE ROLE ETL_DEPLOYER;
USE DATABASE UNIFIED_ETL;
USE SCHEMA PUBLIC;
USE WAREHOUSE DEPLOY_WH;

CREATE STAGE IF NOT EXISTS reference_data_stage
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Parquet files for reference data (lookup tables, normalization params)';

CREATE STAGE IF NOT EXISTS python_packages_stage
    COMMENT = 'Python packages for UDFs and stored procedures';

SHOW STAGES;

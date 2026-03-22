-- =============================================================================
-- 02_databases_and_schemas.sql
-- Run as: SYSADMIN
-- Creates databases, schemas, and warehouses.
-- =============================================================================

-- SYSADMIN: creates databases, warehouses, and other database objects
USE ROLE SYSADMIN;

-- -----------------------------------------------------------------------------
-- 1. Production Database + Schemas
-- -----------------------------------------------------------------------------

CREATE DATABASE IF NOT EXISTS UNIFIED_ETL
    COMMENT = 'Unified ETL platform — production data';

USE DATABASE UNIFIED_ETL;

-- PUBLIC schema exists by default — used for reference/lookup tables
ALTER SCHEMA IF EXISTS PUBLIC SET COMMENT = 'Reference tables, lookup maps, normalization params';

CREATE SCHEMA IF NOT EXISTS RAW
    COMMENT = 'Raw ingested data (NFL play-by-play, etc.)';

CREATE SCHEMA IF NOT EXISTS FEATURES
    COMMENT = 'Engineered feature tables produced by training pipeline';

CREATE SCHEMA IF NOT EXISTS MODELS
    COMMENT = 'Model metadata and serialized artifacts';

CREATE SCHEMA IF NOT EXISTS MONITORING
    COMMENT = 'Prediction logs, performance snapshots, drift data';

-- -----------------------------------------------------------------------------
-- 2. Test Database (isolated, can be dropped/recreated freely)
-- -----------------------------------------------------------------------------

CREATE DATABASE IF NOT EXISTS UNIFIED_ETL_TEST
    COMMENT = 'Isolated test database for pytest and CI — safe to drop';

-- -----------------------------------------------------------------------------
-- 3. Warehouses
-- -----------------------------------------------------------------------------

-- Training warehouse — for bulk compute
CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND   = 60
    AUTO_RESUME    = TRUE
    COMMENT        = 'Training and batch compute';

-- Deploy warehouse — lightweight, for DDL and stage operations
CREATE WAREHOUSE IF NOT EXISTS DEPLOY_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND   = 60
    AUTO_RESUME    = TRUE
    COMMENT        = 'Deployment operations (stages, sprocs, tasks)';

-- Test warehouse — smallest possible, fast auto-suspend
CREATE WAREHOUSE IF NOT EXISTS TEST_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND   = 30
    AUTO_RESUME    = TRUE
    COMMENT        = 'pytest and CI test runs';

-- -----------------------------------------------------------------------------
-- 4. Verify
-- -----------------------------------------------------------------------------

SHOW SCHEMAS IN DATABASE UNIFIED_ETL;
SHOW SCHEMAS IN DATABASE UNIFIED_ETL_TEST;
SHOW WAREHOUSES LIKE '%_WH';

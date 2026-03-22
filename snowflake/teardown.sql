-- =============================================================================
-- teardown.sql
-- Run as: ACCOUNTADMIN
-- Drops everything created by the setup scripts. Use for trial resets.
-- WARNING: This is destructive and irreversible.
-- =============================================================================

USE ROLE ACCOUNTADMIN;

-- 1. Drop tasks first (they reference warehouses and procedures)
DROP TASK IF EXISTS UNIFIED_ETL.PUBLIC.feature_pipeline_daily;
DROP TASK IF EXISTS UNIFIED_ETL.PUBLIC.batch_inference;

-- 2. Drop users
DROP USER IF EXISTS ETL_DEPLOY_SVC;
DROP USER IF EXISTS ETL_TRAIN_SVC;
DROP USER IF EXISTS ETL_TEST_SVC;

-- 3. Drop databases (cascades schemas, tables, stages, procedures)
DROP DATABASE IF EXISTS UNIFIED_ETL;
DROP DATABASE IF EXISTS UNIFIED_ETL_TEST;

-- 4. Drop warehouses
DROP WAREHOUSE IF EXISTS COMPUTE_WH;
DROP WAREHOUSE IF EXISTS DEPLOY_WH;
DROP WAREHOUSE IF EXISTS TEST_WH;

-- 5. Drop roles (bottom-up to respect hierarchy)
DROP ROLE IF EXISTS ETL_TEST_RUNNER;
DROP ROLE IF EXISTS ETL_TRAINER;
DROP ROLE IF EXISTS ETL_DEPLOYER;
DROP ROLE IF EXISTS ETL_ADMIN;

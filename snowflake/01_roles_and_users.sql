-- =============================================================================
-- 01_roles_and_users.sql
-- Run as: USERADMIN (roles + users), briefly SECURITYADMIN (role hierarchy)
-- Creates the role hierarchy and service users for the unified-etl platform.
-- =============================================================================

-- USERADMIN: creates and manages roles and users
USE ROLE USERADMIN;

-- -----------------------------------------------------------------------------
-- 1. Custom Roles
-- -----------------------------------------------------------------------------

CREATE ROLE IF NOT EXISTS ETL_ADMIN
    COMMENT = 'Platform admin — creates databases, warehouses, manages grants';

CREATE ROLE IF NOT EXISTS ETL_DEPLOYER
    COMMENT = 'Deploys stages, stored procedures, and tasks';

CREATE ROLE IF NOT EXISTS ETL_TRAINER
    COMMENT = 'Runs training pipelines — reads raw data, writes features/models';

CREATE ROLE IF NOT EXISTS ETL_TEST_RUNNER
    COMMENT = 'CI/CD and local test runner — full access to test database only';

-- -----------------------------------------------------------------------------
-- 2. Role Hierarchy (all roll up to ETL_ADMIN)
-- -----------------------------------------------------------------------------

-- USERADMIN can grant roles it owns to other roles
GRANT ROLE ETL_DEPLOYER    TO ROLE ETL_ADMIN;
GRANT ROLE ETL_TRAINER     TO ROLE ETL_ADMIN;
GRANT ROLE ETL_TEST_RUNNER TO ROLE ETL_ADMIN;

-- SECURITYADMIN needed to wire custom roles into the system hierarchy
USE ROLE SECURITYADMIN;
GRANT ROLE ETL_ADMIN TO ROLE SYSADMIN;

-- Back to USERADMIN for user management
USE ROLE USERADMIN;

-- -----------------------------------------------------------------------------
-- 3. Grant ETL_ADMIN to your human user
-- -----------------------------------------------------------------------------

SET my_username = 'edeleiter';
GRANT ROLE ETL_ADMIN TO USER IDENTIFIER($my_username);

-- -----------------------------------------------------------------------------
-- 4. Service Users
--
--    >>> SET PASSWORDS BEFORE RUNNING <<<
--    Generate strong passwords and set them below. Do NOT commit passwords
--    to version control. Use: openssl rand -base64 24
-- -----------------------------------------------------------------------------

SET deploy_pw  = '';  -- <<< SET THIS (e.g., openssl rand -base64 24)
SET train_pw   = '';  -- <<< SET THIS
SET test_pw    = '';  -- <<< SET THIS

-- Deploy service user
CREATE USER IF NOT EXISTS ETL_DEPLOY_SVC
    PASSWORD           = $deploy_pw
    DEFAULT_ROLE       = ETL_DEPLOYER
    DEFAULT_WAREHOUSE  = DEPLOY_WH
    DEFAULT_NAMESPACE  = UNIFIED_ETL.PUBLIC
    MUST_CHANGE_PASSWORD = FALSE
    COMMENT            = 'Service user for deploy.py and CI/CD deploys';

GRANT ROLE ETL_DEPLOYER TO USER ETL_DEPLOY_SVC;

-- Training service user
CREATE USER IF NOT EXISTS ETL_TRAIN_SVC
    PASSWORD           = $train_pw
    DEFAULT_ROLE       = ETL_TRAINER
    DEFAULT_WAREHOUSE  = COMPUTE_WH
    DEFAULT_NAMESPACE  = UNIFIED_ETL.RAW
    MUST_CHANGE_PASSWORD = FALSE
    COMMENT            = 'Service user for training pipeline';

GRANT ROLE ETL_TRAINER TO USER ETL_TRAIN_SVC;

-- Test runner service user
CREATE USER IF NOT EXISTS ETL_TEST_SVC
    PASSWORD           = $test_pw
    DEFAULT_ROLE       = ETL_TEST_RUNNER
    DEFAULT_WAREHOUSE  = TEST_WH
    DEFAULT_NAMESPACE  = UNIFIED_ETL_TEST.PUBLIC
    MUST_CHANGE_PASSWORD = FALSE
    COMMENT            = 'Service user for pytest and CI test runs';

GRANT ROLE ETL_TEST_RUNNER TO USER ETL_TEST_SVC;

-- -----------------------------------------------------------------------------
-- 5. Verify
-- -----------------------------------------------------------------------------

SHOW ROLES LIKE 'ETL_%';
SHOW USERS LIKE 'ETL_%';

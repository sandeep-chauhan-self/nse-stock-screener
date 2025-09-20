@echo off
REM NSE Stock Screener - Windows Production Deployment Script
REM Secure deployment with validation and rollback capabilities

setlocal EnableDelayedExpansion

REM Script configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\"
set "DEPLOYMENT_LOG=C:\logs\nse-screener\deployment.log"

REM Default configuration
if "%ENVIRONMENT%"=="" set "ENVIRONMENT=production"
if "%REGISTRY%"=="" set "REGISTRY=ghcr.io"
if "%IMAGE_NAME%"=="" set "IMAGE_NAME=nse-stock-screener"
if "%IMAGE_TAG%"=="" set "IMAGE_TAG=latest"
if "%NAMESPACE%"=="" set "NAMESPACE=nse-screener"
if "%BACKUP_ENABLED%"=="" set "BACKUP_ENABLED=true"
if "%HEALTH_CHECK_TIMEOUT%"=="" set "HEALTH_CHECK_TIMEOUT=300"
if "%ROLLBACK_ON_FAILURE%"=="" set "ROLLBACK_ON_FAILURE=true"

REM Create log directory
if not exist "C:\logs\nse-screener\" mkdir "C:\logs\nse-screener\"

REM Logging functions
:log_info
echo [%date% %time%] [INFO] %~1
echo [%date% %time%] [INFO] %~1 >> "%DEPLOYMENT_LOG%"
goto :eof

:log_warn
echo [%date% %time%] [WARN] %~1
echo [%date% %time%] [WARN] %~1 >> "%DEPLOYMENT_LOG%"
goto :eof

:log_error
echo [%date% %time%] [ERROR] %~1
echo [%date% %time%] [ERROR] %~1 >> "%DEPLOYMENT_LOG%"
goto :eof

:log_success
echo [%date% %time%] [SUCCESS] %~1
echo [%date% %time%] [SUCCESS] %~1 >> "%DEPLOYMENT_LOG%"
goto :eof

REM Validation functions
:validate_environment
call :log_info "Validating deployment environment..."

REM Check required tools
call :check_tool "docker" || goto :error
call :check_tool "docker-compose" || goto :error
call :check_tool "curl" || goto :error

call :log_success "All required tools are available"

REM Check environment variables
set "missing_vars="
call :check_env_var "NSE_SCREENER_DB_PASSWORD" || set "missing_vars=!missing_vars! NSE_SCREENER_DB_PASSWORD"
call :check_env_var "NSE_SCREENER_API_KEY_NSE" || set "missing_vars=!missing_vars! NSE_SCREENER_API_KEY_NSE"
call :check_env_var "NSE_SCREENER_REDIS_PASSWORD" || set "missing_vars=!missing_vars! NSE_SCREENER_REDIS_PASSWORD"

if not "!missing_vars!"=="" (
    call :log_error "Missing required environment variables:!missing_vars!"
    goto :error
)
call :log_success "All required environment variables are set"

REM Validate Docker daemon
docker info >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker daemon is not running or accessible"
    goto :error
)
call :log_success "Docker daemon is accessible"
goto :eof

:check_tool
where %1 >nul 2>&1
if errorlevel 1 (
    call :log_error "Required tool not found: %1"
    exit /b 1
)
exit /b 0

:check_env_var
if "%!%1%!"=="" exit /b 1
exit /b 0

:validate_configuration
call :log_info "Validating deployment configuration..."

REM Validate Docker Compose file
cd /d "%PROJECT_ROOT%"
docker-compose -f docker-compose.yml config --quiet
if errorlevel 1 (
    call :log_error "Docker Compose configuration is invalid"
    goto :error
)
call :log_success "Docker Compose configuration is valid"

REM Validate secrets
if exist "%PROJECT_ROOT%src\security\validate_secrets.py" (
    python "%PROJECT_ROOT%src\security\validate_secrets.py" --validate-only
    if errorlevel 1 (
        call :log_error "Secrets validation failed"
        goto :error
    )
    call :log_success "Secrets validation passed"
) else (
    call :log_warn "Secrets validator not found, skipping validation"
)
goto :eof

:security_scan
call :log_info "Running pre-deployment security scan..."

set "image_full_name=%REGISTRY%/%IMAGE_NAME%:%IMAGE_TAG%"

REM Pull the latest image
call :log_info "Pulling container image: !image_full_name!"
docker pull "!image_full_name!"
if errorlevel 1 (
    call :log_error "Failed to pull container image"
    goto :error
)

REM Run basic security scan if trivy is available
where trivy >nul 2>&1
if not errorlevel 1 (
    call :log_info "Running container security scan..."
    trivy image --severity HIGH,CRITICAL "!image_full_name!"
    if errorlevel 1 (
        call :log_warn "Security vulnerabilities found in container image"
        set /p "continue=Continue deployment despite vulnerabilities? (y/N): "
        if /i not "!continue!"=="y" (
            call :log_error "Deployment aborted due to security concerns"
            goto :error
        )
    )
) else (
    call :log_warn "Trivy not available, skipping container security scan"
)

call :log_success "Security scan completed"
goto :eof

:backup_current_deployment
if "%BACKUP_ENABLED%" neq "true" (
    call :log_info "Backup disabled, skipping..."
    goto :eof
)

call :log_info "Creating backup of current deployment..."

set "backup_dir=C:\backups\nse-screener\%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "backup_dir=!backup_dir: =0!"
mkdir "!backup_dir!" 2>nul

REM Backup configuration files
if exist "%PROJECT_ROOT%docker-compose.yml" (
    copy "%PROJECT_ROOT%docker-compose.yml" "!backup_dir!\" >nul
    call :log_info "Backed up Docker Compose configuration"
)

if exist "%PROJECT_ROOT%.env" (
    copy "%PROJECT_ROOT%.env" "!backup_dir!\" >nul
    call :log_info "Backed up environment configuration"
)

REM Backup database if running
docker-compose ps postgres | findstr "Up" >nul
if not errorlevel 1 (
    call :log_info "Backing up database..."
    docker-compose exec -T postgres pg_dump -U "%NSE_SCREENER_DB_USERNAME%" "%NSE_SCREENER_DB_NAME%" > "!backup_dir!\database_backup.sql"
    call :log_success "Database backup completed"
)

REM Store backup path for potential rollback
echo !backup_dir! > "%TEMP%\nse_screener_last_backup.txt"
call :log_success "Backup completed: !backup_dir!"
goto :eof

:deploy_services
call :log_info "Starting deployment of NSE Stock Screener services..."

cd /d "%PROJECT_ROOT%"

REM Set build arguments
for /f %%i in ('powershell -command "Get-Date -UFormat '%%Y-%%m-%%dT%%H:%%M:%%SZ'"') do set "BUILD_DATE=%%i"
for /f %%i in ('git rev-parse HEAD 2^>nul') do set "VCS_REF=%%i"
if "%VCS_REF%"=="" set "VCS_REF=unknown"

REM Deploy infrastructure services first
call :log_info "Deploying infrastructure services..."
docker-compose up -d postgres redis

REM Wait for infrastructure to be ready
call :log_info "Waiting for infrastructure services to be ready..."
set /a "max_attempts=30"
set /a "attempt=0"

:wait_db_loop
if !attempt! geq !max_attempts! (
    call :log_error "Database failed to become ready within timeout"
    goto :error
)

docker-compose exec postgres pg_isready -U "%NSE_SCREENER_DB_USERNAME%" >nul 2>&1
if not errorlevel 1 (
    call :log_success "Database is ready"
    goto :db_ready
)

set /a "attempt+=1"
call :log_info "Waiting for database... (attempt !attempt!/!max_attempts!)"
timeout /t 10 /nobreak >nul
goto :wait_db_loop

:db_ready
REM Deploy main application
call :log_info "Deploying main application..."
docker-compose up -d nse-screener

call :log_success "All services deployed successfully"
goto :eof

:health_check
call :log_info "Running health checks..."

set /a "start_time=%time:~0,2%*3600 + %time:~3,2%*60 + %time:~6,2%"
set /a "timeout=%HEALTH_CHECK_TIMEOUT%"

:health_check_loop
set /a "current_time=%time:~0,2%*3600 + %time:~3,2%*60 + %time:~6,2%"
set /a "elapsed=current_time - start_time"

if !elapsed! gtr !timeout! (
    call :log_error "Health check timeout after !timeout!s"
    exit /b 1
)

REM Check main application health
docker-compose exec -T nse-screener python healthcheck.py --json >nul 2>&1
if not errorlevel 1 (
    call :log_success "Application health check passed"
    goto :extended_health_checks
)

call :log_info "Health check in progress... (!elapsed!s/!timeout!s)"
timeout /t 10 /nobreak >nul
goto :health_check_loop

:extended_health_checks
call :log_info "Running extended health checks..."

REM Check database connectivity
docker-compose exec -T postgres pg_isready -U "%NSE_SCREENER_DB_USERNAME%" >nul 2>&1
if errorlevel 1 (
    call :log_error "Database connectivity check failed"
    exit /b 1
)
call :log_success "Database connectivity check passed"

REM Check Redis connectivity
docker-compose exec -T redis redis-cli --no-auth-warning -a "%NSE_SCREENER_REDIS_PASSWORD%" ping | findstr "PONG" >nul
if errorlevel 1 (
    call :log_error "Redis connectivity check failed"
    exit /b 1
)
call :log_success "Redis connectivity check passed"

call :log_success "All health checks passed"
exit /b 0

:rollback_deployment
call :log_warn "Initiating deployment rollback..."

if not exist "%TEMP%\nse_screener_last_backup.txt" (
    call :log_error "No backup found for rollback"
    exit /b 1
)

set /p "backup_dir=" < "%TEMP%\nse_screener_last_backup.txt"
if not exist "!backup_dir!" (
    call :log_error "Backup directory not found: !backup_dir!"
    exit /b 1
)

REM Stop current services
call :log_info "Stopping current services..."
docker-compose down

REM Restore configuration
if exist "!backup_dir!\docker-compose.yml" (
    copy "!backup_dir!\docker-compose.yml" "%PROJECT_ROOT%" >nul
    call :log_info "Restored Docker Compose configuration"
)

if exist "!backup_dir!\.env" (
    copy "!backup_dir!\.env" "%PROJECT_ROOT%" >nul
    call :log_info "Restored environment configuration"
)

REM Restore database if backup exists
if exist "!backup_dir!\database_backup.sql" (
    call :log_info "Restoring database..."
    docker-compose up -d postgres
    timeout /t 30 /nobreak >nul
    docker-compose exec -T postgres psql -U "%NSE_SCREENER_DB_USERNAME%" -d "%NSE_SCREENER_DB_NAME%" < "!backup_dir!\database_backup.sql"
    call :log_success "Database restored"
)

REM Start services
call :log_info "Starting rolled-back services..."
docker-compose up -d

call :log_success "Rollback completed successfully"
goto :eof

:post_deployment_tasks
call :log_info "Running post-deployment tasks..."

REM Clean up old Docker images and containers
call :log_info "Cleaning up old Docker images..."
docker image prune -f >nul

docker container prune -f >nul

REM Generate deployment report
set "report_file=C:\logs\nse-screener\deployment-report-%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%.json"
set "report_file=!report_file: =0!"

(
echo {
echo   "deployment_id": "%RANDOM%",
echo   "timestamp": "%BUILD_DATE%",
echo   "environment": "%ENVIRONMENT%",
echo   "image_tag": "%IMAGE_TAG%",
echo   "git_commit": "%VCS_REF%",
echo   "status": "success"
echo }
) > "!report_file!"

call :log_success "Deployment report generated: !report_file!"
call :log_success "Post-deployment tasks completed"
goto :eof

:main
call :log_info "Starting NSE Stock Screener deployment"
call :log_info "Environment: %ENVIRONMENT%"
call :log_info "Image: %REGISTRY%/%IMAGE_NAME%:%IMAGE_TAG%"

REM Validation phase
call :validate_environment || goto :error
call :validate_configuration || goto :error

REM Security phase
call :security_scan || goto :error

REM Backup phase
call :backup_current_deployment || goto :error

REM Deployment phase
call :deploy_services || goto :error

REM Verification phase
call :health_check
if errorlevel 1 (
    call :log_error "Health checks failed"
    goto :error
)

REM Post-deployment phase
call :post_deployment_tasks || goto :error

call :log_success "NSE Stock Screener deployment completed successfully!"
call :log_info "Access the application at: http://localhost:8000"
call :log_info "Monitoring dashboard: http://localhost:3000 (if enabled)"
goto :end

:error
call :log_error "Deployment failed"
if "%ROLLBACK_ON_FAILURE%"=="true" (
    call :log_warn "Initiating automatic rollback..."
    call :rollback_deployment
)
exit /b 1

REM Command line interface
:start
if "%~1"=="" goto :main
if /i "%~1"=="deploy" goto :main
if /i "%~1"=="rollback" (
    call :rollback_deployment
    goto :end
)
if /i "%~1"=="health-check" (
    call :health_check
    goto :end
)
if /i "%~1"=="backup" (
    call :backup_current_deployment
    goto :end
)
if /i "%~1"=="validate" (
    call :validate_environment
    call :validate_configuration
    goto :end
)

echo Usage: %0 {deploy^|rollback^|health-check^|backup^|validate}
echo.
echo Commands:
echo   deploy       - Full deployment process (default)
echo   rollback     - Rollback to previous deployment
echo   health-check - Run health checks on current deployment
echo   backup       - Create backup of current deployment
echo   validate     - Validate environment and configuration
exit /b 1

:end
endlocal
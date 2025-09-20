#!/bin/bash
# NSE Stock Screener - Production Deployment Script
# Secure deployment with validation and rollback capabilities

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_LOG="/var/log/nse-screener/deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
ENVIRONMENT="${ENVIRONMENT:-production}"
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-nse-stock-screener}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
NAMESPACE="${NAMESPACE:-nse-screener}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Logging functions
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$DEPLOYMENT_LOG"
}

log_info() {
    log "INFO" "${BLUE}[INFO]${NC} $*"
}

log_warn() {
    log "WARN" "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    log "ERROR" "${RED}[ERROR]${NC} $*"
}

log_success() {
    log "SUCCESS" "${GREEN}[SUCCESS]${NC} $*"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code: $exit_code"
        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            log_warn "Initiating automatic rollback..."
            rollback_deployment
        fi
    fi
    exit $exit_code
}

trap cleanup EXIT

# Validation functions
validate_environment() {
    log_info "Validating deployment environment..."
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done
    log_success "All required tools are available"
    
    # Check environment variables
    local required_vars=(
        "NSE_SCREENER_DB_PASSWORD"
        "NSE_SCREENER_API_KEY_NSE"
        "NSE_SCREENER_REDIS_PASSWORD"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        exit 1
    fi
    log_success "All required environment variables are set"
    
    # Validate Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or accessible"
        exit 1
    fi
    log_success "Docker daemon is accessible"
}

validate_configuration() {
    log_info "Validating deployment configuration..."
    
    # Validate Docker Compose file
    if ! docker-compose -f docker-compose.yml config --quiet; then
        log_error "Docker Compose configuration is invalid"
        exit 1
    fi
    log_success "Docker Compose configuration is valid"
    
    # Validate secrets
    if [ -f "$PROJECT_ROOT/src/security/validate_secrets.py" ]; then
        if ! python "$PROJECT_ROOT/src/security/validate_secrets.py" --validate-only; then
            log_error "Secrets validation failed"
            exit 1
        fi
        log_success "Secrets validation passed"
    else
        log_warn "Secrets validator not found, skipping validation"
    fi
}

# Security functions
security_scan() {
    log_info "Running pre-deployment security scan..."
    
    local image_full_name="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    # Pull the latest image
    log_info "Pulling container image: $image_full_name"
    if ! docker pull "$image_full_name"; then
        log_error "Failed to pull container image"
        exit 1
    fi
    
    # Run basic security scan
    log_info "Running container security scan..."
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL "$image_full_name" || {
            log_warn "Security vulnerabilities found in container image"
            read -p "Continue deployment despite vulnerabilities? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_error "Deployment aborted due to security concerns"
                exit 1
            fi
        }
    else
        log_warn "Trivy not available, skipping container security scan"
    fi
    
    log_success "Security scan completed"
}

# Backup functions
backup_current_deployment() {
    if [ "$BACKUP_ENABLED" != "true" ]; then
        log_info "Backup disabled, skipping..."
        return 0
    fi
    
    log_info "Creating backup of current deployment..."
    
    local backup_dir="/var/backups/nse-screener/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup configuration files
    if [ -f docker-compose.yml ]; then
        cp docker-compose.yml "$backup_dir/"
        log_info "Backed up Docker Compose configuration"
    fi
    
    # Backup environment file if it exists
    if [ -f .env ]; then
        cp .env "$backup_dir/"
        log_info "Backed up environment configuration"
    fi
    
    # Backup database if running
    if docker-compose ps postgres | grep -q "Up"; then
        log_info "Backing up database..."
        docker-compose exec -T postgres pg_dump -U "${NSE_SCREENER_DB_USERNAME}" "${NSE_SCREENER_DB_NAME}" > "$backup_dir/database_backup.sql"
        log_success "Database backup completed"
    fi
    
    # Store backup path for potential rollback
    echo "$backup_dir" > /tmp/nse_screener_last_backup
    log_success "Backup completed: $backup_dir"
}

# Deployment functions
deploy_services() {
    log_info "Starting deployment of NSE Stock Screener services..."
    
    # Set build arguments
    export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    export VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    
    # Deploy infrastructure services first (database, cache)
    log_info "Deploying infrastructure services..."
    docker-compose up -d postgres redis
    
    # Wait for infrastructure to be ready
    log_info "Waiting for infrastructure services to be ready..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose exec postgres pg_isready -U "${NSE_SCREENER_DB_USERNAME}" &> /dev/null; then
            log_success "Database is ready"
            break
        fi
        attempt=$((attempt + 1))
        log_info "Waiting for database... (attempt $attempt/$max_attempts)"
        sleep 10
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "Database failed to become ready within timeout"
        exit 1
    fi
    
    # Deploy main application
    log_info "Deploying main application..."
    docker-compose up -d nse-screener
    
    log_success "All services deployed successfully"
}

# Health check functions
health_check() {
    log_info "Running health checks..."
    
    local start_time=$(date +%s)
    local timeout=$HEALTH_CHECK_TIMEOUT
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout ]; then
            log_error "Health check timeout after ${timeout}s"
            return 1
        fi
        
        # Check main application health
        if docker-compose exec -T nse-screener python healthcheck.py --json &> /dev/null; then
            log_success "Application health check passed"
            break
        fi
        
        log_info "Health check in progress... (${elapsed}s/${timeout}s)"
        sleep 10
    done
    
    # Additional health checks
    log_info "Running extended health checks..."
    
    # Check database connectivity
    if docker-compose exec -T postgres pg_isready -U "${NSE_SCREENER_DB_USERNAME}" &> /dev/null; then
        log_success "Database connectivity check passed"
    else
        log_error "Database connectivity check failed"
        return 1
    fi
    
    # Check Redis connectivity
    if docker-compose exec -T redis redis-cli --no-auth-warning -a "${NSE_SCREENER_REDIS_PASSWORD}" ping | grep -q PONG; then
        log_success "Redis connectivity check passed"
    else
        log_error "Redis connectivity check failed"
        return 1
    fi
    
    log_success "All health checks passed"
    return 0
}

# Rollback functions
rollback_deployment() {
    log_warn "Initiating deployment rollback..."
    
    if [ ! -f /tmp/nse_screener_last_backup ]; then
        log_error "No backup found for rollback"
        return 1
    fi
    
    local backup_dir=$(cat /tmp/nse_screener_last_backup)
    if [ ! -d "$backup_dir" ]; then
        log_error "Backup directory not found: $backup_dir"
        return 1
    fi
    
    # Stop current services
    log_info "Stopping current services..."
    docker-compose down
    
    # Restore configuration
    if [ -f "$backup_dir/docker-compose.yml" ]; then
        cp "$backup_dir/docker-compose.yml" .
        log_info "Restored Docker Compose configuration"
    fi
    
    if [ -f "$backup_dir/.env" ]; then
        cp "$backup_dir/.env" .
        log_info "Restored environment configuration"
    fi
    
    # Restore database if backup exists
    if [ -f "$backup_dir/database_backup.sql" ]; then
        log_info "Restoring database..."
        docker-compose up -d postgres
        sleep 30  # Wait for database to start
        docker-compose exec -T postgres psql -U "${NSE_SCREENER_DB_USERNAME}" -d "${NSE_SCREENER_DB_NAME}" < "$backup_dir/database_backup.sql"
        log_success "Database restored"
    fi
    
    # Start services
    log_info "Starting rolled-back services..."
    docker-compose up -d
    
    log_success "Rollback completed successfully"
}

# Monitoring and cleanup
post_deployment_tasks() {
    log_info "Running post-deployment tasks..."
    
    # Clean up old Docker images
    log_info "Cleaning up old Docker images..."
    docker image prune -f
    
    # Clean up old containers
    docker container prune -f
    
    # Update monitoring configuration if needed
    if [ -f "$PROJECT_ROOT/docker/prometheus.yml" ]; then
        log_info "Updating monitoring configuration..."
        # Add monitoring setup logic here
    fi
    
    # Generate deployment report
    local report_file="/var/log/nse-screener/deployment-report-$(date +%Y%m%d_%H%M%S).json"
    cat > "$report_file" << EOF
{
  "deployment_id": "$(date +%s)",
  "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
  "environment": "$ENVIRONMENT",
  "image_tag": "$IMAGE_TAG",
  "git_commit": "${VCS_REF:-unknown}",
  "services": $(docker-compose ps --format json),
  "status": "success"
}
EOF
    
    log_success "Deployment report generated: $report_file"
    log_success "Post-deployment tasks completed"
}

# Main deployment function
main() {
    log_info "Starting NSE Stock Screener deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    # Create log directory
    mkdir -p "$(dirname "$DEPLOYMENT_LOG")"
    
    # Validation phase
    validate_environment
    validate_configuration
    
    # Security phase
    security_scan
    
    # Backup phase
    backup_current_deployment
    
    # Deployment phase
    deploy_services
    
    # Verification phase
    if ! health_check; then
        log_error "Health checks failed"
        exit 1
    fi
    
    # Post-deployment phase
    post_deployment_tasks
    
    log_success "NSE Stock Screener deployment completed successfully!"
    log_info "Access the application at: http://localhost:8000"
    log_info "Monitoring dashboard: http://localhost:3000 (if enabled)"
}

# Command line interface
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "health-check")
        health_check
        ;;
    "backup")
        backup_current_deployment
        ;;
    "validate")
        validate_environment
        validate_configuration
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health-check|backup|validate}"
        echo ""
        echo "Commands:"
        echo "  deploy       - Full deployment process (default)"
        echo "  rollback     - Rollback to previous deployment"
        echo "  health-check - Run health checks on current deployment"
        echo "  backup       - Create backup of current deployment"
        echo "  validate     - Validate environment and configuration"
        exit 1
        ;;
esac
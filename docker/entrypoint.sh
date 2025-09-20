#!/bin/bash
# NSE Stock Screener - Docker Entrypoint Script
# Handles initialization, configuration validation, and graceful startup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Signal handlers for graceful shutdown
cleanup() {
    log_info "Received termination signal, shutting down gracefully..."
    # Kill any background processes
    jobs -p | xargs -r kill
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main initialization function
initialize_application() {
    log_info "Starting NSE Stock Screener initialization..."
    
    # Check if running as non-root user
    if [ "$EUID" -eq 0 ]; then
        log_error "Container should not run as root user for security reasons"
        exit 1
    fi
    
    # Validate required directories exist
    log_info "Checking directory structure..."
    for dir in "/app/data" "/app/output" "/app/logs" "/app/temp"; do
        if [ ! -d "$dir" ]; then
            log_warn "Creating missing directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Validate Python environment
    log_info "Checking Python environment..."
    python --version || {
        log_error "Python not found or not working"
        exit 1
    }
    
    # Validate secrets (basic check)
    log_info "Validating configuration..."
    if [ -f "/app/src/security/validate_secrets.py" ]; then
        python /app/src/security/validate_secrets.py --validate-only || {
            log_error "Secrets validation failed"
            log_info "Use --template to see required environment variables"
            exit 1
        }
        log_success "Configuration validation passed"
    else
        log_warn "Secrets validator not found, skipping validation"
    fi
    
    # Check dependencies
    log_info "Checking Python dependencies..."
    if [ -f "/app/scripts/check_deps.py" ]; then
        python /app/scripts/check_deps.py || {
            log_error "Dependency check failed"
            exit 1
        }
        log_success "Dependencies check passed"
    else
        log_warn "Dependency checker not found, skipping check"
    fi
    
    log_success "Initialization completed successfully"
}

# Health check function
health_check() {
    log_info "Running health check..."
    
    # Check if main modules can be imported
    python -c "
import sys
sys.path.append('/app/src')
try:
    from enhanced_launcher import main
    from security.secrets_manager import get_secrets_manager
    print('✓ Core modules import successfully')
except ImportError as e:
    print(f'✗ Module import failed: {e}')
    sys.exit(1)
" || {
        log_error "Health check failed"
        return 1
    }
    
    log_success "Health check passed"
    return 0
}

# Main execution logic
main() {
    log_info "NSE Stock Screener Docker Container Starting..."
    log_info "Container user: $(whoami)"
    log_info "Working directory: $(pwd)"
    log_info "Python path: $PYTHONPATH"
    
    # Initialize application
    initialize_application
    
    # Handle different startup modes
    case "${1:-default}" in
        "health-check")
            health_check
            exit $?
            ;;
        "validate-only")
            log_info "Running validation only mode"
            python /app/src/security/validate_secrets.py --validate-only
            exit $?
            ;;
        "interactive")
            log_info "Starting interactive shell"
            exec /bin/bash
            ;;
        "help"|"--help")
            log_info "Showing help for enhanced launcher"
            exec python -m src.enhanced_launcher --help
            ;;
        "default"|"")
            log_info "Starting default application mode"
            exec python -m src.enhanced_launcher
            ;;
        *)
            log_info "Starting with custom command: $*"
            exec "$@"
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
#!/usr/bin/env python3
"""
NSE Stock Screener - Docker Health Check Script
Performs comprehensive health checks for containerized deployment.
"""

import sys
import os
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Quiet by default for health checks
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, '/app/src')


class HealthChecker:
    """Comprehensive health check for NSE Stock Screener."""
    
    def __init__(self):
        self.checks = []
        self.start_time = time.time()
    
    def add_check(self, name: str, status: bool, message: str = "", details: Dict = None):
        """Add a health check result."""
        self.checks.append({
            "name": name,
            "status": "pass" if status else "fail",
            "message": message,
            "details": details or {}
        })
    
    def check_python_environment(self) -> bool:
        """Check Python environment and basic imports."""
        try:
            # Check Python version
            version_info = sys.version_info
            if version_info.major != 3 or version_info.minor < 11:
                self.add_check(
                    "python_version", 
                    False, 
                    f"Python {version_info.major}.{version_info.minor} not supported, need 3.11+"
                )
                return False
            
            self.add_check(
                "python_version", 
                True, 
                f"Python {version_info.major}.{version_info.minor}.{version_info.micro}"
            )
            
            # Check critical imports
            critical_modules = [
                "pandas", "numpy", "requests", "yfinance", 
                "prometheus_client", "joblib", "redis"
            ]
            
            missing_modules = []
            for module in critical_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)
            
            if missing_modules:
                self.add_check(
                    "critical_imports", 
                    False, 
                    f"Missing modules: {', '.join(missing_modules)}"
                )
                return False
            else:
                self.add_check(
                    "critical_imports", 
                    True, 
                    f"All {len(critical_modules)} critical modules available"
                )
            
            return True
            
        except Exception as e:
            self.add_check("python_environment", False, f"Error: {e}")
            return False
    
    def check_application_modules(self) -> bool:
        """Check if application modules can be imported."""
        try:
            # Test core application imports
            app_modules = [
                "enhanced_launcher",
                "advanced_backtester", 
                "enhanced_early_warning_system",
                "composite_scorer"
            ]
            
            import_errors = []
            for module in app_modules:
                try:
                    __import__(module)
                except ImportError as e:
                    import_errors.append(f"{module}: {e}")
            
            if import_errors:
                self.add_check(
                    "application_modules", 
                    False, 
                    f"Import errors: {'; '.join(import_errors)}"
                )
                return False
            else:
                self.add_check(
                    "application_modules", 
                    True, 
                    f"All {len(app_modules)} application modules importable"
                )
                return True
                
        except Exception as e:
            self.add_check("application_modules", False, f"Error: {e}")
            return False
    
    def check_secrets_management(self) -> bool:
        """Check secrets management system."""
        try:
            from security.secrets_manager import SecretsManager, SecretValidationError
            
            # Try to initialize secrets manager
            try:
                secrets = SecretsManager()
                health_status = secrets.health_check()
                
                if health_status["status"] == "healthy":
                    self.add_check(
                        "secrets_management", 
                        True, 
                        f"Secrets loaded: {health_status['secrets_loaded']}"
                    )
                    return True
                else:
                    self.add_check(
                        "secrets_management", 
                        False, 
                        f"Secrets unhealthy: {health_status.get('error', 'Unknown')}"
                    )
                    return False
                    
            except SecretValidationError as e:
                self.add_check(
                    "secrets_management", 
                    False, 
                    f"Validation failed: {e}"
                )
                return False
                
        except ImportError:
            self.add_check(
                "secrets_management", 
                False, 
                "Secrets manager module not available"
            )
            return False
        except Exception as e:
            self.add_check("secrets_management", False, f"Error: {e}")
            return False
    
    def check_file_system(self) -> bool:
        """Check required directories and permissions."""
        try:
            required_dirs = [
                "/app/data",
                "/app/output", 
                "/app/logs",
                "/app/temp"
            ]
            
            dir_issues = []
            for dir_path in required_dirs:
                path = Path(dir_path)
                if not path.exists():
                    dir_issues.append(f"{dir_path}: does not exist")
                elif not path.is_dir():
                    dir_issues.append(f"{dir_path}: not a directory")
                elif not os.access(dir_path, os.W_OK):
                    dir_issues.append(f"{dir_path}: not writable")
            
            if dir_issues:
                self.add_check(
                    "file_system", 
                    False, 
                    f"Directory issues: {'; '.join(dir_issues)}"
                )
                return False
            else:
                self.add_check(
                    "file_system", 
                    True, 
                    f"All {len(required_dirs)} required directories accessible"
                )
                return True
                
        except Exception as e:
            self.add_check("file_system", False, f"Error: {e}")
            return False
    
    def check_security_posture(self) -> bool:
        """Check security configuration."""
        try:
            security_issues = []
            
            # Check if running as root (security risk)
            if os.getuid() == 0:
                security_issues.append("Running as root user")
            
            # Check environment variables for security
            sensitive_vars = [
                "NSE_SCREENER_DB_PASSWORD",
                "NSE_SCREENER_API_KEY_NSE",
                "NSE_SCREENER_REDIS_PASSWORD"
            ]
            
            missing_vars = []
            for var in sensitive_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                # Don't fail health check for missing vars in development
                logger.warning(f"Missing environment variables: {missing_vars}")
            
            if security_issues:
                self.add_check(
                    "security_posture", 
                    False, 
                    f"Security issues: {'; '.join(security_issues)}"
                )
                return False
            else:
                self.add_check(
                    "security_posture", 
                    True, 
                    "Security posture acceptable"
                )
                return True
                
        except Exception as e:
            self.add_check("security_posture", False, f"Error: {e}")
            return False
    
    def run_health_check(self) -> bool:
        """Run all health checks and return overall status."""
        logger.info("Starting comprehensive health check...")
        
        # Run all checks
        checks_passed = 0
        total_checks = 0
        
        check_methods = [
            self.check_python_environment,
            self.check_application_modules,
            self.check_file_system,
            self.check_security_posture,
            self.check_secrets_management  # This may fail in dev
        ]
        
        for check_method in check_methods:
            try:
                result = check_method()
                total_checks += 1
                if result:
                    checks_passed += 1
            except Exception as e:
                logger.error(f"Health check {check_method.__name__} failed: {e}")
                total_checks += 1
        
        # Calculate health score
        health_score = (checks_passed / total_checks) * 100 if total_checks > 0 else 0
        elapsed_time = time.time() - self.start_time
        
        # Overall status
        overall_healthy = health_score >= 80  # 80% threshold for passing
        
        self.add_check(
            "overall_health", 
            overall_healthy, 
            f"Health score: {health_score:.1f}% ({checks_passed}/{total_checks} checks passed)",
            {
                "health_score": health_score,
                "checks_passed": checks_passed,
                "total_checks": total_checks,
                "elapsed_time": elapsed_time
            }
        )
        
        return overall_healthy
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "timestamp": time.time(),
            "overall_status": "healthy" if any(
                check["name"] == "overall_health" and check["status"] == "pass" 
                for check in self.checks
            ) else "unhealthy",
            "checks": self.checks,
            "environment": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "working_directory": os.getcwd(),
                "user_id": os.getuid(),
                "environment_vars": len([k for k in os.environ.keys() if k.startswith("NSE_SCREENER")])
            }
        }


def main():
    """Main health check function."""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="NSE Stock Screener Health Check")
        parser.add_argument("--json", action="store_true", help="Output JSON format")
        parser.add_argument("--verbose", action="store_true", help="Verbose output")
        args = parser.parse_args()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.INFO)
        
        # Run health check
        checker = HealthChecker()
        overall_healthy = checker.run_health_check()
        
        # Output results
        if args.json:
            print(json.dumps(checker.get_health_report(), indent=2))
        else:
            # Human-readable output
            report = checker.get_health_report()
            print(f"NSE Stock Screener Health Check")
            print(f"Status: {report['overall_status'].upper()}")
            print(f"Python: {report['environment']['python_version']}")
            print(f"User ID: {report['environment']['user_id']}")
            print()
            
            for check in checker.checks:
                status_symbol = "✓" if check["status"] == "pass" else "✗"
                print(f"{status_symbol} {check['name']}: {check['message']}")
        
        # Exit with appropriate code
        sys.exit(0 if overall_healthy else 1)
        
    except Exception as e:
        logger.error(f"Health check failed with exception: {e}")
        if args.json:
            print(json.dumps({"status": "error", "error": str(e)}))
        else:
            print(f"✗ Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
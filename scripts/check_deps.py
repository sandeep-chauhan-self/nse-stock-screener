import logging
#!/usr/bin/env python3
"""
Enhanced Dependency Checker for NSE Stock Screener

This script validates all dependencies are properly installed and configured
for the testing and CI infrastructure.

Features:
- Validates core dependencies from requirements.txt
- Checks testing dependencies for pytest, coverage, etc.
- Validates Docker installation for containerization
- Checks development tools (ruff, mypy, etc.)
- Provides detailed installation instructions
- Validates package versions for compatibility
"""

from pathlib import Path
import json
import os
import sys

from typing import List, Dict, Tuple, Optional
import importlib
import importlib.util
import pkg_resources
import subprocess


class Color:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class DependencyChecker:
    """Enhanced dependency checker with comprehensive validation"""
    
    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path(__file__).parent.parent
        self.requirements_file = self.root_dir / "requirements.txt"
        self.pyproject_file = self.root_dir / "pyproject.toml"
        self.results = {
            'core': {'passed': [], 'failed': [], 'warnings': []},
            'testing': {'passed': [], 'failed': [], 'warnings': []},
            'dev': {'passed': [], 'failed': [], 'warnings': []},
            'docker': {'passed': [], 'failed': [], 'warnings': []},
            'system': {'passed': [], 'failed': [], 'warnings': []}
        }
        
    def print_header(self, title: str):
        """Print a formatted section header"""
        print(f"\n{Color.BOLD}{Color.BLUE}{'='*60}{Color.END}")
        print(f"{Color.BOLD}{Color.BLUE}{title:^60}{Color.END}")
        print(f"{Color.BOLD}{Color.BLUE}{'='*60}{Color.END}")
        
    def print_status(self, message: str, status: str = "info", indent: int = 0):
        """Print a formatted status message"""
        indent_str = "  " * indent
        if status == "pass":
            print(f"{indent_str}{Color.GREEN}✓{Color.END} {message}")
        elif status == "fail":
            print(f"{indent_str}{Color.RED}✗{Color.END} {message}")
        elif status == "warn":
            print(f"{indent_str}{Color.YELLOW}⚠{Color.END} {message}")
        elif status == "info":
            print(f"{indent_str}{Color.CYAN}ℹ{Color.END} {message}")
        else:
            print(f"{indent_str}{message}")
            
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements"""
        self.print_header("Python Version Check")
        
        version = sys.version_info
        required_major, required_minor = 3, 9
        
        if version.major == required_major and version.minor >= required_minor:
            self.print_status(f"Python {version.major}.{version.minor}.{version.micro} ✓", "pass")
            self.results['system']['passed'].append(f"Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.print_status(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python {required_major}.{required_minor}+", "fail")
            self.results['system']['failed'].append(f"Python version {version.major}.{version.minor}.{version.micro}")
            return False
            
    def get_installed_packages(self) -> Dict[str, str]:
        """Get dictionary of installed packages and their versions"""
        try:
            installed = {}
            for dist in pkg_resources.working_set:
                installed[dist.project_name.lower()] = dist.version
            return installed
        except Exception as e:
            self.print_status(f"Error getting installed packages: {e}", "warn")
            return {}
            
    def parse_requirements(self) -> List[Tuple[str, str, str]]:
        """Parse requirements.txt and return list of (package, operator, version)"""
        requirements = []
        
        if not self.requirements_file.exists():
            self.print_status(f"Requirements file not found: {self.requirements_file}", "fail")
            return requirements
            
        try:
            with open(self.requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package==version or package>=version
                        if '>=' in line:
                            package, version = line.split('>=')
                            requirements.append((package.strip(), '>=', version.strip()))
                        elif '==' in line:
                            package, version = line.split('==')
                            requirements.append((package.strip(), '==', version.strip()))
                        elif '>' in line:
                            package, version = line.split('>')
                            requirements.append((package.strip(), '>', version.strip()))
                        else:
                            # Package without version specifier
                            requirements.append((line.strip(), '', ''))
        except Exception as e:
            self.print_status(f"Error parsing requirements.txt: {e}", "fail")
            
        return requirements
        
    def check_core_dependencies(self) -> bool:
        """Check core application dependencies"""
        self.print_header("Core Dependencies Check")
        
        requirements = self.parse_requirements()
        if not requirements:
            self.print_status("No requirements found to check", "warn")
            return False
            
        installed = self.get_installed_packages()
        all_passed = True
        
        # Core packages that are critical
        core_packages = {
            'pandas', 'numpy', 'matplotlib', 'yfinance', 'requests',
            'scipy', 'scikit-learn', 'ta-lib', 'plotly'
        }
        
        for package, operator, version in requirements:
            package_lower = package.lower()
            
            if package_lower in core_packages:
                if package_lower in installed:
                    installed_version = installed[package_lower]
                    
                    # Check version compatibility
                    version_ok = True
                    if operator == '>=' and version:
                        from packaging import version as pkg_version
                        version_ok = pkg_version.parse(installed_version) >= pkg_version.parse(version)
                    elif operator == '==' and version:
                        version_ok = installed_version == version
                        
                    if version_ok:
                        self.print_status(f"{package} {installed_version} ✓", "pass", 1)
                        self.results['core']['passed'].append(f"{package} {installed_version}")
                    else:
                        self.print_status(f"{package} {installed_version} (required: {operator}{version})", "fail", 1)
                        self.results['core']['failed'].append(f"{package} version mismatch")
                        all_passed = False
                else:
                    self.print_status(f"{package} - Not installed", "fail", 1)
                    self.results['core']['failed'].append(f"{package} missing")
                    all_passed = False
                    
        return all_passed
        
    def check_testing_dependencies(self) -> bool:
        """Check testing framework dependencies"""
        self.print_header("Testing Dependencies Check")
        
        testing_packages = [
            ('pytest', 'Test framework'),
            ('pytest-cov', 'Coverage plugin'),
            ('pytest-mock', 'Mocking framework'),
            ('pytest-xdist', 'Parallel testing'),
            ('pytest-timeout', 'Test timeout handling'),
            ('coverage', 'Coverage measurement'),
            ('factory-boy', 'Test data generation'),
            ('faker', 'Fake data generation')
        ]
        
        installed = self.get_installed_packages()
        all_passed = True
        
        for package, description in testing_packages:
            package_lower = package.lower()
            if package_lower in installed:
                version = installed[package_lower]
                self.print_status(f"{package} {version} - {description} ✓", "pass", 1)
                self.results['testing']['passed'].append(f"{package} {version}")
            else:
                self.print_status(f"{package} - {description} (Not installed)", "warn", 1)
                self.results['testing']['warnings'].append(f"{package} missing")
                # Testing deps are warnings, not failures for core functionality
                
        # Check if pytest is functional
        try:
            result = subprocess.run([sys.executable, '-m', 'pytest', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.print_status("pytest is functional ✓", "pass", 1)
                self.results['testing']['passed'].append("pytest functional")
            else:
                self.print_status("pytest command failed", "fail", 1)
                self.results['testing']['failed'].append("pytest non-functional")
                all_passed = False
        except Exception as e:
            self.print_status(f"pytest check failed: {e}", "warn", 1)
            self.results['testing']['warnings'].append("pytest check failed")
            
        return all_passed
        
    def check_dev_tools(self) -> bool:
        """Check development tools"""
        self.print_header("Development Tools Check")
        
        dev_tools = [
            ('ruff', 'Code linting and formatting'),
            ('mypy', 'Static type checking'),
            ('black', 'Code formatting'),
            ('isort', 'Import sorting'),
            ('bandit', 'Security linting'),
            ('pre-commit', 'Git hooks')
        ]
        
        installed = self.get_installed_packages()
        
        for tool, description in dev_tools:
            tool_lower = tool.lower()
            if tool_lower in installed:
                version = installed[tool_lower]
                self.print_status(f"{tool} {version} - {description} ✓", "pass", 1)
                self.results['dev']['passed'].append(f"{tool} {version}")
                
                # Test if tool is functional
                try:
                    if tool == 'ruff':
                        result = subprocess.run([tool, '--version'], capture_output=True, text=True, timeout=5)
                    elif tool == 'mypy':
                        result = subprocess.run([tool, '--version'], capture_output=True, text=True, timeout=5)
                    else:
                        continue  # Skip functionality test for other tools
                        
                    if result.returncode == 0:
                        self.print_status(f"{tool} command functional", "pass", 2)
                    else:
                        self.print_status(f"{tool} command failed", "warn", 2)
                        self.results['dev']['warnings'].append(f"{tool} non-functional")
                except Exception:
                    self.print_status(f"{tool} command test failed", "warn", 2)
                    self.results['dev']['warnings'].append(f"{tool} test failed")
            else:
                self.print_status(f"{tool} - {description} (Not installed)", "warn", 1)
                self.results['dev']['warnings'].append(f"{tool} missing")
                
        return True  # Dev tools are not critical for core functionality
        
    def check_docker(self) -> bool:
        """Check Docker installation and functionality"""
        self.print_header("Docker Environment Check")
        
        # Check Docker command
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.print_status(f"Docker installed: {version} ✓", "pass", 1)
                self.results['docker']['passed'].append(f"Docker: {version}")
                
                # Check if Docker daemon is running
                try:
                    result = subprocess.run(['docker', 'info'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.print_status("Docker daemon is running ✓", "pass", 1)
                        self.results['docker']['passed'].append("Docker daemon running")
                        
                        # Check Docker Compose
                        try:
                            result = subprocess.run(['docker', 'compose', 'version'], 
                                                  capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                compose_version = result.stdout.strip()
                                self.print_status(f"Docker Compose: {compose_version} ✓", "pass", 1)
                                self.results['docker']['passed'].append(f"Docker Compose: {compose_version}")
                            else:
                                self.print_status("Docker Compose not available", "warn", 1)
                                self.results['docker']['warnings'].append("Docker Compose missing")
                        except Exception:
                            self.print_status("Docker Compose check failed", "warn", 1)
                            self.results['docker']['warnings'].append("Docker Compose check failed")
                            
                        return True
                    else:
                        self.print_status("Docker daemon not running", "warn", 1)
                        self.results['docker']['warnings'].append("Docker daemon not running")
                        return False
                except Exception as e:
                    self.print_status(f"Docker daemon check failed: {e}", "warn", 1)
                    self.results['docker']['warnings'].append("Docker daemon check failed")
                    return False
            else:
                self.print_status("Docker command failed", "warn", 1)
                self.results['docker']['warnings'].append("Docker command failed")
                return False
        except FileNotFoundError:
            self.print_status("Docker not installed", "warn", 1)
            self.results['docker']['warnings'].append("Docker not installed")
            return False
        except Exception as e:
            self.print_status(f"Docker check failed: {e}", "warn", 1)
            self.results['docker']['warnings'].append(f"Docker check error: {e}")
            return False
            
    def check_project_structure(self) -> bool:
        """Check if project structure is correct"""
        self.print_header("Project Structure Check")
        
        required_files = [
            ('requirements.txt', 'Dependencies specification'),
            ('pyproject.toml', 'Project configuration'),
            ('Dockerfile', 'Container configuration'),
            ('docker-compose.yml', 'Multi-container setup'),
            ('.github/workflows/ci.yml', 'CI pipeline'),
            ('.github/workflows/release.yml', 'Release pipeline'),
            ('src/__init__.py', 'Source package'),
            ('tests/conftest.py', 'Test configuration'),
            ('tests/fixtures/test_data.py', 'Test fixtures')
        ]
        
        required_dirs = [
            ('src', 'Source code'),
            ('tests', 'Test suite'),
            ('data', 'Data directory'),
            ('output', 'Output directory'),
            ('scripts', 'Utility scripts')
        ]
        
        all_passed = True
        
        # Check files
        for file_path, description in required_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                self.print_status(f"{file_path} - {description} ✓", "pass", 1)
                self.results['system']['passed'].append(f"File: {file_path}")
            else:
                self.print_status(f"{file_path} - {description} (Missing)", "warn", 1)
                self.results['system']['warnings'].append(f"Missing file: {file_path}")
                
        # Check directories
        for dir_path, description in required_dirs:
            full_path = self.root_dir / dir_path
            if full_path.exists() and full_path.is_dir():
                self.print_status(f"{dir_path}/ - {description} ✓", "pass", 1)
                self.results['system']['passed'].append(f"Directory: {dir_path}")
            else:
                self.print_status(f"{dir_path}/ - {description} (Missing)", "warn", 1)
                self.results['system']['warnings'].append(f"Missing directory: {dir_path}")
                
        return all_passed
        
    def run_quick_tests(self) -> bool:
        """Run a quick test to verify the system works"""
        self.print_header("Quick Functionality Test")
        
        try:
            # Test basic imports
            test_imports = [
                ('pandas', 'Data manipulation'),
                ('numpy', 'Numerical computing'),
                ('matplotlib', 'Plotting'),
                ('src', 'Local source code')
            ]
            
            for module, description in test_imports:
                try:
                    importlib.import_module(module)
                    self.print_status(f"Import {module} - {description} ✓", "pass", 1)
                    self.results['system']['passed'].append(f"Import: {module}")
                except ImportError as e:
                    self.print_status(f"Import {module} failed: {e}", "fail", 1)
                    self.results['system']['failed'].append(f"Import failed: {module}")
                    return False
                    
            # Test pytest if available
            if 'pytest' in self.get_installed_packages():
                try:
                    # Run a simple test discovery
                    result = subprocess.run([
                        sys.executable, '-m', 'pytest', 
                        '--collect-only', '-q', str(self.root_dir / 'tests')
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        test_count = result.stdout.count('::')
                        self.print_status(f"Test discovery successful ({test_count} tests found) ✓", "pass", 1)
                        self.results['testing']['passed'].append(f"Test discovery: {test_count} tests")
                    else:
                        self.print_status("Test discovery failed", "warn", 1)
                        self.results['testing']['warnings'].append("Test discovery failed")
                except Exception as e:
                    self.print_status(f"Test discovery error: {e}", "warn", 1)
                    self.results['testing']['warnings'].append(f"Test discovery error: {e}")
                    
            return True
            
        except Exception as e:
            self.print_status(f"Quick test failed: {e}", "fail", 1)
            self.results['system']['failed'].append(f"Quick test error: {e}")
            return False
            
    def print_summary(self):
        """Print comprehensive summary of all checks"""
        self.print_header("Dependency Check Summary")
        
        total_passed = sum(len(category['passed']) for category in self.results.values())
        total_failed = sum(len(category['failed']) for category in self.results.values())
        total_warnings = sum(len(category['warnings']) for category in self.results.values())
        
        print(f"\n{Color.BOLD}Overall Status:{Color.END}")
        print(f"  {Color.GREEN}✓ Passed:{Color.END} {total_passed}")
        print(f"  {Color.RED}✗ Failed:{Color.END} {total_failed}")
        logging.warning(f"  {Color.YELLOW}⚠ Warnings:{Color.END} {total_warnings}")
        
        # Category breakdown
        for category, results in self.results.items():
            if results['passed'] or results['failed'] or results['warnings']:
                print(f"\n{Color.BOLD}{category.title()}:{Color.END}")
                for item in results['passed']:
                    print(f"  {Color.GREEN}✓{Color.END} {item}")
                for item in results['failed']:
                    print(f"  {Color.RED}✗{Color.END} {item}")
                for item in results['warnings']:
                    print(f"  {Color.YELLOW}⚠{Color.END} {item}")
                    
        # Installation instructions for missing packages
        if total_failed > 0 or total_warnings > 0:
            self.print_installation_instructions()
            
        # Overall result
        if total_failed == 0:
            print(f"\n{Color.BOLD}{Color.GREEN}🎉 All critical dependencies are satisfied!{Color.END}")
            if total_warnings > 0:
                logging.warning(f"{Color.YELLOW}Note: {total_warnings} warning(s) found but system should work.{Color.END}")
            return True
        else:
            print(f"\n{Color.BOLD}{Color.RED}❌ {total_failed} critical issues found. Please fix before proceeding.{Color.END}")
            return False
            
    def print_installation_instructions(self):
        """Print installation instructions for missing dependencies"""
        print(f"\n{Color.BOLD}{Color.BLUE}Installation Instructions:{Color.END}")
        
        print(f"\n{Color.BOLD}1. Install core dependencies:{Color.END}")
        print("   pip install -r requirements.txt")
        
        print(f"\n{Color.BOLD}2. Install development dependencies:{Color.END}")
        print("   pip install -e \".[dev]\"")
        print("   # or")
        print("   pip install pytest pytest-cov ruff mypy black isort bandit")
        
        print(f"\n{Color.BOLD}3. Install Docker (if needed):{Color.END}")
        print("   - Windows: Download Docker Desktop from docker.com")
        print("   - macOS: Download Docker Desktop from docker.com")
        print("   - Linux: sudo apt-get install docker.io docker-compose-plugin")
        
        print(f"\n{Color.BOLD}4. Verify installation:{Color.END}")
        print("   python scripts/check_deps.py")
        
        print(f"\n{Color.BOLD}5. Run tests:{Color.END}")
        print("   pytest tests/ -v")
        
        print(f"\n{Color.BOLD}6. Run with Docker:{Color.END}")
        print("   docker-compose up --build")


def main():
    """Main entry point"""
    print(f"{Color.BOLD}{Color.BLUE}NSE Stock Screener - Enhanced Dependency Checker{Color.END}")
    print(f"{Color.CYAN}Validating dependencies for testing and CI infrastructure...{Color.END}")
    
    checker = DependencyChecker()
    
    # Run all checks
    checker.check_python_version()
    checker.check_core_dependencies()
    checker.check_testing_dependencies()
    checker.check_dev_tools()
    checker.check_docker()
    checker.check_project_structure()
    checker.run_quick_tests()
    
    # Print summary
    success = checker.print_summary()
    
    # Export results to JSON for CI
    output_file = checker.root_dir / "dependency_check_results.json"
    with open(output_file, 'w') as f:
        json.dump(checker.results, f, indent=2)
    print(f"\n{Color.CYAN}Results exported to: {output_file}{Color.END}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Setup script for NSE Stock Screener development environment.
This script helps developers set up the project quickly with all dependencies.
"""
import os
import subprocess
import sys
from pathlib import Path
def run_command(cmd: str, description: str = "", check: bool = True) -> bool:
    """Run a shell command with error handling."""
    if description:
        print(f"📦 {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
def check_python_version() -> bool:
    """Check if Python version is supported."""
    version = sys.version_info
    if version < (3, 9):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("📋 Required: Python 3.9 or higher")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
    return True
def install_dependencies() -> bool:
    """Install project dependencies."""
    print("📦 Installing project dependencies...")
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        return False
    # Install development dependencies
    if not run_command(f"{sys.executable} -m pip install -e .[dev]"):
        print("⚠️  Failed to install with -e, trying regular install...")
        if not run_command(f"{sys.executable} -m pip install .[dev]"):
            print("❌ Failed to install dependencies")
            return False
    return True
def setup_pre_commit() -> bool:
    """Set up pre-commit hooks."""
    if not Path(".pre-commit-config.yaml").exists():
        print("⚠️  .pre-commit-config.yaml not found, skipping pre-commit setup")
        return True
    print("🔗 Setting up pre-commit hooks...")
    return run_command("pre-commit install", check=False)
def create_directories() -> bool:
    """Create necessary directories."""
    directories = [
        "data",
        "data/temp",
        "output",
        "output/reports",
        "output/charts",
        "output/backtests",
        "logs",
        "test-output"
    ]
    print("📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")
    return True
def check_optional_dependencies() -> None:
    """Check for optional dependencies and provide installation guidance."""
    print("\n🔍 Checking optional dependencies...")
    optional_deps = {
        "ta-lib": "Technical Analysis Library (for advanced indicators)",
        "redis": "Redis server (for caching)",
        "docker": "Docker (for containerization)"
    }
    for dep, description in optional_deps.items():
        try:
            if dep == "docker":
                subprocess.run(["docker", "--version"],
                             capture_output=True, check=True)
            elif dep == "redis":
                subprocess.run(["redis-cli", "--version"],
                             capture_output=True, check=True)
            else:
                __import__(dep.replace("-", "_"))
            print(f"  ✅ {dep}: {description}")
        except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ⚠️  {dep}: {description} (optional)")
def run_initial_checks() -> bool:
    """Run initial health checks."""
    print("\n🏥 Running initial health checks...")
    # Check dependencies
    if Path("scripts/check_deps.py").exists():
        return run_command(f"{sys.executable} scripts/check_deps.py", check=False)
    # Basic import test
    return run_command(
        f"{sys.executable} -c \"import src; print('✅ Basic imports working')\"",
        "Testing basic imports",
        check=False
    )
def show_next_steps() -> None:
    """Show next steps to the user."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\n📋 Next steps:")
    print("  1. Run the dependency checker:")
    print("     python scripts/check_deps.py")
    print("\n  2. Run the test suite:")
    print("     pytest tests/ -v")
    print("\n  3. Start a stock analysis:")
    print("     python -m src.enhanced_early_warning_system --help")
    print("\n  4. Or use Docker:")
    print("     docker-compose up nse-screener")
    print("\n📖 Documentation:")
    print("  - README.md: Getting started guide")
    print("  - Requirements.md: Detailed requirements")
    print("  - docs/: Additional documentation")
    print("\n🐳 Docker usage:")
    print("  - docker build -t nse-screener .")
    print("  - docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output nse-screener")
def main() -> int:
    """Main setup function."""
    print("🚀 NSE Stock Screener Development Setup")
    print("=" * 50)
    # Check Python version
    if not check_python_version():
        return 1
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        return 1
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return 1
    # Setup pre-commit
    setup_pre_commit()
    # Check optional dependencies
    check_optional_dependencies()
    # Run initial checks
    run_initial_checks()
    # Show next steps
    show_next_steps()
    return 0
if __name__ == "__main__":
    sys.exit(main())

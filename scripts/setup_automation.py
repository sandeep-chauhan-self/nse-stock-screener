#!/usr/bin/env python3
"""
NSE Stock Screener - Automation Setup Helper
Install and verify automation dependencies
"""

import subprocess
import sys
from pathlib import Path

from typing import Optional

def check_dependency(module_name: str, package_name: Optional[str] = None) -> bool:
    """Check if a module is available"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {module_name} is available")
        return True
    except ImportError:
        print(f"❌ {module_name} is missing (install: pip install {package_name})")
        return False

def install_dependencies():
    """Install automation dependencies"""
    requirements_file = Path(__file__).parent / "automation_requirements.txt"
    
    if not requirements_file.exists():
        print("❌ automation_requirements.txt not found")
        return False
    
    print("📦 Installing automation dependencies...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def verify_automation_components():
    """Verify automation components are working"""
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"
    
    print("\n🔍 Verifying automation components...")
    
    # Check core scripts exist
    core_scripts = [
        "automation_manager.py",
        "simple_automation.py",
        "smart_scheduler.py",
        "event_driven_automation.py",
        "multi_market_automation.py",
        "sector_automation.py",
        "enterprise_dashboard.py"
    ]
    
    missing_scripts = []
    for script in core_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"✅ {script} found")
        else:
            print(f"❌ {script} missing")
            missing_scripts.append(script)
    
    return len(missing_scripts) == 0

def test_simple_automation():
    """Test basic automation functionality"""
    print("\n🧪 Testing simple automation...")
    
    try:
        # Try to import and test simple automation
        sys.path.insert(0, str(Path(__file__).parent))
        from simple_automation import SimpleAutomation
        
        # Create automation instance
        automation = SimpleAutomation()
        
        # Test status
        status = automation.get_status()
        print(f"✅ Simple automation status: {status}")
        
        return True
    except Exception as e:
        print(f"❌ Simple automation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 NSE Stock Screener - Automation Setup")
    print("=" * 45)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check basic dependencies
    print("\n📋 Checking basic dependencies...")
    basic_deps = [
        ("json", None),
        ("datetime", None),
        ("pathlib", None),
        ("threading", None),
        ("argparse", None)
    ]
    
    basic_ok = True
    for module, package in basic_deps:
        if not check_dependency(module, package):
            basic_ok = False
    
    if not basic_ok:
        print("❌ Basic dependencies missing - Python installation issue")
        return False
    
    # Check optional dependencies
    print("\n📋 Checking optional dependencies...")
    optional_deps = [
        ("schedule", "schedule"),
        ("psutil", "psutil"),
        ("flask", "flask"),
        ("plotly", "plotly"),
        ("requests", "requests"),
        ("websockets", "websockets")
    ]
    
    missing_optional = []
    for module, package in optional_deps:
        if not check_dependency(module, package):
            missing_optional.append(package)
    
    # Offer to install missing dependencies
    if missing_optional:
        print(f"\n📦 {len(missing_optional)} optional dependencies missing")
        install_choice = input("Install missing dependencies? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if install_dependencies():
                print("✅ Installation completed")
            else:
                print("❌ Installation failed")
                print("   You can still use simple_automation.py for basic features")
        else:
            print("⚠️ Skipping installation")
            print("   You can still use simple_automation.py for basic features")
    else:
        print("✅ All optional dependencies available")
    
    # Verify components
    if verify_automation_components():
        print("✅ All automation components available")
    else:
        print("❌ Some automation components missing")
    
    # Test simple automation
    if test_simple_automation():
        print("✅ Simple automation working")
    else:
        print("❌ Simple automation test failed")
    
    print("\n🎯 Setup Summary:")
    print("=" * 20)
    print("✅ Basic automation: simple_automation.py")
    
    if not missing_optional:
        print("✅ Advanced automation: smart_scheduler.py")
        print("✅ Event automation: event_driven_automation.py") 
        print("✅ Multi-market: multi_market_automation.py")
        print("✅ Sector analysis: sector_automation.py")
        print("✅ Enterprise dashboard: enterprise_dashboard.py")
    else:
        print("⚠️ Advanced features: Install dependencies first")
    
    print("\n📚 Quick Start:")
    print("  python scripts/simple_automation.py --run")
    print("  python scripts/simple_automation.py --schedule 8")
    
    if not missing_optional:
        print("  python scripts/smart_scheduler.py --status")
        print("  python scripts/enterprise_dashboard.py --start")

if __name__ == "__main__":
    main()
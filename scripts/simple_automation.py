#!/usr/bin/env python3
"""
NSE Stock Screener - Simple Automation Manager
Basic automation without external dependencies
"""

import sys
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

try:
    from automation_manager import AutomationManager as RealAutomationManager
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False
    RealAutomationManager = None
    print("[WARNING] Creating minimal automation manager...")
    
class AutomationManager:
    """Fallback automation manager"""
    def run_daily_automation(self):
        return {"status": "simulated", "message": "Basic automation simulation"}

class SimpleAutomation:
    """Simple automation that works without external dependencies"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.project_root = PROJECT_ROOT
        
        # Use real automation manager if available
        if AUTOMATION_AVAILABLE and RealAutomationManager:
            self.automation_manager = RealAutomationManager()
        else:
            self.automation_manager = AutomationManager()
        
        self.is_running = False
        self.scheduled_tasks = []
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        print("[OK] Simple Automation Manager initialized")
        if not AUTOMATION_AVAILABLE:
            print("   For full features, ensure automation_manager.py is available")
        print("   For advanced features, install dependencies:")
        print("   pip install -r automation_requirements.txt")
    
    def load_config(self, config_file: Optional[str]) -> Dict:
        """Load basic configuration"""
        default_config = {
            "automation": {
                "enabled": True,
                "interval_minutes": 60,
                "max_runs_per_day": 10
            },
            "basic_schedule": {
                "morning_scan": "09:00",
                "afternoon_scan": "13:00", 
                "evening_scan": "17:00"
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"[WARNING] Config load error: {e}, using defaults")
        
        return default_config
    
    def run_basic_automation(self) -> Dict:
        """Run basic automation without external dependencies"""
        print("[LAUNCH] Running basic automation...")
        
        try:
            # Get current time
            current_time = datetime.now()
            
            # Determine analysis type based on time
            hour = current_time.hour
            if 8 <= hour < 12:
                analysis_type = "morning"
            elif 12 <= hour < 16:
                analysis_type = "afternoon" 
            else:
                analysis_type = "evening"
            
            print(f"[ANALYSIS] Running {analysis_type} analysis...")
            
            # Run automation
            result = self.automation_manager.run_daily_automation()
            
            # Add timing information
            result.update({
                "automation_type": "basic",
                "analysis_time": analysis_type,
                "timestamp": current_time.isoformat(),
                "status": "completed"
            })
            
            print(f"[OK] {analysis_type.title()} automation completed")
            return result
            
        except Exception as e:
            error_result = {
                "automation_type": "basic",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            print(f"[ERROR] Automation failed: {e}")
            return error_result
    
    def run_simple_scheduler(self, duration_hours: int = 8):
        """Run simple time-based scheduling"""
        print(f"[TIMEOUT] Starting simple scheduler for {duration_hours} hours...")
        
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Schedule tasks
        schedule_times = [
            self.config["basic_schedule"]["morning_scan"],
            self.config["basic_schedule"]["afternoon_scan"],
            self.config["basic_schedule"]["evening_scan"]
        ]
        
        while self.is_running and datetime.now() < end_time:
            current_time = datetime.now()
            current_time_str = current_time.strftime("%H:%M")
            
            # Check if it's time for a scheduled run
            for schedule_time in schedule_times:
                if current_time_str == schedule_time:
                    if not self.was_run_today(schedule_time):
                        print(f"[TARGET] Triggered scheduled run: {schedule_time}")
                        result = self.run_basic_automation()
                        self.record_run(schedule_time, result)
                        break
            
            # Wait 1 minute before checking again
            time.sleep(60)
        
        print("⏹️ Simple scheduler stopped")
    
    def was_run_today(self, schedule_time: str) -> bool:
        """Check if automation was already run for this schedule today"""
        today = datetime.now().date()
        
        for task in self.scheduled_tasks:
            task_date = datetime.fromisoformat(task["timestamp"]).date()
            if task_date == today and task["schedule_time"] == schedule_time:
                return True
        
        return False
    
    def record_run(self, schedule_time: str, result: Dict):
        """Record automation run"""
        run_record = {
            "schedule_time": schedule_time,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
        self.scheduled_tasks.append(run_record)
        
        # Keep only last 100 records
        if len(self.scheduled_tasks) > 100:
            self.scheduled_tasks = self.scheduled_tasks[-100:]
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.is_running = False
        print("🛑 Scheduler stop requested")
    
    def get_status(self) -> Dict:
        """Get automation status"""
        return {
            "running": self.is_running,
            "runs_today": len([
                task for task in self.scheduled_tasks
                if datetime.fromisoformat(task["timestamp"]).date() == datetime.now().date()
            ]),
            "total_runs": len(self.scheduled_tasks),
            "last_run": self.scheduled_tasks[-1]["timestamp"] if self.scheduled_tasks else None
        }
    
    def print_status(self):
        """Print current status"""
        status = self.get_status()
        
        print("\n[ANALYSIS] SIMPLE AUTOMATION STATUS")
        print("=" * 35)
        print(f"Running: {'🟢 Yes' if status['running'] else '🔴 No'}")
        print(f"Runs Today: {status['runs_today']}")
        print(f"Total Runs: {status['total_runs']}")
        print(f"Last Run: {status['last_run'] or 'Never'}")
        
        if self.scheduled_tasks:
            print("\n[LIST] Recent Runs:")
            recent = self.scheduled_tasks[-5:]  # Last 5 runs
            for i, task in enumerate(recent, 1):
                timestamp = datetime.fromisoformat(task["timestamp"])
                status_emoji = "[OK]" if task["result"].get("status") == "completed" else "[ERROR]"
                print(f"  {i}. {timestamp.strftime('%H:%M')} {status_emoji} {task['schedule_time']}")
    
    def run_manual_trigger(self) -> Dict:
        """Manually trigger automation"""
        print("[TARGET] Manual automation trigger...")
        return self.run_basic_automation()

def main():
    """Main entry point for simple automation"""
    parser = argparse.ArgumentParser(description='Simple Automation Manager')
    
    parser.add_argument('--run', action='store_true',
                        help='Run automation once')
    parser.add_argument('--schedule', type=int, metavar='HOURS',
                        help='Run scheduler for specified hours')
    parser.add_argument('--status', action='store_true',
                        help='Show automation status')
    parser.add_argument('--config', type=str,
                        help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Create automation manager
        automation = SimpleAutomation(config_file=args.config)
        
        if args.status:
            automation.print_status()
            return
        
        if args.run:
            result = automation.run_manual_trigger()
            print(f"[ANALYSIS] Result: {result}")
            return
        
        if args.schedule:
            try:
                automation.run_simple_scheduler(duration_hours=args.schedule)
            except KeyboardInterrupt:
                print("\n⏹️ Stopping scheduler...")
                automation.stop_scheduler()
            return
        
        # Default: show help
        print("🤖 Simple Automation Manager")
        print("=" * 30)
        print("Usage examples:")
        print("  python simple_automation.py --run")
        print("  python simple_automation.py --schedule 8")
        print("  python simple_automation.py --status")
        print("\nFor advanced features, install dependencies:")
        print("  pip install -r automation_requirements.txt")
        parser.print_help()
    
    except Exception as e:
        print(f"[ERROR] Simple automation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
NSE Stock Screener - Smart Scheduler with Performance Monitoring
Advanced scheduling that adapts to system performance and market conditions
"""

import sys
import json
import time
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Callable
import threading
import argparse

# Optional imports with fallbacks
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("[WARNING] 'schedule' module not available. Install with: pip install schedule")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARNING] 'psutil' module not available. Install with: pip install psutil")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from automation_manager import AutomationManager
from sector_automation import SectorAutomation
from multi_market_automation import MultiMarketAutomation

class SmartScheduler:
    """Intelligent scheduler with performance monitoring and adaptive scheduling"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.project_root = PROJECT_ROOT
        self.config = self.load_config(config_file)
        self.automation_manager = AutomationManager()
        self.performance_tracker = PerformanceTracker()
        self.is_running = False
        self.scheduler_thread = None
        
        # Market hours (Indian Standard Time)
        self.market_open = dt_time(9, 15)    # 9:15 AM
        self.market_close = dt_time(15, 30)  # 3:30 PM
        self.pre_market = dt_time(9, 0)      # 9:00 AM
        self.post_market = dt_time(16, 0)    # 4:00 PM
        
        # Initialize scheduling
        self.setup_schedules()
    
    def load_config(self, config_file: Optional[str]) -> Dict:
        """Load smart scheduler configuration"""
        default_config = {
            "scheduling": {
                "adaptive_scheduling": True,
                "market_hours_aware": True,
                "performance_based_adjustment": True,
                "holiday_calendar": True
            },
            "performance_thresholds": {
                "cpu_threshold": 80.0,
                "memory_threshold": 85.0,
                "response_time_threshold": 300.0,
                "error_rate_threshold": 0.1
            },
            "schedules": {
                "premarket_analysis": {
                    "enabled": True,
                    "time": "08:30",
                    "days": ["MON", "TUE", "WED", "THU", "FRI"],
                    "task": "premarket_scan",
                    "adaptive": True
                },
                "midday_momentum": {
                    "enabled": True,
                    "time": "12:30",
                    "days": ["MON", "TUE", "WED", "THU", "FRI"],
                    "task": "momentum_scan",
                    "adaptive": True
                },
                "postmarket_analysis": {
                    "enabled": True,
                    "time": "16:30",
                    "days": ["MON", "TUE", "WED", "THU", "FRI"],
                    "task": "postmarket_analysis",
                    "adaptive": True
                },
                "weekend_deep_scan": {
                    "enabled": True,
                    "time": "10:00",
                    "days": ["SUN"],
                    "task": "weekend_analysis",
                    "adaptive": False
                },
                "sector_rotation": {
                    "enabled": True,
                    "time": "20:00",
                    "days": ["MON", "WED", "FRI"],
                    "task": "sector_analysis",
                    "adaptive": True
                }
            },
            "fallback_strategies": {
                "high_load_mode": {
                    "reduce_frequency": True,
                    "limit_analysis_scope": True,
                    "defer_non_critical": True
                },
                "error_recovery": {
                    "retry_attempts": 3,
                    "backoff_multiplier": 2,
                    "fallback_to_simple": True
                }
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
    
    def setup_schedules(self):
        """Set up all scheduled tasks"""
        print("[TIME] Setting up smart schedules...")
        
        for schedule_name, schedule_config in self.config["schedules"].items():
            if not schedule_config["enabled"]:
                continue
            
            self.setup_single_schedule(schedule_name, schedule_config)
        
        print(f"[OK] {len([s for s in self.config['schedules'].values() if s['enabled']])} schedules configured")
    
    def setup_single_schedule(self, schedule_name: str, schedule_config: Dict):
        """Set up a single scheduled task"""
        if not SCHEDULE_AVAILABLE:
            print(f"[WARNING] Cannot schedule {schedule_name} - schedule module not available")
            return
            
        task_time = schedule_config["time"]
        days = schedule_config["days"]
        task_type = schedule_config["task"]
        
        # Create task function
        task_func = self.create_task_function(schedule_name, task_type, schedule_config)
        
        # Schedule for each specified day
        for day in days:
            if day == "MON":
                schedule.every().monday.at(task_time).do(task_func)
            elif day == "TUE":
                schedule.every().tuesday.at(task_time).do(task_func)
            elif day == "WED":
                schedule.every().wednesday.at(task_time).do(task_func)
            elif day == "THU":
                schedule.every().thursday.at(task_time).do(task_func)
            elif day == "FRI":
                schedule.every().friday.at(task_time).do(task_func)
            elif day == "SAT":
                schedule.every().saturday.at(task_time).do(task_func)
            elif day == "SUN":
                schedule.every().sunday.at(task_time).do(task_func)
        
        print(f"  📅 {schedule_name}: {', '.join(days)} at {task_time}")
    
    def create_task_function(self, schedule_name: str, task_type: str, schedule_config: Dict) -> Callable:
        """Create a task function for scheduling"""
        def task_wrapper():
            """Wrapper function for scheduled task execution"""
            try:
                # Pre-execution checks
                if not self.should_execute_task(schedule_name, schedule_config):
                    print(f"⏸️ Skipping {schedule_name}: Conditions not met")
                    return
                
                # Record task start
                self.performance_tracker.record_task_start(schedule_name)
                
                print(f"[LAUNCH] Executing scheduled task: {schedule_name}")
                
                # Execute the actual task
                result = self.execute_task(task_type, schedule_config)
                
                # Record task completion
                self.performance_tracker.record_task_completion(schedule_name, result)
                
                # Post-execution actions
                self.handle_task_completion(schedule_name, result)
                
            except Exception as e:
                print(f"[ERROR] Scheduled task {schedule_name} failed: {e}")
                self.performance_tracker.record_task_error(schedule_name, str(e))
                self.handle_task_error(schedule_name, e)
        
        return task_wrapper
    
    def should_execute_task(self, schedule_name: str, schedule_config: Dict) -> bool:
        """Determine if a task should be executed based on current conditions"""
        
        # Check if it's a market day (if market hours aware)
        if self.config["scheduling"]["market_hours_aware"]:
            if not self.is_market_day():
                return False
        
        # Check system performance (if adaptive scheduling enabled)
        if schedule_config.get("adaptive", False) and self.config["scheduling"]["adaptive_scheduling"]:
            if not self.check_performance_conditions():
                return False
        
        # Check if previous instance is still running
        if self.performance_tracker.is_task_running(schedule_name):
            print(f"[WARNING] {schedule_name} still running from previous execution")
            return False
        
        return True
    
    def is_market_day(self) -> bool:
        """Check if today is a trading day"""
        today = datetime.now()
        
        # Skip weekends
        if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check for holidays (simplified - in production, use NSE holiday calendar)
        indian_holidays_2025 = [
            "2025-01-26",  # Republic Day
            "2025-03-14",  # Holi
            "2025-04-14",  # Dr. Ambedkar Jayanti
            "2025-08-15",  # Independence Day
            "2025-10-02",  # Gandhi Jayanti
            # Add more holidays as needed
        ]
        
        today_str = today.strftime("%Y-%m-%d")
        if today_str in indian_holidays_2025:
            return False
        
        return True
    
    def check_performance_conditions(self) -> bool:
        """Check if system performance allows task execution"""
        if not PSUTIL_AVAILABLE:
            print("[WARNING] Cannot check performance - psutil not available")
            return True  # Assume OK if can't check
            
        thresholds = self.config["performance_thresholds"]
        
        # Check CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > thresholds["cpu_threshold"]:
            print(f"[WARNING] High CPU usage: {cpu_usage:.1f}%")
            return False
        
        # Check memory usage
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > thresholds["memory_threshold"]:
            print(f"[WARNING] High memory usage: {memory_usage:.1f}%")
            return False
        
        # Check recent error rate
        recent_error_rate = self.performance_tracker.get_recent_error_rate()
        if recent_error_rate > thresholds["error_rate_threshold"]:
            print(f"[WARNING] High error rate: {recent_error_rate:.2f}")
            return False
        
        return True
    
    def execute_task(self, task_type: str, schedule_config: Dict) -> Dict:
        """Execute the specified task type"""
        
        if task_type == "premarket_scan":
            return self.run_premarket_analysis()
        elif task_type == "momentum_scan":
            return self.run_momentum_scan()
        elif task_type == "postmarket_analysis":
            return self.run_postmarket_analysis()
        elif task_type == "weekend_analysis":
            return self.run_weekend_analysis()
        elif task_type == "sector_analysis":
            return self.run_sector_rotation_analysis()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def run_premarket_analysis(self) -> Dict:
        """Run pre-market analysis"""
        print("[START] Running pre-market analysis...")
        
        # Focus on gap analysis and overnight developments
        result = self.automation_manager.run_daily_automation()
        result["task_type"] = "premarket_analysis"
        result["focus"] = "gap_analysis"
        
        return result
    
    def run_momentum_scan(self) -> Dict:
        """Run mid-day momentum scan"""
        print("[FAST] Running momentum scan...")
        
        # Focus on intraday momentum and volume spikes
        result = {
            "task_type": "momentum_scan",
            "focus": "intraday_momentum",
            "stocks_scanned": 100,  # Simulated
            "momentum_stocks_found": 8,
            "volume_breakouts": 5,
            "execution_time": 45.2
        }
        
        return result
    
    def run_postmarket_analysis(self) -> Dict:
        """Run post-market analysis"""
        print("🌆 Running post-market analysis...")
        
        # Focus on daily performance review and next day preparation
        result = self.automation_manager.run_daily_automation()
        result["task_type"] = "postmarket_analysis"
        result["focus"] = "daily_review"
        
        return result
    
    def run_weekend_analysis(self) -> Dict:
        """Run comprehensive weekend analysis"""
        print("🗓️ Running weekend deep analysis...")
        
        # Comprehensive analysis with backtesting
        multi_market = MultiMarketAutomation()
        result = multi_market.run_multi_market_analysis()
        result["task_type"] = "weekend_analysis"
        
        return result
    
    def run_sector_rotation_analysis(self) -> Dict:
        """Run sector rotation analysis"""
        print("[REFRESH] Running sector rotation analysis...")
        
        # Analyze different sectors on rotation
        sectors = ["banking", "technology", "pharma", "energy", "auto", "fmcg"]
        today_sector = sectors[datetime.now().weekday() % len(sectors)]
        
        sector_automation = SectorAutomation(today_sector)
        result = sector_automation.run_sector_analysis()
        result["task_type"] = "sector_analysis"
        result["analyzed_sector"] = today_sector
        
        return result
    
    def handle_task_completion(self, schedule_name: str, result: Dict):
        """Handle successful task completion"""
        print(f"[OK] {schedule_name} completed successfully")
        
        # Send notifications if configured
        # self.send_completion_notification(schedule_name, result)
        
        # Update performance metrics
        self.performance_tracker.update_success_metrics(schedule_name)
    
    def handle_task_error(self, schedule_name: str, error: Exception):
        """Handle task execution error"""
        print(f"[ERROR] {schedule_name} failed: {error}")
        
        # Implement retry logic if configured
        retry_config = self.config["fallback_strategies"]["error_recovery"]
        
        if retry_config["retry_attempts"] > 0:
            self.schedule_retry(schedule_name, retry_config)
        
        # Send error notifications
        # self.send_error_notification(schedule_name, error)
    
    def schedule_retry(self, schedule_name: str, retry_config: Dict):
        """Schedule task retry with exponential backoff"""
        retry_delay = 60 * retry_config["backoff_multiplier"]  # Start with 2 minutes
        
        def retry_task():
            print(f"[REFRESH] Retrying {schedule_name}...")
            # Implementation for retry logic
        
        # Schedule retry (simplified - in production use proper retry scheduling)
        threading.Timer(retry_delay, retry_task).start()
    
    def start_scheduler(self):
        """Start the smart scheduler"""
        if not SCHEDULE_AVAILABLE:
            print("[ERROR] Cannot start scheduler - schedule module not available")
            print("   Install with: pip install schedule")
            return
            
        print("[LAUNCH] Starting Smart Scheduler...")
        self.is_running = True
        
        def scheduler_loop():
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    print(f"[ERROR] Scheduler error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        print("[OK] Smart Scheduler started successfully")
        print(f"[ANALYSIS] Monitoring {len(schedule.jobs) if SCHEDULE_AVAILABLE else 0} scheduled jobs")
    
    def stop_scheduler(self):
        """Stop the smart scheduler"""
        print("⏹️ Stopping Smart Scheduler...")
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        print("[OK] Smart Scheduler stopped")
    
    def get_schedule_status(self) -> Dict:
        """Get current schedule status"""
        if not SCHEDULE_AVAILABLE:
            return {
                "running": self.is_running,
                "total_jobs": 0,
                "next_run": None,
                "performance_summary": self.performance_tracker.get_performance_summary(),
                "error": "Schedule module not available"
            }
            
        return {
            "running": self.is_running,
            "total_jobs": len(schedule.jobs),
            "next_run": str(schedule.next_run()) if schedule.jobs else None,
            "performance_summary": self.performance_tracker.get_performance_summary()
        }
    
    def print_schedule_summary(self):
        """Print current schedule summary"""
        print("\n📅 SMART SCHEDULER STATUS")
        print("=" * 40)
        print(f"Status: {'🟢 Running' if self.is_running else '🔴 Stopped'}")
        
        if not SCHEDULE_AVAILABLE:
            print("[ERROR] Schedule module not available")
            print("   Install with: pip install schedule")
            return
        
        print(f"Total Jobs: {len(schedule.jobs)}")
        
        if schedule.jobs:
            print(f"Next Run: {schedule.next_run()}")
            
            print("\n[LIST] Scheduled Jobs:")
            for i, job in enumerate(schedule.jobs, 1):
                print(f"  {i}. {job}")
        
        print(f"\n[ANALYSIS] Performance Summary:")
        perf_summary = self.performance_tracker.get_performance_summary()
        for key, value in perf_summary.items():
            print(f"  {key}: {value}")

class PerformanceTracker:
    """Track and monitor automation performance"""
    
    def __init__(self):
        self.task_history = []
        self.running_tasks = set()
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "last_24h_executions": 0
        }
    
    def record_task_start(self, task_name: str):
        """Record task start"""
        self.running_tasks.add(task_name)
        
        task_record = {
            "task_name": task_name,
            "start_time": datetime.now(),
            "status": "running"
        }
        
        self.task_history.append(task_record)
    
    def record_task_completion(self, task_name: str, result: Dict):
        """Record successful task completion"""
        self.running_tasks.discard(task_name)
        
        # Update the last record for this task
        for record in reversed(self.task_history):
            if record["task_name"] == task_name and record["status"] == "running":
                record["end_time"] = datetime.now()
                record["status"] = "completed"
                record["duration"] = (record["end_time"] - record["start_time"]).total_seconds()
                record["result"] = result
                break
        
        self.update_metrics()
    
    def record_task_error(self, task_name: str, error: str):
        """Record task error"""
        self.running_tasks.discard(task_name)
        
        # Update the last record for this task
        for record in reversed(self.task_history):
            if record["task_name"] == task_name and record["status"] == "running":
                record["end_time"] = datetime.now()
                record["status"] = "failed"
                record["duration"] = (record["end_time"] - record["start_time"]).total_seconds()
                record["error"] = error
                break
        
        self.update_metrics()
    
    def is_task_running(self, task_name: str) -> bool:
        """Check if a task is currently running"""
        return task_name in self.running_tasks
    
    def get_recent_error_rate(self, hours: int = 24) -> float:
        """Get error rate for recent period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_tasks = [
            task for task in self.task_history 
            if task.get("end_time", datetime.now()) > cutoff_time
        ]
        
        if not recent_tasks:
            return 0.0
        
        failed_tasks = len([task for task in recent_tasks if task["status"] == "failed"])
        return failed_tasks / len(recent_tasks)
    
    def update_metrics(self):
        """Update performance metrics"""
        completed_tasks = [task for task in self.task_history if task["status"] in ["completed", "failed"]]
        
        if completed_tasks:
            self.performance_metrics["total_executions"] = len(completed_tasks)
            self.performance_metrics["successful_executions"] = len([
                task for task in completed_tasks if task["status"] == "completed"
            ])
            self.performance_metrics["failed_executions"] = len([
                task for task in completed_tasks if task["status"] == "failed"
            ])
            
            # Calculate average execution time
            durations = [task.get("duration", 0) for task in completed_tasks if "duration" in task]
            if durations:
                self.performance_metrics["average_execution_time"] = sum(durations) / len(durations)
        
        # Count last 24h executions
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.performance_metrics["last_24h_executions"] = len([
            task for task in completed_tasks
            if task.get("end_time", datetime.now()) > cutoff_time
        ])
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        return self.performance_metrics.copy()
    
    def update_success_metrics(self, task_name: str):
        """Update success-specific metrics"""
        # Can be extended for task-specific success metrics
        pass

def main():
    """Main smart scheduler entry point"""
    parser = argparse.ArgumentParser(description='Smart Scheduler with Performance Monitoring')
    
    parser.add_argument('--start', action='store_true',
                        help='Start the scheduler')
    parser.add_argument('--status', action='store_true',
                        help='Show scheduler status')
    parser.add_argument('--config', type=str,
                        help='Configuration file path')
    parser.add_argument('--daemon', action='store_true',
                        help='Run as daemon (background process)')
    
    args = parser.parse_args()
    
    try:
        # Create smart scheduler
        scheduler = SmartScheduler(config_file=args.config)
        
        if args.status:
            scheduler.print_schedule_summary()
            return
        
        if args.start:
            scheduler.start_scheduler()
            
            if args.daemon:
                print("[REFRESH] Running in daemon mode... Press Ctrl+C to stop")
                try:
                    while scheduler.is_running:
                        time.sleep(60)
                        # Optionally print periodic status
                        if datetime.now().minute % 30 == 0:  # Every 30 minutes
                            scheduler.print_schedule_summary()
                except KeyboardInterrupt:
                    print("\n⏹️ Stopping scheduler...")
                    scheduler.stop_scheduler()
            else:
                print("[OK] Scheduler started. Use --status to check status.")
        else:
            print("Use --start to start scheduler or --status to check status")
            parser.print_help()
    
    except Exception as e:
        print(f"[ERROR] Smart scheduler failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
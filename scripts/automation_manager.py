#!/usr/bin/env python3
"""
NSE Stock Screener - Automation Manager
Handles batch processing, scheduling, and automated analysis
"""

import sys
import os
import time
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

# Optional imports with fallbacks
PSUTIL_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    pass

EMAIL_AVAILABLE = False
try:
    from email_notifier import EmailNotifier
    EMAIL_AVAILABLE = True
except ImportError:
    pass

class AutomationManager:
    """Manages automated stock screening operations"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize automation manager"""
        self.base_dir = PROJECT_ROOT
        self.output_dir = self.base_dir / 'output'
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize components
        self.email_notifier = None
        if EMAIL_AVAILABLE:
            try:
                self.email_notifier = EmailNotifier()
            except Exception as e:
                self.logger.warning(f"Email notifications not available: {e}")
        
        self.logger.info(f"Automation Manager started - Session: {self.session_id}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.base_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'automation_{self.session_id}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: Optional[str] = None) -> Dict:
        """Load automation configuration"""
        
        # Default configuration
        default_config = {
            "portfolio_capital": 100000.0,
            "max_positions": 10,
            "risk_per_trade": 0.01,
            "min_score_threshold": 65,
            "analysis_timeout": 300,  # 5 minutes
            "resource_checks": {
                "max_cpu_percent": 80,
                "max_memory_percent": 85,
                "min_disk_gb": 5
            },
            "automation_schedules": {
                "daily_analysis": ["09:30", "15:45"],
                "weekly_analysis": ["Sunday:20:00"],
                "monthly_analysis": ["Last_Friday:18:00"]
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    self.logger.info(f"[OK] Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.warning(f"[WARNING] Failed to load config {config_file}: {e}")
        
        return default_config
    
    def check_system_resources(self) -> Tuple[bool, str]:
        """Check if system resources are adequate for analysis"""
        
        if not PSUTIL_AVAILABLE:
            return True, "Resource monitoring not available (psutil not installed)"
        
        try:
            # Get resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Convert bytes to GB
            disk_free_gb = disk.free / (1024**3)
            
            # Check limits
            max_cpu = self.config["resource_checks"]["max_cpu_percent"]
            max_memory = self.config["resource_checks"]["max_memory_percent"]
            min_disk = self.config["resource_checks"]["min_disk_gb"]
            
            if cpu_percent > max_cpu:
                return False, f"CPU usage too high: {cpu_percent:.1f}% > {max_cpu}%"
            
            if memory.percent > max_memory:
                return False, f"Memory usage too high: {memory.percent:.1f}% > {max_memory}%"
            
            if disk_free_gb < min_disk:
                return False, f"Disk space too low: {disk_free_gb:.1f}GB < {min_disk}GB"
            
            self.logger.info(f"[OK] System resources OK - Memory: {memory.percent:.1f}%, CPU: {cpu_percent:.1f}%, Disk: {disk_free_gb:.1f}GB")
            return True, "System resources OK"
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error checking system resources: {e}")
            return True, f"Resource check failed: {e}"  # Default to allowing analysis
    
    def run_analysis_command(self, command: List[str], task_name: str, timeout: int = 300) -> Tuple[bool, str]:
        """Run analysis command with timeout and error handling"""
        
        self.logger.info(f"[RUNNING] Starting {task_name}...")
        
        try:
            # Run command with timeout
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.base_dir)
            )
            
            # Combine stdout and stderr for full output
            full_output = result.stdout
            if result.stderr:
                full_output += "\n" + result.stderr
            
            if result.returncode == 0:
                self.logger.info(f"[OK] {task_name} completed successfully")
                return True, full_output
            else:
                self.logger.error(f"[ERROR] {task_name} failed with code {result.returncode}")
                self.logger.error(f"Output: {full_output}")
                return False, full_output
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"[TIMEOUT] {task_name} timed out after {timeout} seconds")
            return False, f"Timeout after {timeout} seconds"
        except Exception as e:
            self.logger.error(f"[ERROR] {task_name} failed with exception: {e}")
            return False, str(e)
    
    def run_daily_automation(self) -> Dict:
        """Run daily automation tasks"""
        
        self.logger.info("[START] Starting daily automation...")
        
        start_time = datetime.now()
        results = {
            'session_id': self.session_id,
            'start_time': start_time.isoformat(),
            'tasks': {},
            'success': False
        }
        
        # Check system resources
        resource_ok, resource_msg = self.check_system_resources()
        if not resource_ok:
            self.logger.error(f"[ERROR] Insufficient system resources: {resource_msg}")
            results['error'] = resource_msg
            return results
        
        # Define daily tasks
        daily_tasks = [
            {
                'name': 'Daily Banking Analysis',
                'command': [
                    sys.executable, str(self.base_dir / 'cli_analyzer.py'), 
                    '--banking', '--min-score', '65'
                ]
            },
            {
                'name': 'Daily Technology Analysis', 
                'command': [
                    sys.executable, str(self.base_dir / 'cli_analyzer.py'),
                    '--technology', '--min-score', '60'
                ]
            }
        ]
        
        # Execute tasks
        overall_success = True
        for task in daily_tasks:
            task_name = task['name']
            task_command = task['command']
            
            success, output = self.run_analysis_command(
                task_command, 
                task_name, 
                timeout=self.config['analysis_timeout']
            )
            
            results['tasks'][task_name.lower().replace(' ', '_')] = {
                'success': success,
                'output': output
            }
            
            if not success:
                overall_success = False
        
        # Complete results
        end_time = datetime.now()
        duration = end_time - start_time
        
        results.update({
            'success': overall_success,
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds()
        })
        
        self.logger.info(f"[COMPLETED] Daily automation completed - Success: {results['success']}, Duration: {duration}")
        
        # Save results and send notifications if successful
        if overall_success:
            self.save_results(results)
            self.send_notification(results)
        
        return results
    
    def run_weekly_automation(self) -> Dict:
        """Run weekly automation tasks"""
        
        self.logger.info("[START] Starting weekly automation...")
        
        start_time = datetime.now()
        results = {
            'session_id': self.session_id,
            'start_time': start_time.isoformat(),
            'tasks': {},
            'success': False
        }
        
        # Check system resources
        resource_ok, resource_msg = self.check_system_resources()
        if not resource_ok:
            self.logger.error(f"[ERROR] Insufficient system resources: {resource_msg}")
            results['error'] = resource_msg
            return results
        
        # Define weekly tasks
        weekly_tasks = [
            {
                'name': 'Full NSE Analysis',
                'command': [
                    sys.executable, str(self.base_dir / 'run_screener.py'),
                    '--mode', 'full', '--min-score', '70'
                ]
            },
            {
                'name': 'Sector Analysis',
                'command': [
                    sys.executable, str(self.base_dir / 'cli_analyzer.py'),
                    '--technology', '--min-score', '65'
                ]
            },
            {
                'name': 'Portfolio Review',
                'command': [
                    sys.executable, 'scripts/generate_summary.py',
                    '--mode', 'weekly'
                ]
            }
        ]
        
        # Execute tasks
        overall_success = True
        for task in weekly_tasks:
            task_name = task['name']
            task_command = task['command']
            
            success, output = self.run_analysis_command(
                task_command,
                task_name,
                timeout=self.config['analysis_timeout'] * 3  # Longer timeout for weekly tasks
            )
            
            results['tasks'][task_name.lower().replace(' ', '_')] = {
                'success': success,
                'output': output
            }
            
            if not success:
                overall_success = False
        
        # Complete results
        end_time = datetime.now()
        duration = end_time - start_time
        
        results.update({
            'success': overall_success,
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds()
        })
        
        self.logger.info(f"[COMPLETED] Weekly automation completed - Success: {results['success']}, Duration: {duration}")
        
        # Save results and send notifications
        self.save_results(results)
        if overall_success:
            self.send_notification(results)
        
        return results
    
    def save_results(self, results: Dict):
        """Save automation results to file"""
        try:
            results_dir = self.output_dir / 'automation_results'
            results_dir.mkdir(exist_ok=True)
            
            result_file = results_dir / f'automation_{self.session_id}.json'
            
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Results saved to {result_file}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to save results: {e}")
    
    def send_notification(self, results: Dict):
        """Send notification about automation results"""
        try:
            if not self.email_notifier:
                self.logger.info("Email notifications not configured")
                return
            
            # Prepare notification content
            subject = f"NSE Automation Report - {results['session_id']}"
            
            # Create summary
            success_count = sum(1 for task in results['tasks'].values() if task['success'])
            total_count = len(results['tasks'])
            
            body = f"""
            NSE Stock Screener Automation Report
            =====================================
            
            Session: {results['session_id']}
            Start Time: {results['start_time']}
            End Time: {results['end_time']}
            Duration: {results.get('duration_seconds', 0):.1f} seconds
            
            Task Summary:
            - Total Tasks: {total_count}
            - Successful: {success_count}
            - Failed: {total_count - success_count}
            - Overall Success: {results['success']}
            
            Task Details:
            """
            
            for task_name, task_result in results['tasks'].items():
                status = "[OK]" if task_result['success'] else "[ERROR]"
                body += f"- {task_name}: {status}\n"
            
            # Send notification
            self.email_notifier.send_notification(subject, body)
            self.logger.info("Notification sent successfully")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to send notification: {e}")
    
    def generate_summary(self) -> Dict:
        """Generate automation summary"""
        try:
            # This would typically aggregate results from multiple runs
            summary = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'automation_status': 'active',
                'next_scheduled_run': 'Not configured',
                'recent_success_rate': 'Not calculated',
                'success': True
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to generate summary: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Main function for automation manager CLI"""
    parser = argparse.ArgumentParser(description='NSE Stock Screener Automation Manager')
    
    parser.add_argument('--mode', choices=['daily', 'weekly', 'summary'], 
                       default='daily', help='Automation mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    try:
        # Create automation manager
        manager = AutomationManager(config_file=args.config)
        
        if args.debug:
            manager.logger.setLevel(logging.DEBUG)
        
        # Run requested automation
        if args.mode == 'daily':
            result = manager.run_daily_automation()
        elif args.mode == 'weekly':
            result = manager.run_weekly_automation()
        elif args.mode == 'summary':
            result = manager.generate_summary()
        else:
            result = {'success': False, 'error': 'Unknown mode'}
        
        # Exit with appropriate code
        exit_code = 0 if result.get('success', False) else 1
        manager.logger.info(f"[COMPLETED] Automation completed with exit code: {exit_code}")
        return exit_code
        
    except Exception as e:
        if 'manager' in locals():
            manager.logger.error(f"[ERROR] Automation failed with exception: {e}")
        else:
            print(f"[ERROR] Failed to initialize automation manager: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
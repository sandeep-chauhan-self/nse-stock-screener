#!/usr/bin/env python3
"""
NSE Stock Screener - Enterprise Integration and Dashboard System
Web-based dashboard with enterprise integrations for automation monitoring
"""

import sys
import json
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import argparse
from dataclasses import dataclass, asdict
import logging

# Optional imports for enterprise features
try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("[WARNING] Flask not available. Install with: pip install flask")

try:
    import plotly.graph_objects as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARNING] Plotly not available. Install with: pip install plotly")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from automation_manager import AutomationManager
from smart_scheduler import SmartScheduler
from event_driven_automation import EventDrivenAutomation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure"""
    total_automations: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    active_schedules: int = 0
    monitored_events: int = 0
    last_run_time: Optional[str] = None
    system_health: str = "UNKNOWN"
    automation_efficiency: float = 0.0

class EnterpriseIntegration:
    """Enterprise integration and dashboard system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.project_root = PROJECT_ROOT
        self.config = self.load_config(config_file)
        
        # Initialize automation components
        self.automation_manager = AutomationManager()
        self.smart_scheduler = SmartScheduler()
        self.event_automation = EventDrivenAutomation()
        
        # Flask app for dashboard (only if Flask is available)
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self.setup_routes()
        else:
            self.app = None
            print("[WARNING] Flask not available - dashboard features disabled")
        
        # Metrics tracking
        self.metrics = DashboardMetrics()
        self.performance_history = []
        self.alert_history = []
        
        # Background threads
        self.metrics_thread = None
        self.is_running = False
    
    def load_config(self, config_file: Optional[str]) -> Dict:
        """Load enterprise integration configuration"""
        default_config = {
            "dashboard": {
                "host": "localhost",
                "port": 5000,
                "debug": False,
                "auto_refresh_interval": 30
            },
            "integrations": {
                "slack": {
                    "enabled": False,
                    "webhook_url": "",
                    "channels": ["#trading-alerts", "#automation-status"]
                },
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "recipients": ["admin@company.com"]
                },
                "database": {
                    "enabled": False,
                    "type": "postgresql",
                    "connection_string": ""
                },
                "api": {
                    "enabled": True,
                    "api_key_required": False,
                    "rate_limit": 100
                }
            },
            "alerts": {
                "automation_failure": True,
                "performance_degradation": True,
                "high_opportunity_alerts": True,
                "system_health_alerts": True
            },
            "reporting": {
                "daily_summary": True,
                "weekly_performance": True,
                "monthly_analytics": True,
                "custom_reports": True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.error(f"Config load error: {e}, using defaults")
        
        return default_config
    
    def setup_routes(self):
        """Set up Flask routes for the dashboard"""
        
        if not FLASK_AVAILABLE or not self.app:
            return
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template_string(self.get_dashboard_template())
        
        @self.app.route('/api/metrics')
        def api_metrics():
            """API endpoint for current metrics"""
            return jsonify(asdict(self.metrics))
        
        @self.app.route('/api/performance')
        def api_performance():
            """API endpoint for performance history"""
            return jsonify({
                "performance_history": self.performance_history[-100:],  # Last 100 data points
                "last_updated": datetime.now().isoformat()
            })
        
        @self.app.route('/api/alerts')
        def api_alerts():
            """API endpoint for alerts"""
            return jsonify({
                "alerts": self.alert_history[-50:],  # Last 50 alerts
                "total_alerts": len(self.alert_history)
            })
        
        @self.app.route('/api/automation/status')
        def api_automation_status():
            """API endpoint for automation status"""
            return jsonify({
                "scheduler_status": self.smart_scheduler.get_schedule_status(),
                "event_automation_status": self.event_automation.get_status(),
                "automation_manager_status": {
                    "last_run": getattr(self.automation_manager, 'last_run_time', None),
                    "total_runs": getattr(self.automation_manager, 'total_runs', 0)
                }
            })
        
        @self.app.route('/api/automation/trigger', methods=['POST'])
        def api_trigger_automation():
            """API endpoint to trigger automation manually"""
            try:
                automation_type = request.json.get('type', 'daily')
                result = self.trigger_manual_automation(automation_type)
                return jsonify({"success": True, "result": result})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500
        
        @self.app.route('/api/reports/generate', methods=['POST'])
        def api_generate_report():
            """API endpoint to generate custom reports"""
            try:
                report_type = request.json.get('type', 'performance')
                period = request.json.get('period', '7d')
                report = self.generate_custom_report(report_type, period)
                return jsonify({"success": True, "report": report})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500
    
    def get_dashboard_template(self) -> str:
        """Get HTML template for dashboard"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSE Stock Screener - Enterprise Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .metric-label {
            color: #666;
            text-transform: uppercase;
            font-size: 0.9em;
        }
        .status-indicators {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status-good { border-left: 4px solid #4caf50; }
        .status-warning { border-left: 4px solid #ff9800; }
        .status-error { border-left: 4px solid #f44336; }
        .charts-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .controls {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .log-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        .log-entry {
            padding: 8px;
            border-bottom: 1px solid #eee;
            font-family: monospace;
            font-size: 0.9em;
        }
        .log-info { color: #2196f3; }
        .log-success { color: #4caf50; }
        .log-warning { color: #ff9800; }
        .log-error { color: #f44336; }
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>[LAUNCH] NSE Stock Screener Enterprise Dashboard</h1>
            <p>Real-time automation monitoring and control center</p>
            <div id="last-updated">Last updated: <span id="update-time">Loading...</span></div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="total-automations">-</div>
                <div class="metric-label">Total Automations</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="success-rate">-</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="active-schedules">-</div>
                <div class="metric-label">Active Schedules</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="monitored-events">-</div>
                <div class="metric-label">Monitored Events</div>
            </div>
        </div>

        <div class="status-indicators">
            <div class="status-card status-good">
                <h3>🟢 System Health</h3>
                <p id="system-health">Monitoring...</p>
            </div>
            <div class="status-card status-good">
                <h3>[ANALYSIS] Automation Efficiency</h3>
                <p id="automation-efficiency">Calculating...</p>
            </div>
            <div class="status-card status-good">
                <h3>[TIMEOUT] Last Run</h3>
                <p id="last-run">Checking...</p>
            </div>
            <div class="status-card status-good">
                <h3>[TARGET] Recent Alerts</h3>
                <p id="recent-alerts">Loading...</p>
            </div>
        </div>

        <div class="controls">
            <h3>[CONTROL] Automation Controls</h3>
            <button class="btn" onclick="triggerAutomation('daily')">Run Daily Automation</button>
            <button class="btn" onclick="triggerAutomation('market_scan')">Market Scan</button>
            <button class="btn" onclick="triggerAutomation('sector_analysis')">Sector Analysis</button>
            <button class="btn" onclick="generateReport('performance')">Generate Performance Report</button>
        </div>

        <div class="charts-section">
            <h3>[CHART] Performance Analytics</h3>
            <div id="performance-chart" style="height: 400px;"></div>
        </div>

        <div class="log-section">
            <h3>[LIST] Activity Log</h3>
            <div id="activity-log">
                <div class="log-entry log-info">[INFO] Dashboard initialized</div>
                <div class="log-entry log-success">[SUCCESS] Automation system connected</div>
                <div class="log-entry log-info">[INFO] Monitoring real-time metrics...</div>
            </div>
        </div>
    </div>

    <script>
        // Dashboard JavaScript functionality
        let updateInterval;

        async function fetchMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const metrics = await response.json();
                updateMetricsDisplay(metrics);
            } catch (error) {
                console.error('Error fetching metrics:', error);
            }
        }

        function updateMetricsDisplay(metrics) {
            document.getElementById('total-automations').textContent = metrics.total_automations;
            document.getElementById('active-schedules').textContent = metrics.active_schedules;
            document.getElementById('monitored-events').textContent = metrics.monitored_events;
            
            const successRate = metrics.total_automations > 0 
                ? ((metrics.successful_runs / metrics.total_automations) * 100).toFixed(1) + '%'
                : '0%';
            document.getElementById('success-rate').textContent = successRate;
            
            document.getElementById('system-health').textContent = metrics.system_health;
            document.getElementById('automation-efficiency').textContent = 
                (metrics.automation_efficiency * 100).toFixed(1) + '%';
            document.getElementById('last-run').textContent = 
                metrics.last_run_time || 'Never';
            
            document.getElementById('update-time').textContent = new Date().toLocaleTimeString();
        }

        async function triggerAutomation(type) {
            try {
                const response = await fetch('/api/automation/trigger', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({type: type})
                });
                const result = await response.json();
                
                if (result.success) {
                    addLogEntry(`[SUCCESS] ${type} automation triggered successfully`, 'success');
                } else {
                    addLogEntry(`[ERROR] Failed to trigger ${type}: ${result.error}`, 'error');
                }
            } catch (error) {
                addLogEntry(`[ERROR] Automation trigger failed: ${error}`, 'error');
            }
        }

        async function generateReport(type) {
            try {
                const response = await fetch('/api/reports/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({type: type, period: '7d'})
                });
                const result = await response.json();
                
                if (result.success) {
                    addLogEntry(`[SUCCESS] ${type} report generated`, 'success');
                } else {
                    addLogEntry(`[ERROR] Report generation failed: ${result.error}`, 'error');
                }
            } catch (error) {
                addLogEntry(`[ERROR] Report generation failed: ${error}`, 'error');
            }
        }

        function addLogEntry(message, type = 'info') {
            const logContainer = document.getElementById('activity-log');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${type}`;
            logEntry.textContent = `[${timestamp}] ${message}`;
            
            logContainer.insertBefore(logEntry, logContainer.firstChild);
            
            // Keep only last 20 entries
            while (logContainer.children.length > 20) {
                logContainer.removeChild(logContainer.lastChild);
            }
        }

        async function updatePerformanceChart() {
            try {
                const response = await fetch('/api/performance');
                const data = await response.json();
                
                const trace = {
                    x: data.performance_history.map(p => p.timestamp),
                    y: data.performance_history.map(p => p.efficiency),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Automation Efficiency',
                    line: {color: '#667eea'}
                };
                
                const layout = {
                    title: 'Automation Performance Over Time',
                    xaxis: {title: 'Time'},
                    yaxis: {title: 'Efficiency %'},
                    showlegend: false
                };
                
                Plotly.newPlot('performance-chart', [trace], layout);
            } catch (error) {
                console.error('Error updating performance chart:', error);
            }
        }

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            fetchMetrics();
            updatePerformanceChart();
            
            // Set up auto-refresh
            updateInterval = setInterval(() => {
                fetchMetrics();
                updatePerformanceChart();
            }, 30000); // Update every 30 seconds
            
            addLogEntry('[INFO] Real-time dashboard initialized', 'info');
        });

        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
        });
    </script>
</body>
</html>
        """
    
    def start_dashboard(self):
        """Start the dashboard server"""
        if not FLASK_AVAILABLE or not self.app:
            print("[ERROR] Dashboard cannot start - Flask not available")
            print("   Install Flask with: pip install flask")
            return
            
        print("[NETWORK] Starting Enterprise Dashboard...")
        
        # Start metrics collection thread
        self.is_running = True
        self.metrics_thread = threading.Thread(target=self.collect_metrics_loop, daemon=True)
        self.metrics_thread.start()
        
        # Start Flask app
        host = self.config["dashboard"]["host"]
        port = self.config["dashboard"]["port"]
        debug = self.config["dashboard"]["debug"]
        
        print(f"[OK] Dashboard available at: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)
    
    def collect_metrics_loop(self):
        """Background thread to collect metrics"""
        while self.is_running:
            try:
                self.collect_current_metrics()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def collect_current_metrics(self):
        """Collect current system metrics"""
        try:
            # Get scheduler status
            scheduler_status = self.smart_scheduler.get_schedule_status()
            
            # Get event automation status
            event_status = self.event_automation.get_status()
            
            # Update metrics
            self.metrics.active_schedules = scheduler_status.get("total_jobs", 0)
            self.metrics.monitored_events = event_status.get("events_processed", 0)
            
            # Calculate automation efficiency (simplified)
            total_runs = scheduler_status.get("performance_summary", {}).get("total_executions", 0)
            successful_runs = scheduler_status.get("performance_summary", {}).get("successful_executions", 0)
            
            self.metrics.total_automations = total_runs
            self.metrics.successful_runs = successful_runs
            self.metrics.failed_runs = total_runs - successful_runs
            
            if total_runs > 0:
                self.metrics.automation_efficiency = successful_runs / total_runs
            
            # System health assessment
            if self.metrics.automation_efficiency >= 0.95:
                self.metrics.system_health = "EXCELLENT"
            elif self.metrics.automation_efficiency >= 0.85:
                self.metrics.system_health = "GOOD"
            elif self.metrics.automation_efficiency >= 0.70:
                self.metrics.system_health = "WARNING"
            else:
                self.metrics.system_health = "CRITICAL"
            
            self.metrics.last_run_time = datetime.now().isoformat()
            
            # Add to performance history
            self.performance_history.append({
                "timestamp": datetime.now().isoformat(),
                "efficiency": self.metrics.automation_efficiency * 100,
                "total_runs": total_runs,
                "successful_runs": successful_runs
            })
            
            # Keep only last 1000 entries
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
        
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def trigger_manual_automation(self, automation_type: str) -> Dict:
        """Trigger automation manually from dashboard"""
        try:
            if automation_type == "daily":
                result = self.automation_manager.run_daily_automation()
            elif automation_type == "market_scan":
                result = {"type": "market_scan", "status": "completed", "timestamp": datetime.now().isoformat()}
            elif automation_type == "sector_analysis":
                result = {"type": "sector_analysis", "status": "completed", "timestamp": datetime.now().isoformat()}
            else:
                raise ValueError(f"Unknown automation type: {automation_type}")
            
            # Add to alert history
            self.alert_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "manual_trigger",
                "message": f"Manual {automation_type} automation triggered",
                "severity": "INFO"
            })
            
            return result
        
        except Exception as e:
            logger.error(f"Manual automation trigger error: {e}")
            raise
    
    def generate_custom_report(self, report_type: str, period: str) -> Dict:
        """Generate custom reports"""
        try:
            end_date = datetime.now()
            
            if period == "1d":
                start_date = end_date - timedelta(days=1)
            elif period == "7d":
                start_date = end_date - timedelta(days=7)
            elif period == "30d":
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=7)
            
            if report_type == "performance":
                # Filter performance data for period
                period_data = [
                    p for p in self.performance_history
                    if datetime.fromisoformat(p["timestamp"]) >= start_date
                ]
                
                return {
                    "report_type": report_type,
                    "period": period,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "data_points": len(period_data),
                    "average_efficiency": sum(p["efficiency"] for p in period_data) / len(period_data) if period_data else 0,
                    "total_runs": sum(p["total_runs"] for p in period_data),
                    "successful_runs": sum(p["successful_runs"] for p in period_data),
                    "generated_at": datetime.now().isoformat()
                }
            else:
                return {
                    "report_type": report_type,
                    "period": period,
                    "message": f"Report type {report_type} not implemented yet",
                    "generated_at": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            raise
    
    def stop_dashboard(self):
        """Stop the dashboard and background processes"""
        print("⏹️ Stopping Enterprise Dashboard...")
        self.is_running = False
        
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)

def main():
    """Main entry point for enterprise dashboard"""
    parser = argparse.ArgumentParser(description='Enterprise Integration Dashboard')
    
    parser.add_argument('--start', action='store_true',
                        help='Start the dashboard server')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for dashboard server')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host for dashboard server')
    parser.add_argument('--config', type=str,
                        help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Create enterprise integration
        integration = EnterpriseIntegration(config_file=args.config)
        
        # Override config with command line args
        if args.port:
            integration.config["dashboard"]["port"] = args.port
        if args.host:
            integration.config["dashboard"]["host"] = args.host
        
        if args.start:
            integration.start_dashboard()
        else:
            print("Use --start to launch the dashboard")
            parser.print_help()
    
    except Exception as e:
        print(f"[ERROR] Enterprise dashboard failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
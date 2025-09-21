#!/usr/bin/env python3
"""
NSE Stock Screener - Email Notification System
Sends analysis results via email
"""

import sys
import json
import smtplib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Try importing email modules with fallback
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    from email.mime.base import MimeBase
    from email import encoders
    EMAIL_AVAILABLE = True
except ImportError as e:
    print(f"Email modules not available: {e}")
    EMAIL_AVAILABLE = False
    # Create dummy classes for testing
    class MimeText:
        def __init__(self, *args, **kwargs): pass
    class MimeMultipart:
        def __init__(self, *args, **kwargs): pass
        def attach(self, *args): pass
        def as_string(self): return ""
        def __setitem__(self, key, value): pass
    class MimeBase:
        def __init__(self, *args, **kwargs): pass
        def set_payload(self, *args): pass
        def add_header(self, *args, **kwargs): pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

class EmailNotifier:
    """Email notification system for analysis results"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.project_root = PROJECT_ROOT
        self.config_file = config_file or self.project_root / 'config' / 'email_config.json'
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load email configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    return json.load(f)
            else:
                return self._create_default_config()
        except Exception as e:
            print(f"⚠️ Config load error: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """Create default email configuration"""
        default_config = {
            "enabled": False,
            "smtp": {
                "server": "smtp.gmail.com",
                "port": 587,
                "username": "your_email@gmail.com",
                "password": "your_app_password",
                "use_tls": True
            },
            "recipients": [
                "recipient1@example.com",
                "recipient2@example.com"
            ],
            "sender": {
                "name": "NSE Stock Screener",
                "email": "your_email@gmail.com"
            },
            "templates": {
                "daily_subject": "NSE Daily Analysis - {date}",
                "weekly_subject": "NSE Weekly Analysis - Week ending {date}",
                "error_subject": "NSE Analysis Error - {date}"
            }
        }
        
        # Save default config
        self.config_file.parent.mkdir(exist_ok=True)
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"📁 Created default email config: {self.config_file}")
            print("⚠️ Please update email credentials and enable notifications")
        except Exception as e:
            print(f"❌ Failed to save default config: {e}")
        
        return default_config
    
    def is_enabled(self) -> bool:
        """Check if email notifications are enabled"""
        return EMAIL_AVAILABLE and self.config.get("enabled", False)
    
    def create_daily_email(self, summary: Dict, attachments: List[Path] = None) -> MimeMultipart:
        """Create daily analysis email"""
        msg = MimeMultipart()
        msg['From'] = f"{self.config['sender']['name']} <{self.config['sender']['email']}>"
        msg['To'] = ", ".join(self.config['recipients'])
        msg['Subject'] = self.config['templates']['daily_subject'].format(
            date=summary.get('date', datetime.now().strftime('%Y-%m-%d'))
        )
        
        # Email body
        body = self._create_daily_body(summary)
        msg.attach(MimeText(body, 'html'))
        
        # Add attachments
        if attachments:
            for file_path in attachments:
                if file_path.exists():
                    with open(file_path, 'rb') as attachment:
                        part = MimeBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {file_path.name}'
                        )
                        msg.attach(part)
        
        return msg
    
    def create_weekly_email(self, summary: Dict, attachments: List[Path] = None) -> MimeMultipart:
        """Create weekly analysis email"""
        msg = MimeMultipart()
        msg['From'] = f"{self.config['sender']['name']} <{self.config['sender']['email']}>"
        msg['To'] = ", ".join(self.config['recipients'])
        msg['Subject'] = self.config['templates']['weekly_subject'].format(
            date=summary.get('week_ending', datetime.now().strftime('%Y-%m-%d'))
        )
        
        # Email body
        body = self._create_weekly_body(summary)
        msg.attach(MimeText(body, 'html'))
        
        # Add attachments
        if attachments:
            for file_path in attachments:
                if file_path.exists():
                    with open(file_path, 'rb') as attachment:
                        part = MimeBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {file_path.name}'
                        )
                        msg.attach(part)
        
        return msg
    
    def create_error_email(self, error_details: Dict) -> MimeMultipart:
        """Create error notification email"""
        msg = MimeMultipart()
        msg['From'] = f"{self.config['sender']['name']} <{self.config['sender']['email']}>"
        msg['To'] = ", ".join(self.config['recipients'])
        msg['Subject'] = self.config['templates']['error_subject'].format(
            date=datetime.now().strftime('%Y-%m-%d %H:%M')
        )
        
        # Email body
        body = self._create_error_body(error_details)
        msg.attach(MimeText(body, 'html'))
        
        return msg
    
    def _create_daily_body(self, summary: Dict) -> str:
        """Create HTML body for daily email"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .summary {{ background-color: #ecf0f1; padding: 15px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: white; border-left: 4px solid #3498db; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .alert {{ background-color: #e74c3c; color: white; padding: 10px; margin: 10px 0; }}
                .success {{ background-color: #27ae60; color: white; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 NSE Daily Analysis Report</h1>
                <p>{summary.get('date', 'N/A')}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">
                    <div class="metric-value">{summary['totals']['stocks_analyzed']}</div>
                    <div class="metric-label">Stocks Analyzed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['totals']['entry_ready']}</div>
                    <div class="metric-label">Entry Ready</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['totals']['avg_score']}</div>
                    <div class="metric-label">Average Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['totals']['high_score_stocks']}</div>
                    <div class="metric-label">High Score (70+)</div>
                </div>
            </div>
            
            {"<div class='alert'>⚠️ No analysis results found for today</div>" if summary['report_count'] == 0 else ""}
            {"<div class='success'>✅ Analysis completed successfully</div>" if summary['report_count'] > 0 else ""}
        """
        
        if summary['reports']:
            html += """
            <h2>Recent Reports</h2>
            <table>
                <tr>
                    <th>Report</th>
                    <th>Stocks</th>
                    <th>Entry Ready</th>
                    <th>Avg Score</th>
                    <th>High Score</th>
                </tr>
            """
            for report in summary['reports'][-5:]:  # Last 5 reports
                html += f"""
                <tr>
                    <td>{report['file']}</td>
                    <td>{report['total_stocks']}</td>
                    <td>{report['entry_ready']}</td>
                    <td>{report['avg_score']}</td>
                    <td>{report['high_score_count']}</td>
                </tr>
                """
            html += "</table>"
        
        html += """
            <p style="margin-top: 30px; font-size: 12px; color: #7f8c8d;">
                This is an automated report from NSE Stock Screener.<br>
                Generated at {timestamp}
            </p>
        </body>
        </html>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        return html
    
    def _create_weekly_body(self, summary: Dict) -> str:
        """Create HTML body for weekly email"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #8e44ad; color: white; padding: 20px; text-align: center; }}
                .summary {{ background-color: #ecf0f1; padding: 15px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: white; border-left: 4px solid #9b59b6; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #9b59b6; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📅 NSE Weekly Analysis Report</h1>
                <p>Week ending {summary.get('week_ending', 'N/A')}</p>
            </div>
            
            <div class="summary">
                <h2>Weekly Summary</h2>
                <div class="metric">
                    <div class="metric-value">{summary['weekly_totals']['stocks_analyzed']}</div>
                    <div class="metric-label">Total Stocks</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['weekly_totals']['entry_ready']}</div>
                    <div class="metric-label">Entry Ready</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['weekly_totals']['avg_score']}</div>
                    <div class="metric-label">Average Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{summary['weekly_totals']['high_score_stocks']}</div>
                    <div class="metric-label">High Score (70+)</div>
                </div>
            </div>
        """
        
        if summary['daily_summaries']:
            html += """
            <h2>Daily Breakdown</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Reports</th>
                    <th>Stocks</th>
                    <th>Entry Ready</th>
                    <th>Avg Score</th>
                </tr>
            """
            for day in summary['daily_summaries']:
                html += f"""
                <tr>
                    <td>{day['date']}</td>
                    <td>{day['reports']}</td>
                    <td>{day['stocks_analyzed']}</td>
                    <td>{day['entry_ready']}</td>
                    <td>{day['avg_score']}</td>
                </tr>
                """
            html += "</table>"
        
        html += """
            <p style="margin-top: 30px; font-size: 12px; color: #7f8c8d;">
                This is an automated weekly report from NSE Stock Screener.<br>
                Generated at {timestamp}
            </p>
        </body>
        </html>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        return html
    
    def _create_error_body(self, error_details: Dict) -> str:
        """Create HTML body for error email"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #e74c3c; color: white; padding: 20px; text-align: center; }}
                .error {{ background-color: #fadbd8; padding: 15px; margin: 20px 0; border-left: 4px solid #e74c3c; }}
                .details {{ background-color: #f8f9fa; padding: 15px; margin: 20px 0; }}
                code {{ background-color: #f1f2f6; padding: 2px 4px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 NSE Analysis Error</h1>
                <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="error">
                <h2>Error Details</h2>
                <p><strong>Error:</strong> {error_details.get('error', 'Unknown error')}</p>
                <p><strong>Session:</strong> {error_details.get('session_id', 'N/A')}</p>
                <p><strong>Mode:</strong> {error_details.get('mode', 'N/A')}</p>
            </div>
            
            <div class="details">
                <h3>System Information</h3>
                <p><strong>Timestamp:</strong> {error_details.get('timestamp', 'N/A')}</p>
                <p><strong>Duration:</strong> {error_details.get('duration', 'N/A')}</p>
            </div>
            
            <p style="margin-top: 30px; font-size: 12px; color: #7f8c8d;">
                This is an automated error notification from NSE Stock Screener.<br>
                Please check the system logs for more details.
            </p>
        </body>
        </html>
        """
        
        return html
    
    def send_email(self, msg: MimeMultipart) -> bool:
        """Send email message"""
        if not self.is_enabled():
            print("📧 Email notifications disabled")
            return False
        
        try:
            # Create SMTP session
            server = smtplib.SMTP(self.config['smtp']['server'], self.config['smtp']['port'])
            
            if self.config['smtp']['use_tls']:
                server.starttls()
            
            # Login
            server.login(self.config['smtp']['username'], self.config['smtp']['password'])
            
            # Send email
            text = msg.as_string()
            server.sendmail(self.config['sender']['email'], self.config['recipients'], text)
            server.quit()
            
            print(f"✅ Email sent successfully to {len(self.config['recipients'])} recipients")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False
    
    def send_daily_notification(self, summary: Dict, attachments: List[Path] = None):
        """Send daily analysis notification"""
        msg = self.create_daily_email(summary, attachments)
        return self.send_email(msg)
    
    def send_weekly_notification(self, summary: Dict, attachments: List[Path] = None):
        """Send weekly analysis notification"""
        msg = self.create_weekly_email(summary, attachments)
        return self.send_email(msg)
    
    def send_error_notification(self, error_details: Dict):
        """Send error notification"""
        msg = self.create_error_email(error_details)
        return self.send_email(msg)

def main():
    """Test email notification system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NSE Stock Screener - Email Notifier')
    parser.add_argument('--test', action='store_true',
                        help='Send test email')
    parser.add_argument('--config', action='store_true',
                        help='Show configuration status')
    
    args = parser.parse_args()
    
    notifier = EmailNotifier()
    
    if args.config:
        print(f"📧 Email Configuration Status:")
        print(f"   Config file: {notifier.config_file}")
        print(f"   Enabled: {notifier.is_enabled()}")
        print(f"   Recipients: {len(notifier.config.get('recipients', []))}")
        print(f"   SMTP Server: {notifier.config.get('smtp', {}).get('server', 'Not configured')}")
        return
    
    if args.test:
        # Create test summary
        test_summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "report_count": 2,
            "totals": {
                "stocks_analyzed": 150,
                "entry_ready": 12,
                "avg_score": 58.3,
                "high_score_stocks": 8
            },
            "reports": [
                {
                    "file": "test_report_1.csv",
                    "total_stocks": 75,
                    "entry_ready": 6,
                    "avg_score": 60.1,
                    "high_score_count": 4
                },
                {
                    "file": "test_report_2.csv",
                    "total_stocks": 75,
                    "entry_ready": 6,
                    "avg_score": 56.5,
                    "high_score_count": 4
                }
            ]
        }
        
        print("📧 Sending test email...")
        success = notifier.send_daily_notification(test_summary)
        if success:
            print("✅ Test email sent successfully")
        else:
            print("❌ Failed to send test email")
    else:
        print("Use --test to send test email or --config to check configuration")

if __name__ == "__main__":
    main()
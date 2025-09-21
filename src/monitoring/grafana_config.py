"""
Grafana Dashboard Configuration for NSE Stock Screener

This module provides pre-configured Grafana dashboards, alerting rules,
and monitoring infrastructure for comprehensive observability.

Key Features:
- Pre-built dashboard configurations in JSON format
- Automated dashboard provisioning
- Alert rule definitions with thresholds
- Performance monitoring panels
- Business metrics visualization
- System health monitoring
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from ..logging_config import get_logger
except ImportError:
    def get_logger(name): return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class GrafanaConfig:
    """Configuration for Grafana dashboard setup."""
    grafana_url: str = "http://localhost:3000"
    api_key: Optional[str] = None
    org_id: int = 1
    dashboard_dir: str = "monitoring/dashboards"
    alert_dir: str = "monitoring/alerts"
    datasource_name: str = "prometheus"
    datasource_url: str = "http://localhost:9090"


class DashboardBuilder:
    """Builder for creating Grafana dashboard configurations."""

    def __init__(self, config: GrafanaConfig):
        self.config = config
        self.dashboard_template = {
            "dashboard": {
                "id": None,
                "title": "",
                "tags": ["nse-screener"],
                "timezone": "browser",
                "panels": [],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s",
                "schemaVersion": 30,
                "version": 1,
                "editable": True,
                "gnetId": None,
                "graphTooltip": 0,
                "links": []
            },
            "folderId": 0,
            "overwrite": True
        }

    def create_main_dashboard(self) -> Dict[str, Any]:
        """Create the main NSE Stock Screener dashboard."""
        dashboard = self._create_base_dashboard(
            title="NSE Stock Screener - Main Dashboard",
            description="Main dashboard for NSE Stock Screener monitoring"
        )

        panels = []
        panel_id = 1

        # System Overview Row
        panels.append(self._create_row_panel("System Overview", panel_id))
        panel_id += 1

        # Request Rate Panel
        panels.append(self._create_stat_panel(
            title="Request Rate",
            targets=[{
                "expr": "rate(nse_screener_requests_total[5m])",
                "legendFormat": "{{operation}}"
            }],
            unit="reqps",
            position={"x": 0, "y": 1, "w": 6, "h": 8},
            panel_id=panel_id
        ))
        panel_id += 1

        # Error Rate Panel
        panels.append(self._create_stat_panel(
            title="Error Rate",
            targets=[{
                "expr": "rate(nse_screener_errors_total[5m]) / rate(nse_screener_requests_total[5m]) * 100",
                "legendFormat": "Error Rate %"
            }],
            unit="percent",
            position={"x": 6, "y": 1, "w": 6, "h": 8},
            panel_id=panel_id,
            thresholds=[
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 5}
            ]
        ))
        panel_id += 1

        # Response Time Panel
        panels.append(self._create_graph_panel(
            title="Response Time",
            targets=[
                {
                    "expr": "histogram_quantile(0.50, rate(nse_screener_processing_duration_seconds_bucket[5m]))",
                    "legendFormat": "50th percentile"
                },
                {
                    "expr": "histogram_quantile(0.95, rate(nse_screener_processing_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile"
                },
                {
                    "expr": "histogram_quantile(0.99, rate(nse_screener_processing_duration_seconds_bucket[5m]))",
                    "legendFormat": "99th percentile"
                }
            ],
            unit="s",
            position={"x": 12, "y": 1, "w": 12, "h": 8},
            panel_id=panel_id
        ))
        panel_id += 1

        # Cache Performance Row
        panels.append(self._create_row_panel("Cache Performance", panel_id))
        panel_id += 1

        # Cache Hit Rate Panel
        panels.append(self._create_stat_panel(
            title="Cache Hit Rate",
            targets=[{
                "expr": "nse_screener_cache_hits_total / (nse_screener_cache_hits_total + nse_screener_cache_misses_total) * 100",
                "legendFormat": "Hit Rate %"
            }],
            unit="percent",
            position={"x": 0, "y": 10, "w": 8, "h": 8},
            panel_id=panel_id,
            thresholds=[
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 70},
                {"color": "green", "value": 90}
            ]
        ))
        panel_id += 1

        # Cache Operations Panel
        panels.append(self._create_graph_panel(
            title="Cache Operations",
            targets=[
                {
                    "expr": "rate(nse_screener_cache_hits_total[5m])",
                    "legendFormat": "Cache Hits/sec"
                },
                {
                    "expr": "rate(nse_screener_cache_misses_total[5m])",
                    "legendFormat": "Cache Misses/sec"
                }
            ],
            unit="ops",
            position={"x": 8, "y": 10, "w": 16, "h": 8},
            panel_id=panel_id
        ))
        panel_id += 1

        # System Resources Row
        panels.append(self._create_row_panel("System Resources", panel_id))
        panel_id += 1

        # CPU Usage Panel
        panels.append(self._create_graph_panel(
            title="CPU Usage",
            targets=[{
                "expr": "nse_screener_cpu_usage_percent",
                "legendFormat": "CPU Usage %"
            }],
            unit="percent",
            position={"x": 0, "y": 19, "w": 8, "h": 8},
            panel_id=panel_id,
            y_axis={"min": 0, "max": 100}
        ))
        panel_id += 1

        # Memory Usage Panel
        panels.append(self._create_graph_panel(
            title="Memory Usage",
            targets=[
                {
                    "expr": "nse_screener_system_memory_usage_percent",
                    "legendFormat": "System Memory %"
                },
                {
                    "expr": "nse_screener_memory_usage_bytes / 1024 / 1024",
                    "legendFormat": "Application Memory MB"
                }
            ],
            unit="percent",
            position={"x": 8, "y": 19, "w": 8, "h": 8},
            panel_id=panel_id
        ))
        panel_id += 1

        # Disk Usage Panel
        panels.append(self._create_stat_panel(
            title="Disk Usage",
            targets=[{
                "expr": "nse_screener_disk_usage_percent",
                "legendFormat": "Disk Usage %"
            }],
            unit="percent",
            position={"x": 16, "y": 19, "w": 8, "h": 8},
            panel_id=panel_id,
            thresholds=[
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 80},
                {"color": "red", "value": 90}
            ]
        ))
        panel_id += 1

        # Business Metrics Row
        panels.append(self._create_row_panel("Business Metrics", panel_id))
        panel_id += 1

        # Symbols Processed Panel
        panels.append(self._create_stat_panel(
            title="Symbols Processed",
            targets=[{
                "expr": "rate(nse_screener_symbols_processed_total[5m]) * 60",
                "legendFormat": "Symbols/min"
            }],
            unit="short",
            position={"x": 0, "y": 28, "w": 6, "h": 8},
            panel_id=panel_id
        ))
        panel_id += 1

        # Analysis Signals Panel
        panels.append(self._create_graph_panel(
            title="Analysis Signals",
            targets=[{
                "expr": "rate(nse_screener_analysis_signals_total[5m])",
                "legendFormat": "{{signal_type}}"
            }],
            unit="short",
            position={"x": 6, "y": 28, "w": 12, "h": 8},
            panel_id=panel_id
        ))
        panel_id += 1

        # Portfolio Value Panel
        panels.append(self._create_stat_panel(
            title="Portfolio Value",
            targets=[{
                "expr": "nse_screener_portfolio_value",
                "legendFormat": "Portfolio Value"
            }],
            unit="currencyINR",
            position={"x": 18, "y": 28, "w": 6, "h": 8},
            panel_id=panel_id
        ))
        panel_id += 1

        dashboard["dashboard"]["panels"] = panels
        return dashboard

    def create_performance_dashboard(self) -> Dict[str, Any]:
        """Create performance-focused dashboard."""
        dashboard = self._create_base_dashboard(
            title="NSE Stock Screener - Performance Analysis",
            description="Detailed performance analysis and optimization metrics"
        )

        panels = []
        panel_id = 1

        # Performance Overview
        panels.append(self._create_row_panel("Performance Overview", panel_id))
        panel_id += 1

        # Latency Heatmap
        panels.append(self._create_heatmap_panel(
            title="Response Time Heatmap",
            targets=[{
                "expr": "increase(nse_screener_processing_duration_seconds_bucket[5m])",
                "legendFormat": "{{le}}"
            }],
            position={"x": 0, "y": 1, "w": 24, "h": 8},
            panel_id=panel_id
        ))
        panel_id += 1

        # Throughput Analysis
        panels.append(self._create_graph_panel(
            title="Throughput by Operation",
            targets=[{
                "expr": "rate(nse_screener_requests_total[5m])",
                "legendFormat": "{{operation}}"
            }],
            unit="reqps",
            position={"x": 0, "y": 10, "w": 12, "h": 8},
            panel_id=panel_id
        ))
        panel_id += 1

        # Error Analysis
        panels.append(self._create_graph_panel(
            title="Error Rate by Operation",
            targets=[{
                "expr": "rate(nse_screener_errors_total[5m])",
                "legendFormat": "{{operation}} - {{error_type}}"
            }],
            unit="short",
            position={"x": 12, "y": 10, "w": 12, "h": 8},
            panel_id=panel_id
        ))
        panel_id += 1

        dashboard["dashboard"]["panels"] = panels
        return dashboard

    def _create_base_dashboard(self, title: str, description: str) -> Dict[str, Any]:
        """Create base dashboard structure."""
        dashboard = json.loads(json.dumps(self.dashboard_template))
        dashboard["dashboard"]["title"] = title
        dashboard["dashboard"]["description"] = description
        dashboard["dashboard"]["uid"] = title.lower().replace(" ", "-").replace("nse-stock-screener-", "")
        return dashboard

    def _create_row_panel(self, title: str, panel_id: int) -> Dict[str, Any]:
        """Create a row panel for organizing dashboard sections."""
        return {
            "collapsed": False,
            "datasource": None,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0},
            "id": panel_id,
            "panels": [],
            "title": title,
            "type": "row"
        }

    def _create_stat_panel(self, title: str, targets: List[Dict], unit: str,
                          position: Dict[str, int], panel_id: int,
                          thresholds: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Create a stat panel."""
        panel = {
            "id": panel_id,
            "title": title,
            "type": "stat",
            "targets": [self._format_target(target) for target in targets],
            "gridPos": {
                "h": position["h"],
                "w": position["w"],
                "x": position["x"],
                "y": position["y"]
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds or [
                            {"color": "green", "value": None}
                        ]
                    }
                }
            },
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto"
            }
        }
        return panel

    def _create_graph_panel(self, title: str, targets: List[Dict], unit: str,
                           position: Dict[str, int], panel_id: int,
                           y_axis: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a time series graph panel."""
        panel = {
            "id": panel_id,
            "title": title,
            "type": "timeseries",
            "targets": [self._format_target(target) for target in targets],
            "gridPos": {
                "h": position["h"],
                "w": position["w"],
                "x": position["x"],
                "y": position["y"]
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {"mode": "none", "group": "A"},
                        "axisPlacement": "auto",
                        "axisLabel": "",
                        "scaleDistribution": {"type": "linear"}
                    }
                }
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {
                    "displayMode": "list",
                    "placement": "bottom"
                }
            }
        }

        if y_axis:
            panel["fieldConfig"]["defaults"]["min"] = y_axis.get("min")
            panel["fieldConfig"]["defaults"]["max"] = y_axis.get("max")

        return panel

    def _create_heatmap_panel(self, title: str, targets: List[Dict],
                             position: Dict[str, int], panel_id: int) -> Dict[str, Any]:
        """Create a heatmap panel."""
        return {
            "id": panel_id,
            "title": title,
            "type": "heatmap",
            "targets": [self._format_target(target) for target in targets],
            "gridPos": {
                "h": position["h"],
                "w": position["w"],
                "x": position["x"],
                "y": position["y"]
            },
            "options": {
                "calculate": False,
                "yAxis": {
                    "unit": "s"
                },
                "cellGap": 2,
                "cellRadius": 0,
                "color": {
                    "mode": "spectrum",
                    "scheme": "Spectral",
                    "steps": 128
                }
            }
        }

    def _format_target(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Format Prometheus target query."""
        return {
            "expr": target["expr"],
            "legendFormat": target.get("legendFormat", ""),
            "refId": target.get("refId", "A"),
            "datasource": {
                "type": "prometheus",
                "uid": self.config.datasource_name
            }
        }


class AlertRuleBuilder:
    """Builder for creating Grafana alert rules."""

    def __init__(self, config: GrafanaConfig):
        self.config = config

    def create_alert_rules(self) -> List[Dict[str, Any]]:
        """Create all alert rules for NSE Stock Screener."""
        rules = []

        # High Error Rate Alert
        rules.append(self._create_alert_rule(
            name="High Error Rate",
            condition="rate(nse_screener_errors_total[5m]) / rate(nse_screener_requests_total[5m]) > 0.05",
            description="Error rate exceeds 5%",
            severity="critical",
            for_duration="2m"
        ))

        # High Response Time Alert
        rules.append(self._create_alert_rule(
            name="High Response Time",
            condition="histogram_quantile(0.95, rate(nse_screener_processing_duration_seconds_bucket[5m])) > 10",
            description="95th percentile response time exceeds 10 seconds",
            severity="warning",
            for_duration="5m"
        ))

        # Low Cache Hit Rate Alert
        rules.append(self._create_alert_rule(
            name="Low Cache Hit Rate",
            condition="nse_screener_cache_hits_total / (nse_screener_cache_hits_total + nse_screener_cache_misses_total) < 0.7",
            description="Cache hit rate below 70%",
            severity="warning",
            for_duration="10m"
        ))

        # High CPU Usage Alert
        rules.append(self._create_alert_rule(
            name="High CPU Usage",
            condition="nse_screener_cpu_usage_percent > 90",
            description="CPU usage exceeds 90%",
            severity="critical",
            for_duration="5m"
        ))

        # High Memory Usage Alert
        rules.append(self._create_alert_rule(
            name="High Memory Usage",
            condition="nse_screener_system_memory_usage_percent > 85",
            description="Memory usage exceeds 85%",
            severity="warning",
            for_duration="5m"
        ))

        # Disk Space Alert
        rules.append(self._create_alert_rule(
            name="Low Disk Space",
            condition="nse_screener_disk_usage_percent > 90",
            description="Disk usage exceeds 90%",
            severity="critical",
            for_duration="1m"
        ))

        # Data Freshness Alert
        rules.append(self._create_alert_rule(
            name="Stale Data",
            condition="nse_screener_data_freshness_seconds > 1800",
            description="Data is older than 30 minutes",
            severity="warning",
            for_duration="5m"
        ))

        return rules

    def _create_alert_rule(self, name: str, condition: str, description: str,
                          severity: str, for_duration: str) -> Dict[str, Any]:
        """Create individual alert rule."""
        return {
            "alert": name,
            "expr": condition,
            "for": for_duration,
            "labels": {
                "severity": severity,
                "service": "nse-stock-screener"
            },
            "annotations": {
                "summary": name,
                "description": description,
                "runbook_url": "https://github.com/your-org/nse-stock-screener/wiki/alerts"
            }
        }


class GrafanaProvisioner:
    """Manages Grafana dashboard and alert provisioning."""

    def __init__(self, config: GrafanaConfig):
        self.config = config
        self.dashboard_builder = DashboardBuilder(config)
        self.alert_builder = AlertRuleBuilder(config)

        # Create directories
        Path(config.dashboard_dir).mkdir(parents=True, exist_ok=True)
        Path(config.alert_dir).mkdir(parents=True, exist_ok=True)

    def provision_dashboards(self):
        """Create and save all dashboard configurations."""
        dashboards = [
            ("main-dashboard.json", self.dashboard_builder.create_main_dashboard()),
            ("performance-dashboard.json", self.dashboard_builder.create_performance_dashboard())
        ]

        for filename, dashboard_config in dashboards:
            dashboard_path = Path(self.config.dashboard_dir) / filename
            with open(dashboard_path, 'w') as f:
                json.dump(dashboard_config, f, indent=2)

            logger.info(f"Dashboard configuration saved: {dashboard_path}")

    def provision_alerts(self):
        """Create and save alert rule configurations."""
        alert_rules = self.alert_builder.create_alert_rules()

        alert_config = {
            "groups": [
                {
                    "name": "nse-stock-screener",
                    "rules": alert_rules
                }
            ]
        }

        alert_path = Path(self.config.alert_dir) / "alert-rules.yml"

        # Convert to YAML format
        import yaml
        with open(alert_path, 'w') as f:
            yaml.dump(alert_config, f, default_flow_style=False)

        logger.info(f"Alert rules saved: {alert_path}")

    def create_datasource_config(self) -> Dict[str, Any]:
        """Create Prometheus datasource configuration."""
        return {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": self.config.datasource_name,
                    "type": "prometheus",
                    "url": self.config.datasource_url,
                    "access": "proxy",
                    "isDefault": True,
                    "jsonData": {
                        "timeInterval": "30s"
                    }
                }
            ]
        }

    def provision_all(self):
        """Provision all Grafana configurations."""
        logger.info("Starting Grafana provisioning...")

        # Create dashboards
        self.provision_dashboards()

        # Create alert rules
        self.provision_alerts()

        # Create datasource config
        datasource_config = self.create_datasource_config()
        datasource_path = Path(self.config.dashboard_dir) / "datasources.yml"

        import yaml
        with open(datasource_path, 'w') as f:
            yaml.dump(datasource_config, f, default_flow_style=False)

        logger.info(f"Datasource configuration saved: {datasource_path}")
        logger.info("Grafana provisioning completed")

    def create_docker_compose(self) -> str:
        """Create Docker Compose configuration for monitoring stack."""
        return """version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: nse-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alerts:/etc/prometheus/rules
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'

  grafana:
    image: grafana/grafana:latest
    container_name: nse-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false

  redis:
    image: redis:alpine
    container_name: nse-redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data

volumes:
  grafana-storage:
  redis-data:
"""

    def create_prometheus_config(self) -> str:
        """Create Prometheus configuration."""
        return """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  - job_name: 'nse-stock-screener'
    static_configs:
      - targets: ['host.docker.internal:8000']
    scrape_interval: 30s
    metrics_path: /metrics

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""


def main():
    """Main function to provision all monitoring configurations."""
    config = GrafanaConfig()
    provisioner = GrafanaProvisioner(config)

    # Create all configurations
    provisioner.provision_all()

    # Create Docker Compose file
    docker_compose = provisioner.create_docker_compose()
    with open("docker-compose.monitoring.yml", 'w') as f:
        f.write(docker_compose)

    # Create Prometheus config
    prometheus_config = provisioner.create_prometheus_config()
    prometheus_dir = Path("monitoring")
    prometheus_dir.mkdir(exist_ok=True)
    with open(prometheus_dir / "prometheus.yml", 'w') as f:
        f.write(prometheus_config)

    print("Monitoring stack configurations created successfully!")
    print("\nTo start the monitoring stack:")
    print("1. docker-compose -f docker-compose.monitoring.yml up -d")
    print("2. Access Grafana at http://localhost:3000 (admin:admin123)")
    print("3. Access Prometheus at http://localhost:9090")


if __name__ == "__main__":
    main()
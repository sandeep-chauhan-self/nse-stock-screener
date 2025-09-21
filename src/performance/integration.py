"""
FS.8 Performance, Scaling & Operationalization Integration
This module integrates all performance and scaling components to provide
a complete solution for scaling the NSE Stock Screener to full NSE universe
with production-level reliability and monitoring.
Components Integrated:
- Parallel processing engine for concurrent symbol analysis
- Comprehensive profiling and optimization framework
- Prometheus metrics and monitoring system
- Production-ready caching with Redis support
- Grafana dashboards and alerting infrastructure
Usage:
    from src.performance.integration import initialize_performance_stack

    # Initialize complete performance stack
    performance_stack = initialize_performance_stack()

    # Process symbols at scale
    results = performance_stack.process_symbols_parallel(symbol_list)
"""
import logging
import asyncio
from pathlib import Path
from typing import Dict[str, Any], List[str], Any, Optional
from dataclasses import dataclass

# Import all performance components
from .parallel_engine import ParallelStockProcessor, ParallelConfig
from .profiling_engine import PerformanceProfiler, ProfilingConfig
from .caching_system import initialize_cache, CacheConfig
from ..monitoring.prometheus_metrics import initialize_monitoring, MetricConfig
from ..monitoring.grafana_config import GrafanaProvisioner, GrafanaConfig
try:
    from ..logging_config import get_logger
except ImportError:
    def get_logger(name): return logging.getLogger(name)
logger = get_logger(__name__)
@dataclass
class PerformanceStackConfig:
    """Complete configuration for performance stack."""

    # Parallel processing settings
    max_workers: int = 8
    batch_size: int = 50
    enable_async_io: bool = True

    # Caching settings
    cache_backend: str = "memory"
  # "memory", "redis", "disk"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600

    # Monitoring settings
    prometheus_port: int = 8000
    enable_grafana: bool = True
    grafana_port: int = 3000

    # Profiling settings
    enable_profiling: bool = True
    profile_sample_rate: float = 0.1
  # Profile 10% of operations
    # Optimization settings
    enable_auto_optimization: bool = True
    optimization_interval: int = 3600
  # 1 hour
class PerformanceStack:
    """Complete performance and scaling stack for NSE Stock Screener."""
    def __init__(self, config: Optional[PerformanceStackConfig] = None) -> None:
        self.config = config or PerformanceStackConfig()

        # Initialize components
        self.parallel_processor = None
        self.profiler = None
        self.cache = None
        self.monitoring = None
        self.grafana_provisioner = None

        # Performance metrics
        self.metrics = {
            "total_symbols_processed": 0,
            "average_processing_time": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0
        }
        logger.info("Performance stack initialized")
    def initialize_all_components(self):
        """Initialize all performance stack components."""
        logger.info("Initializing performance stack components...")

        # 1. Initialize caching system
        self._initialize_caching()

        # 2. Initialize monitoring
        self._initialize_monitoring()

        # 3. Initialize parallel processing
        self._initialize_parallel_processing()

        # 4. Initialize profiling
        self._initialize_profiling()

        # 5. Setup Grafana dashboards
        self._initialize_grafana()
        logger.info("All performance stack components initialized successfully")
    def _initialize_caching(self):
        """Initialize the caching system."""
        cache_config = CacheConfig(
            backend=self.config.cache_backend,
            redis_url=self.config.redis_url,
            default_ttl=self.config.cache_ttl,
            enable_metrics=True
        )
        self.cache = initialize_cache(cache_config)
        logger.info(f"Caching system initialized with {self.config.cache_backend} backend")
    def _initialize_monitoring(self):
        """Initialize Prometheus monitoring."""
        metric_config = MetricConfig(
            prometheus_port=self.config.prometheus_port,
            export_system_metrics=True
        )
        self.monitoring = initialize_monitoring(metric_config)
        logger.info(f"Monitoring system initialized on port {self.config.prometheus_port}")
    def _initialize_parallel_processing(self):
        """Initialize parallel processing engine."""
        parallel_config = ParallelConfig(
            max_workers=self.config.max_workers,
            batch_size=self.config.batch_size,
            enable_async_io=self.config.enable_async_io
        )
        self.parallel_processor = ParallelStockProcessor(parallel_config)
        logger.info(f"Parallel processing initialized with {self.config.max_workers} workers")
    def _initialize_profiling(self):
        """Initialize profiling system."""
        if not self.config.enable_profiling:
            logger.info("Profiling disabled in configuration")
            return
        profiling_config = ProfilingConfig(
            enable_cprofile=True,
            enable_memory_profiling=True,
            enable_line_profiling=False,
  # Expensive, use sparingly
            cache_size=1000,
            sample_rate=self.config.profile_sample_rate
        )
        self.profiler = PerformanceProfiler(profiling_config)
        logger.info("Profiling system initialized")
    def _initialize_grafana(self):
        """Initialize Grafana dashboard provisioning."""
        if not self.config.enable_grafana:
            logger.info("Grafana disabled in configuration")
            return
        grafana_config = GrafanaConfig(
            grafana_url=f"http://localhost:{self.config.grafana_port}",
            datasource_url=f"http://localhost:{self.config.prometheus_port}"
        )
        self.grafana_provisioner = GrafanaProvisioner(grafana_config)
        self.grafana_provisioner.provision_all()
        logger.info("Grafana dashboards provisioned")
    def process_symbols_parallel(self, symbols: List[str],
                                     analysis_func: Optional[callable] = None) -> Dict[str, Any]:
        """Process symbols in parallel with full performance monitoring."""
        if not self.parallel_processor:
            raise RuntimeError("Parallel processor not initialized")
        logger.info(f"Starting parallel processing of {len(symbols)} symbols")

        # Start profiling if enabled
        profile_context = None
        if self.profiler:
            profile_context = self.profiler.profile_function_calls("symbol_analysis_batch")
            profile_context.__enter__()
        try:

            # Use monitoring timer
            with self.monitoring.metrics.timer("batch_symbol_processing"):

                # Process symbols using parallel engine
                raw_results = self.parallel_processor.process_symbol_universe(
                    symbols, analysis_func
                )

                # Convert Dict[str, Any] results to List[str] format expected by metrics
                results = List[str](raw_results.values()) if isinstance($1, Dict[str, Any]) else raw_results

                # Update metrics
                self._update_performance_metrics(results)

                # Record business metrics
                successful_count = len([r for r in results if isinstance($1, Dict[str, Any])])
  # Count valid results
                self.monitoring.metrics.symbols_processed.labels(status="success").inc(successful_count)
                failed_count = len(results) - successful_count
                if failed_count > 0:
                    self.monitoring.metrics.symbols_processed.labels(status="error").inc(failed_count)
                logger.info(f"Processed {successful_count}/{len(symbols)} symbols successfully")
                return {
                    "results": raw_results,
  # Return original Dict[str, Any] format
                    "total_symbols": len(symbols),
                    "successful": successful_count,
                    "failed": failed_count,
                    "cache_hit_rate": self._calculate_cache_hit_rate(),
                    "processing_time": sum(r.get("processing_time", 0) for r in results if isinstance($1, Dict[str, Any]))
                }
        finally:

            # Stop profiling
            if profile_context:
                profile_context.__exit__(None, None, None)
    def process_symbols_sync(self, symbols: List[str],
                           analysis_func: Optional[callable] = None) -> Dict[str, Any]:
        """Synchronous wrapper for parallel symbol processing."""
        return self.process_symbols_parallel(symbols, analysis_func)
    def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization analysis and recommendations."""
        if not self.profiler:
            return {"error": "Profiling not enabled"}
        logger.info("Running performance optimization analysis...")

        # Get profiling insights
        optimization_results = self.profiler.get_optimization_recommendations()

        # Get cache performance
        cache_stats = self.cache.get_stats() if self.cache else {}

        # Get system metrics
        system_status = self.monitoring.get_system_status() if self.monitoring else {}
        recommendations = []

        # Analyze cache performance
        if cache_stats.get("analytics", {}).get("recent_hit_rate", 0) < 0.8:
            recommendations.append({
                "type": "cache_optimization",
                "priority": "high",
                "description": "Cache hit rate is below 80%. Consider increasing cache size or improving cache key strategies.",
                "current_hit_rate": cache_stats.get("analytics", {}).get("recent_hit_rate", 0)
            })

        # Analyze system resources
        health_status = system_status.get("system_health", {})
        if health_status.get("overall_status") != "healthy":
            recommendations.append({
                "type": "system_resources",
                "priority": "critical",
                "description": "System health issues detected. Check CPU, memory, and disk usage.",
                "health_details": health_status
            })

        # Add profiling recommendations
        if optimization_results.get("recommendations"):
            recommendations.extend(optimization_results["recommendations"])
        return {
            "optimization_results": optimization_results,
            "cache_performance": cache_stats,
            "system_status": system_status,
            "recommendations": recommendations,
            "timestamp": self.monitoring.get_system_status()["timestamp"] if self.monitoring else None
        }
    def _update_performance_metrics(self, results: List[Dict[str, Any]]):
        """Update internal performance metrics."""
        if not results:
            return
        self.metrics["total_symbols_processed"] += len(results)

        # Calculate average processing time
        processing_times = [r.get("processing_time", 0) for r in results if r.get("processing_time")]
        if processing_times:
            self.metrics["average_processing_time"] = sum(processing_times) / len(processing_times)

        # Calculate error rate
        errors = sum(1 for r in results if r.get("status") != "success")
        self.metrics["error_rate"] = errors / len(results) if results else 0
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate current cache hit rate."""
        if not self.cache:
            return 0.0
        stats = self.cache.get_stats()
        return stats.get("analytics", {}).get("recent_hit_rate", 0.0)
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "metrics": self.metrics,
            "cache_stats": self.cache.get_stats() if self.cache else {},
            "system_status": self.monitoring.get_system_status() if self.monitoring else {},
            "parallel_config": {
                "max_workers": self.config.max_workers,
                "batch_size": self.config.batch_size,
                "async_io_enabled": self.config.enable_async_io
            },
            "profiling_enabled": self.config.enable_profiling,
            "monitoring_enabled": self.monitoring is not None
        }
    def shutdown(self):
        """Gracefully shutdown all performance stack components."""
        logger.info("Shutting down performance stack...")
        if self.parallel_processor:

            # The parallel processor will clean up automatically
            pass
        if self.cache:
            self.cache.stop_background_cleanup()
        if self.monitoring:
            self.monitoring.stop_monitoring()
        logger.info("Performance stack shutdown complete")
def initialize_performance_stack(config: Optional[PerformanceStackConfig] = None) -> PerformanceStack:
    """Initialize the complete performance stack."""
    stack = PerformanceStack(config)
    stack.initialize_all_components()
    return stack
def create_deployment_package():
    """Create deployment package with Docker configurations."""
    logger.info("Creating deployment package...")

    # Create monitoring directory structure
    monitoring_dir = Path("monitoring")
    monitoring_dir.mkdir(exist_ok=True)

    # Initialize Grafana provisioner to create configs
    config = GrafanaConfig()
    provisioner = GrafanaProvisioner(config)
    provisioner.provision_all()

    # Create Docker Compose file
    docker_compose = provisioner.create_docker_compose()
    with open("docker-compose.monitoring.yml", 'w') as f:
        f.write(docker_compose)

    # Create Prometheus config
    prometheus_config = provisioner.create_prometheus_config()
    with open(monitoring_dir / "prometheus.yml", 'w') as f:
        f.write(prometheus_config)

    # Create startup script
    startup_script = """#!/bin/bash

# NSE Stock Screener Performance Stack Startup Script
echo "Starting NSE Stock Screener Performance Stack..."

# Start monitoring stack
echo "Starting monitoring services..."
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check if services are running
echo "Checking service status..."
docker-compose -f docker-compose.monitoring.yml ps
echo "Performance stack started successfully!"
echo "- Grafana: http://localhost:3000 (admin:admin123)"
echo "- Prometheus: http://localhost:9090"
echo "- Redis: localhost:6379"
echo ""
echo "Start your NSE Stock Screener application to begin monitoring."
"""
    with open("start_monitoring.sh", 'w') as f:
        f.write(startup_script)

    # Make script executable
    import os
    os.chmod("start_monitoring.sh", 0o755)

    # Create README for deployment
    readme_content = """# NSE Stock Screener Performance Stack
This deployment package includes all components for scaling the NSE Stock Screener
to handle the full NSE universe with production-level reliability.

## Components Included
1. **Parallel Processing Engine**: Concurrent symbol analysis with async I/O
2. **Intelligent Caching**: Redis-backed caching with multiple strategies
3. **Monitoring & Metrics**: Prometheus metrics with comprehensive observability
4. **Grafana Dashboards**: Pre-configured dashboards for performance monitoring
5. **Profiling System**: Automated performance profiling and optimization

## Quick Start
1. Install Docker and Docker Compose
2. Run the startup script: `./start_monitoring.sh`
3. Import and use the performance stack in your application:
```python
from src.performance.integration import initialize_performance_stack

# Initialize performance stack
stack = initialize_performance_stack()

# Process symbols at scale
results = stack.process_symbols_sync(symbol_list, your_analysis_function)

# Get performance insights
optimization = stack.optimize_performance()
print(optimization["recommendations"])
```

## Monitoring Access
- **Grafana Dashboard**: http://localhost:3000 (admin:admin123)
- **Prometheus Metrics**: http://localhost:9090
- **Application Metrics**: http://localhost:8000/metrics

## Configuration
Customize the performance stack by modifying `PerformanceStackConfig`:
```python
config = PerformanceStackConfig(
    max_workers=16,
          # Increase for more parallelism
    cache_backend="redis",
   # Use Redis for distributed caching
    enable_profiling=True,
   # Enable performance profiling
    prometheus_port=8000
     # Metrics endpoint port
)
stack = initialize_performance_stack(config)
```

## Scaling Recommendations
- **Memory**: 8GB+ recommended for full NSE universe (5000+ symbols)
- **CPU**: 8+ cores for optimal parallel processing
- **Redis**: Use Redis for distributed caching in production
- **Monitoring**: Keep Grafana dashboards open during heavy processing

## Troubleshooting
1. Check service logs: `docker-compose -f docker-compose.monitoring.yml logs`
2. Verify metrics endpoint: `curl http://localhost:8000/metrics`
3. Monitor system resources in Grafana dashboards
4. Review optimization recommendations: `stack.optimize_performance()`
For more information, see the project documentation.
"""
    with open("PERFORMANCE_DEPLOYMENT.md", 'w') as f:
        f.write(readme_content)
    logger.info("Deployment package created successfully!")
    print("\nDeployment package created with the following files:")
    print("- docker-compose.monitoring.yml (monitoring stack)")
    print("- monitoring/ (Grafana dashboards and Prometheus config)")
    print("- start_monitoring.sh (startup script)")
    print("- PERFORMANCE_DEPLOYMENT.md (deployment guide)")
    print("\nRun './start_monitoring.sh' to start the monitoring stack")
if __name__ == "__main__":

    # Example usage and testing
    import time
    print("Testing performance stack integration...")

    # Initialize with test configuration
    config = PerformanceStackConfig(
        max_workers=4,
        cache_backend="memory",
        enable_profiling=True,
        prometheus_port=8001
  # Use different port for testing
    )
    stack = initialize_performance_stack(config)

    # Test with sample symbols
    test_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

    # Mock analysis function
    def mock_analysis(symbol):
        time.sleep(0.1)
  # Simulate analysis time
        return {
            "symbol": symbol,
            "signal": "BUY" if hash(symbol) % 2 else "HOLD",
            "confidence": 0.8,
            "timestamp": time.time()
        }

    # Process symbols
    print(f"Processing {len(test_symbols)} symbols...")
    results = stack.process_symbols_sync(test_symbols, mock_analysis)
    print(f"Processed {results['successful']}/{results['total_symbols']} symbols")
    print(f"Cache hit rate: {results['cache_hit_rate']:.2%}")

    # Get performance summary
    summary = stack.get_performance_summary()
    print(f"Average processing time: {summary['metrics']['average_processing_time']:.3f}s")

    # Get optimization recommendations
    optimization = stack.optimize_performance()
    print(f"Optimization recommendations: {len(optimization['recommendations'])}")

    # Create deployment package
    create_deployment_package()

    # Cleanup
    stack.shutdown()
    print("Performance stack test completed successfully!")

"""
Parameter Persistence System for Scoring Configurations

This module implements database storage for scoring configurations with metadata,
run tracking, and configuration hashing as required by FS.4.

Features:
- Store scoring configurations with versioning
- Track scoring runs with performance metrics
- Configuration hash-based deduplication
- Metadata and audit trail storage
- Query configurations by performance criteria

Usage:
    from src.scoring import ParameterStore, ScoringConfig

    # Initialize store
    store = ParameterStore("sqlite:///scoring_configs.db")

    # Save configuration
    config_id = store.save_config(config, metadata={"created_by": "user1"})

    # Track scoring run
    store.track_run(config_id, results, performance_metrics)
"""

import sqlite3
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import asdict
from contextlib import contextmanager

from .scoring_schema import ScoringConfig
from .scoring_engine import ScoringResult

logger = logging.getLogger(__name__)


class ParameterStore:
    """
    Database store for scoring configurations and run tracking.

    Supports SQLite for local development and can be extended for
    PostgreSQL/MySQL for production deployments.
    """

    def __init__(self, db_path: str = "scoring_parameters.db", create_tables: bool = True):
        """
        Initialize parameter store.

        Args:
            db_path: Database connection string or file path for SQLite
            create_tables: Whether to create tables if they don't exist
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        if create_tables:
            self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables for storing configurations and runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scoring_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_hash TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    created_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config_json TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    is_active BOOLEAN DEFAULT 1,
                    performance_score REAL DEFAULT 0.0
                );
            """)

            # Scoring runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scoring_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_id INTEGER NOT NULL,
                    run_hash TEXT UNIQUE NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    symbol_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    avg_processing_time_ms REAL DEFAULT 0.0,
                    avg_confidence REAL DEFAULT 0.0,
                    metadata_json TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'running',
                    FOREIGN KEY (config_id) REFERENCES scoring_configs (id)
                );
            """)

            # Individual symbol results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS symbol_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    total_score REAL NOT NULL,
                    probability_level TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    processing_time_ms REAL DEFAULT 0.0,
                    component_scores_json TEXT DEFAULT '{}',
                    bonus_penalty_json TEXT DEFAULT '{}',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES scoring_runs (id)
                );
            """)

            # Performance metrics table for tracking config effectiveness
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_id INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    measurement_date DATE NOT NULL,
                    measurement_period TEXT DEFAULT 'daily',
                    metadata_json TEXT DEFAULT '{}',
                    UNIQUE(config_id, metric_name, measurement_date),
                    FOREIGN KEY (config_id) REFERENCES scoring_configs (id)
                );
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_configs_hash ON scoring_configs(config_hash);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_configs_active ON scoring_configs(is_active);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_config ON scoring_runs(config_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON scoring_runs(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_run ON symbol_results(run_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_symbol ON symbol_results(symbol);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_config ON config_performance(config_id);")

            conn.commit()
            self.logger.info("Database tables created successfully")

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def save_config(self,
                   config: ScoringConfig,
                   metadata: Optional[Dict[str, Any]] = None,
                   replace_existing: bool = False) -> int:
        """
        Save scoring configuration to database.

        Args:
            config: ScoringConfig to save
            metadata: Additional metadata for the configuration
            replace_existing: Whether to replace existing config with same hash

        Returns:
            Configuration ID
        """
        config_hash = config.get_config_hash()
        config_json = json.dumps(self._config_to_dict(config), indent=2)
        metadata_json = json.dumps(metadata or {})

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if config already exists
            cursor.execute(
                "SELECT id FROM scoring_configs WHERE config_hash = ?",
                (config_hash,)
            )
            existing = cursor.fetchone()

            if existing and not replace_existing:
                self.logger.info(f"Configuration with hash {config_hash} already exists (ID: {existing['id']})")
                return existing['id']

            if existing and replace_existing:
                # Update existing configuration
                cursor.execute("""
                    UPDATE scoring_configs
                    SET name = ?, version = ?, description = ?, updated_at = ?,
                        config_json = ?, metadata_json = ?
                    WHERE config_hash = ?
                """, (
                    config.name, config.version, config.description,
                    datetime.now(), config_json, metadata_json, config_hash
                ))
                config_id = existing['id']
                self.logger.info(f"Updated existing configuration (ID: {config_id})")
            else:
                # Insert new configuration
                cursor.execute("""
                    INSERT INTO scoring_configs
                    (config_hash, name, version, description, created_by, config_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    config_hash, config.name, config.version, config.description,
                    config.created_by, config_json, metadata_json
                ))
                config_id = cursor.lastrowid
                self.logger.info(f"Saved new configuration (ID: {config_id})")

            conn.commit()
            return config_id

    def get_config(self, config_id: int) -> Optional[Tuple[ScoringConfig, Dict[str, Any]]]:
        """
        Retrieve scoring configuration by ID.

        Args:
            config_id: Configuration ID

        Returns:
            Tuple of (ScoringConfig, metadata) or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT config_json, metadata_json FROM scoring_configs WHERE id = ?",
                (config_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            try:
                config_dict = json.loads(row['config_json'])
                metadata = json.loads(row['metadata_json'])

                # Reconstruct ScoringConfig (simplified - would need full parser)
                # For now, return the dict - in production would reconstruct full object
                return config_dict, metadata

            except Exception as e:
                self.logger.error(f"Error deserializing config {config_id}: {e}")
                return None

    def start_run(self,
                 config_id: int,
                 metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Start a new scoring run.

        Args:
            config_id: Configuration ID to use for this run
            metadata: Additional run metadata

        Returns:
            Run ID
        """
        run_hash = self._generate_run_hash(config_id, datetime.now())
        metadata_json = json.dumps(metadata or {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scoring_runs
                (config_id, run_hash, metadata_json, status)
                VALUES (?, ?, ?, 'running')
            """, (config_id, run_hash, metadata_json))

            run_id = cursor.lastrowid
            conn.commit()

            self.logger.info(f"Started scoring run (ID: {run_id}) with config {config_id}")
            return run_id

    def record_symbol_result(self, run_id: int, result: ScoringResult) -> None:
        """
        Record individual symbol scoring result.

        Args:
            run_id: Run ID
            result: ScoringResult for the symbol
        """
        component_scores = {
            cs.name: {
                'raw_score': cs.raw_score,
                'weighted_score': cs.weighted_score,
                'confidence': cs.confidence
            }
            for cs in result.component_scores
        }

        bonus_penalty = {
            bp.rule_name: {
                'applied': bp.applied,
                'value': bp.value,
                'condition_met': bp.condition_met
            }
            for bp in result.bonus_penalty_results
        }

        metadata = {
            'normalized_score': result.normalized_score,
            'market_regime': result.market_regime,
            'regime_adjustments_applied': result.regime_adjustments_applied,
            'indicators_used': result.indicators_used,
            'indicators_missing': result.indicators_missing,
            'validation_warnings': result.validation_warnings
        }

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO symbol_results
                (run_id, symbol, total_score, probability_level, confidence,
                 processing_time_ms, component_scores_json, bonus_penalty_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, result.symbol, result.total_score, result.probability_level,
                result.confidence, result.processing_time_ms,
                json.dumps(component_scores), json.dumps(bonus_penalty),
                json.dumps(metadata)
            ))
            conn.commit()

    def complete_run(self,
                    run_id: int,
                    symbol_count: int,
                    success_count: int,
                    error_count: int,
                    avg_processing_time: float,
                    avg_confidence: float) -> None:
        """
        Mark a scoring run as completed with summary statistics.

        Args:
            run_id: Run ID
            symbol_count: Total symbols processed
            success_count: Successfully processed symbols
            error_count: Symbols with errors
            avg_processing_time: Average processing time in ms
            avg_confidence: Average confidence score
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE scoring_runs
                SET completed_at = ?, symbol_count = ?, success_count = ?,
                    error_count = ?, avg_processing_time_ms = ?, avg_confidence = ?,
                    status = 'completed'
                WHERE id = ?
            """, (
                datetime.now(), symbol_count, success_count, error_count,
                avg_processing_time, avg_confidence, run_id
            ))
            conn.commit()

            self.logger.info(f"Completed run {run_id}: {success_count}/{symbol_count} successful")

    def get_run_performance(self, run_id: int) -> Dict[str, Any]:
        """
        Get performance statistics for a completed run.

        Args:
            run_id: Run ID

        Returns:
            Dictionary with performance metrics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get run summary
            cursor.execute("""
                SELECT r.*, c.name as config_name, c.version as config_version
                FROM scoring_runs r
                JOIN scoring_configs c ON r.config_id = c.id
                WHERE r.id = ?
            """, (run_id,))
            run_info = cursor.fetchone()

            if not run_info:
                return {}

            # Get score distribution
            cursor.execute("""
                SELECT
                    probability_level,
                    COUNT(*) as count,
                    AVG(total_score) as avg_score,
                    AVG(confidence) as avg_confidence
                FROM symbol_results
                WHERE run_id = ?
                GROUP BY probability_level
            """, (run_id,))
            score_distribution = {
                row['probability_level']: {
                    'count': row['count'],
                    'avg_score': row['avg_score'],
                    'avg_confidence': row['avg_confidence']
                }
                for row in cursor.fetchall()
            }

            # Get top performers
            cursor.execute("""
                SELECT symbol, total_score, probability_level, confidence
                FROM symbol_results
                WHERE run_id = ?
                ORDER BY total_score DESC
                LIMIT 10
            """, (run_id,))
            top_performers = [dict(row) for row in cursor.fetchall()]

            return {
                'run_info': dict(run_info),
                'score_distribution': score_distribution,
                'top_performers': top_performers
            }

    def track_config_performance(self,
                               config_id: int,
                               metric_name: str,
                               metric_value: float,
                               measurement_date: Optional[datetime] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track performance metrics for a configuration.

        Args:
            config_id: Configuration ID
            metric_name: Name of the metric (e.g., 'sharpe_ratio', 'win_rate')
            metric_value: Metric value
            measurement_date: Date of measurement (defaults to today)
            metadata: Additional metric metadata
        """
        measurement_date = measurement_date or datetime.now().date()
        metadata_json = json.dumps(metadata or {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO config_performance
                (config_id, metric_name, metric_value, measurement_date, metadata_json)
                VALUES (?, ?, ?, ?, ?)
            """, (config_id, metric_name, metric_value, measurement_date, metadata_json))
            conn.commit()

            self.logger.info(f"Tracked performance metric {metric_name}={metric_value} for config {config_id}")

    def get_best_configs(self,
                        metric_name: str,
                        limit: int = 10,
                        min_runs: int = 3,
                        days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Get best performing configurations based on a metric.

        Args:
            metric_name: Performance metric to rank by
            limit: Maximum number of configs to return
            min_runs: Minimum number of runs required
            days_back: Look back period in days

        Returns:
            List of configurations ranked by performance
        """
        cutoff_date = datetime.now().date() - timedelta(days=days_back)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    c.id, c.name, c.version, c.description,
                    p.metric_value,
                    COUNT(r.id) as run_count,
                    AVG(r.avg_confidence) as avg_confidence,
                    MAX(r.completed_at) as last_run
                FROM scoring_configs c
                JOIN config_performance p ON c.id = p.config_id
                LEFT JOIN scoring_runs r ON c.id = r.config_id
                    AND r.status = 'completed'
                    AND r.completed_at >= ?
                WHERE p.metric_name = ?
                    AND c.is_active = 1
                GROUP BY c.id
                HAVING run_count >= ?
                ORDER BY p.metric_value DESC
                LIMIT ?
            """, (cutoff_date, metric_name, min_runs, limit))

            return [dict(row) for row in cursor.fetchall()]

    def cleanup_old_runs(self, days_to_keep: int = 90) -> int:
        """
        Clean up old runs and symbol results.

        Args:
            days_to_keep: Number of days to keep

        Returns:
            Number of runs deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get runs to delete
            cursor.execute("""
                SELECT id FROM scoring_runs
                WHERE completed_at < ? OR (started_at < ? AND status = 'running')
            """, (cutoff_date, cutoff_date))
            old_run_ids = [row[0] for row in cursor.fetchall()]

            if not old_run_ids:
                return 0

            # Delete symbol results first (foreign key constraint)
            placeholders = ','.join(['?'] * len(old_run_ids))
            cursor.execute(f"""
                DELETE FROM symbol_results
                WHERE run_id IN ({placeholders})
            """, old_run_ids)

            # Delete runs
            cursor.execute(f"""
                DELETE FROM scoring_runs
                WHERE id IN ({placeholders})
            """, old_run_ids)

            conn.commit()
            deleted_count = len(old_run_ids)

            self.logger.info(f"Cleaned up {deleted_count} old runs")
            return deleted_count

    def _config_to_dict(self, config: ScoringConfig) -> Dict[str, Any]:
        """Convert ScoringConfig to dictionary for JSON storage."""
        # Simplified conversion - in production would need complete serialization
        return {
            "name": config.name,
            "version": config.version,
            "description": config.description,
            "created_by": config.created_by,
            "created_at": config.created_at,
            "components": [
                {
                    "name": comp.name,
                    "weight": comp.weight,
                    "method": comp.method.value,
                    "lookback": comp.lookback,
                    "enabled": comp.enabled,
                    "indicator_key": comp.indicator_key,
                    "fallback_keys": comp.fallback_keys,
                    "parameters": comp.parameters
                }
                for comp in config.components
            ],
            "bonus_penalty_rules": [
                {
                    "name": rule.name,
                    "description": rule.description,
                    "condition": rule.condition,
                    "operator": rule.operator.value,
                    "value": rule.value,
                    "bonus": rule.bonus,
                    "enabled": rule.enabled
                }
                for rule in config.bonus_penalty_rules
            ],
            "probability_thresholds": config.probability_thresholds,
            "max_total_score": config.max_total_score,
            "min_total_score": config.min_total_score
        }

    def _generate_run_hash(self, config_id: int, timestamp: datetime) -> str:
        """Generate unique hash for a scoring run."""
        run_string = f"{config_id}_{timestamp.isoformat()}"
        return hashlib.md5(run_string.encode()).hexdigest()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Configuration stats
            cursor.execute("SELECT COUNT(*) as total, COUNT(CASE WHEN is_active = 1 THEN 1 END) as active FROM scoring_configs")
            config_stats = cursor.fetchone()
            stats['configs'] = dict(config_stats)

            # Run stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'running' THEN 1 END) as running
                FROM scoring_runs
            """)
            run_stats = cursor.fetchone()
            stats['runs'] = dict(run_stats)

            # Symbol results stats
            cursor.execute("SELECT COUNT(*) as total FROM symbol_results")
            result_stats = cursor.fetchone()
            stats['symbol_results'] = dict(result_stats)

            return stats


# Example usage and testing
if __name__ == "__main__":
    # Test the parameter store
    from .scoring_schema import create_default_config

    # Create test configuration
    config = create_default_config()

    # Initialize store
    store = ParameterStore("test_scoring_params.db")

    # Save configuration
    config_id = store.save_config(config, metadata={"test": True, "version": "1.0"})
    print(f"Saved config with ID: {config_id}")

    # Start a run
    run_id = store.start_run(config_id, metadata={"test_run": True})
    print(f"Started run with ID: {run_id}")

    # Track some performance
    store.track_config_performance(config_id, "test_metric", 0.85)

    # Get stats
    stats = store.get_database_stats()
    print(f"Database stats: {stats}")

    print("Parameter store test completed successfully!")
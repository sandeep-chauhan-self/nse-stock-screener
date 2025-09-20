"""
Reproducibility and Persistence Layer for Backtesting
=====================================================

This module provides comprehensive facilities for ensuring reproducible backtests
and persisting results:
- Deterministic random seeding across all components
- Equity curve tracking and serialization
- Trade log persistence with full audit trail
- Configuration versioning and change tracking
- Result caching and retrieval
- Export to multiple formats (CSV, Parquet, JSON, Excel)

Designed to meet FS.6 requirements for reproducibility and audit trails.
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import warnings
from copy import deepcopy
import gzip
import yaml

logger = logging.getLogger(__name__)

# =====================================================================================
# DATA STRUCTURES
# =====================================================================================

@dataclass
class BacktestMetadata:
    """Metadata for backtest runs"""
    
    # Identification
    backtest_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0"
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Data parameters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbols: List[str] = field(default_factory=list)
    
    # Configuration
    config_hash: str = ""
    random_seed: Optional[int] = None
    
    # Environment
    python_version: str = ""
    platform: str = ""
    packages: Dict[str, str] = field(default_factory=dict)
    
    # Results summary
    total_trades: int = 0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # File paths
    equity_curve_file: str = ""
    trade_log_file: str = ""
    config_file: str = ""
    full_results_file: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestMetadata':
        """Create from dictionary"""
        # Handle datetime fields
        datetime_fields = ['created_at', 'started_at', 'completed_at', 'start_date', 'end_date']
        for field_name in datetime_fields:
            if field_name in data and data[field_name] is not None:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)

@dataclass
class EquityCurvePoint:
    """Single point in equity curve"""
    timestamp: datetime
    portfolio_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_commission: float
    
    # Performance metrics
    daily_return: Optional[float] = None
    cumulative_return: Optional[float] = None
    drawdown: Optional[float] = None
    
    # Position details
    active_positions: int = 0
    position_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradeLogEntry:
    """Detailed trade log entry"""
    
    # Trade identification
    trade_id: str
    parent_signal_id: Optional[str] = None
    
    # Basic trade details
    symbol: str = ""
    side: str = ""  # 'buy' or 'sell'
    quantity: int = 0
    
    # Timing
    signal_timestamp: datetime = field(default_factory=datetime.now)
    order_timestamp: datetime = field(default_factory=datetime.now)
    fill_timestamp: Optional[datetime] = None
    exit_timestamp: Optional[datetime] = None
    
    # Prices and execution
    signal_price: float = 0.0
    order_price: float = 0.0
    fill_price: float = 0.0
    exit_price: Optional[float] = None
    
    # Costs
    commission: float = 0.0
    slippage: float = 0.0
    total_cost: float = 0.0
    
    # Performance
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    return_pct: Optional[float] = None
    holding_period_days: Optional[int] = None
    
    # Context
    market_data: Dict[str, Any] = field(default_factory=dict)
    signal_data: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    # Exit details
    exit_reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

# =====================================================================================
# RANDOM SEED MANAGER
# =====================================================================================

class SeedManager:
    """
    Centralized random seed management for reproducibility
    """
    
    def __init__(self, master_seed: int = 42):
        """
        Initialize seed manager
        
        Args:
            master_seed: Master seed for all random operations
        """
        self.master_seed = master_seed
        self.component_seeds: Dict[str, int] = {}
        self.generators: Dict[str, np.random.Generator] = {}
        
        # Initialize global numpy random state
        np.random.seed(master_seed)
        
        logger.info("Initialized SeedManager with master seed %d", master_seed)
    
    def get_seed(self, component: str) -> int:
        """
        Get deterministic seed for a component
        
        Args:
            component: Component name (e.g., 'slippage', 'execution', 'signals')
            
        Returns:
            Deterministic seed for the component
        """
        if component not in self.component_seeds:
            # Generate deterministic seed based on master seed and component name
            component_hash = hashlib.md5(f"{self.master_seed}_{component}".encode()).hexdigest()
            component_seed = int(component_hash[:8], 16) % (2**31 - 1)
            self.component_seeds[component] = component_seed
            
            logger.debug("Generated seed %d for component '%s'", component_seed, component)
        
        return self.component_seeds[component]
    
    def get_generator(self, component: str) -> np.random.Generator:
        """
        Get numpy random generator for a component
        
        Args:
            component: Component name
            
        Returns:
            Seeded numpy random generator
        """
        if component not in self.generators:
            seed = self.get_seed(component)
            self.generators[component] = np.random.default_rng(seed)
        
        return self.generators[component]
    
    def reset_component(self, component: str) -> None:
        """Reset a specific component's random state"""
        if component in self.generators:
            seed = self.component_seeds[component]
            self.generators[component] = np.random.default_rng(seed)
            logger.debug("Reset random state for component '%s'", component)
    
    def reset_all(self) -> None:
        """Reset all components' random states"""
        for component in self.generators:
            self.reset_component(component)
        
        # Reset global numpy state
        np.random.seed(self.master_seed)
        logger.info("Reset all random states")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get information about current random states"""
        return {
            'master_seed': self.master_seed,
            'component_seeds': self.component_seeds.copy(),
            'active_generators': list(self.generators.keys())
        }

# =====================================================================================
# PERSISTENCE LAYER
# =====================================================================================

class BacktestPersistence:
    """
    Handles persistence of backtest results and metadata
    """
    
    def __init__(self, base_path: Union[str, Path] = "backtest_results"):
        """
        Initialize persistence layer
        
        Args:
            base_path: Base directory for storing results
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "metadata").mkdir(exist_ok=True)
        (self.base_path / "equity_curves").mkdir(exist_ok=True)
        (self.base_path / "trade_logs").mkdir(exist_ok=True)
        (self.base_path / "configs").mkdir(exist_ok=True)
        (self.base_path / "full_results").mkdir(exist_ok=True)
        (self.base_path / "exports").mkdir(exist_ok=True)
        
        logger.info("Initialized BacktestPersistence at %s", self.base_path)
    
    def generate_run_id(self, prefix: str = "backtest") -> str:
        """Generate unique run ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_{short_uuid}"
    
    def save_metadata(self, metadata: BacktestMetadata) -> str:
        """
        Save backtest metadata
        
        Args:
            metadata: Backtest metadata
            
        Returns:
            Path to saved file
        """
        filename = f"metadata_{metadata.backtest_id}.json"
        filepath = self.base_path / "metadata" / filename
        
        # Convert to dict and handle datetime serialization
        data = metadata.to_dict()
        
        # Custom JSON encoder for datetime
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=DateTimeEncoder)
        
        logger.info("Saved metadata to %s", filepath)
        return str(filepath)
    
    def load_metadata(self, backtest_id: str) -> BacktestMetadata:
        """
        Load backtest metadata
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            BacktestMetadata object
        """
        filename = f"metadata_{backtest_id}.json"
        filepath = self.base_path / "metadata" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return BacktestMetadata.from_dict(data)
    
    def save_equity_curve(self, backtest_id: str, 
                         equity_curve: List[EquityCurvePoint]) -> str:
        """
        Save equity curve data
        
        Args:
            backtest_id: Backtest ID
            equity_curve: List of equity curve points
            
        Returns:
            Path to saved file
        """
        filename = f"equity_curve_{backtest_id}.parquet"
        filepath = self.base_path / "equity_curves" / filename
        
        # Convert to DataFrame
        df_data = []
        for point in equity_curve:
            row = asdict(point)
            # Convert datetime to timestamp
            row['timestamp'] = point.timestamp
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        
        # Save as Parquet for efficiency
        df.to_parquet(filepath, compression='gzip')
        
        logger.info("Saved equity curve (%d points) to %s", len(equity_curve), filepath)
        return str(filepath)
    
    def load_equity_curve(self, backtest_id: str) -> List[EquityCurvePoint]:
        """
        Load equity curve data
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            List of EquityCurvePoint objects
        """
        filename = f"equity_curve_{backtest_id}.parquet"
        filepath = self.base_path / "equity_curves" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Equity curve file not found: {filepath}")
        
        df = pd.read_parquet(filepath)
        
        # Convert back to EquityCurvePoint objects
        equity_curve = []
        for timestamp, row in df.iterrows():
            point = EquityCurvePoint(
                timestamp=timestamp,
                portfolio_value=row['portfolio_value'],
                cash=row['cash'],
                positions_value=row['positions_value'],
                unrealized_pnl=row['unrealized_pnl'],
                realized_pnl=row['realized_pnl'],
                total_commission=row['total_commission'],
                daily_return=row.get('daily_return'),
                cumulative_return=row.get('cumulative_return'),
                drawdown=row.get('drawdown'),
                active_positions=row.get('active_positions', 0),
                position_details=row.get('position_details', {})
            )
            equity_curve.append(point)
        
        return equity_curve
    
    def save_trade_log(self, backtest_id: str, 
                      trade_log: List[TradeLogEntry]) -> str:
        """
        Save trade log
        
        Args:
            backtest_id: Backtest ID
            trade_log: List of trade log entries
            
        Returns:
            Path to saved file
        """
        filename = f"trade_log_{backtest_id}.parquet"
        filepath = self.base_path / "trade_logs" / filename
        
        # Convert to DataFrame
        df_data = [entry.to_dict() for entry in trade_log]
        df = pd.DataFrame(df_data)
        
        if not df.empty:
            # Convert timestamp columns
            timestamp_cols = ['signal_timestamp', 'order_timestamp', 'fill_timestamp', 'exit_timestamp']
            for col in timestamp_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Save as Parquet
            df.to_parquet(filepath, compression='gzip')
        
        logger.info("Saved trade log (%d entries) to %s", len(trade_log), filepath)
        return str(filepath)
    
    def load_trade_log(self, backtest_id: str) -> List[TradeLogEntry]:
        """
        Load trade log
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            List of TradeLogEntry objects
        """
        filename = f"trade_log_{backtest_id}.parquet"
        filepath = self.base_path / "trade_logs" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Trade log file not found: {filepath}")
        
        df = pd.read_parquet(filepath)
        
        # Convert back to TradeLogEntry objects
        trade_log = []
        for _, row in df.iterrows():
            entry = TradeLogEntry(
                trade_id=row['trade_id'],
                parent_signal_id=row.get('parent_signal_id'),
                symbol=row['symbol'],
                side=row['side'],
                quantity=row['quantity'],
                signal_timestamp=row['signal_timestamp'],
                order_timestamp=row['order_timestamp'],
                fill_timestamp=row.get('fill_timestamp'),
                exit_timestamp=row.get('exit_timestamp'),
                signal_price=row['signal_price'],
                order_price=row['order_price'],
                fill_price=row['fill_price'],
                exit_price=row.get('exit_price'),
                commission=row['commission'],
                slippage=row['slippage'],
                total_cost=row['total_cost'],
                gross_pnl=row.get('gross_pnl'),
                net_pnl=row.get('net_pnl'),
                return_pct=row.get('return_pct'),
                holding_period_days=row.get('holding_period_days'),
                market_data=row.get('market_data', {}),
                signal_data=row.get('signal_data', {}),
                execution_context=row.get('execution_context', {}),
                exit_reason=row.get('exit_reason', ''),
                stop_loss=row.get('stop_loss'),
                take_profit=row.get('take_profit')
            )
            trade_log.append(entry)
        
        return trade_log
    
    def save_config(self, backtest_id: str, config: Dict[str, Any]) -> str:
        """
        Save configuration
        
        Args:
            backtest_id: Backtest ID
            config: Configuration dictionary
            
        Returns:
            Path to saved file
        """
        filename = f"config_{backtest_id}.yaml"
        filepath = self.base_path / "configs" / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info("Saved config to %s", filepath)
        return str(filepath)
    
    def save_full_results(self, backtest_id: str, results: Dict[str, Any]) -> str:
        """
        Save complete backtest results
        
        Args:
            backtest_id: Backtest ID
            results: Full results dictionary
            
        Returns:
            Path to saved file
        """
        filename = f"results_{backtest_id}.pkl.gz"
        filepath = self.base_path / "full_results" / filename
        
        # Save as compressed pickle
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info("Saved full results to %s", filepath)
        return str(filepath)
    
    def load_full_results(self, backtest_id: str) -> Dict[str, Any]:
        """
        Load complete backtest results
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            Full results dictionary
        """
        filename = f"results_{backtest_id}.pkl.gz"
        filepath = self.base_path / "full_results" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with gzip.open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def list_backtests(self) -> List[Dict[str, Any]]:
        """
        List all available backtests
        
        Returns:
            List of backtest summaries
        """
        metadata_dir = self.base_path / "metadata"
        backtests = []
        
        for metadata_file in metadata_dir.glob("metadata_*.json"):
            try:
                backtest_id = metadata_file.stem.replace("metadata_", "")
                metadata = self.load_metadata(backtest_id)
                
                backtests.append({
                    'backtest_id': backtest_id,
                    'name': metadata.name,
                    'created_at': metadata.created_at,
                    'start_date': metadata.start_date,
                    'end_date': metadata.end_date,
                    'total_trades': metadata.total_trades,
                    'total_return': metadata.total_return,
                    'sharpe_ratio': metadata.sharpe_ratio,
                    'max_drawdown': metadata.max_drawdown
                })
            except Exception as e:
                logger.warning("Error loading metadata for %s: %s", metadata_file, e)
        
        # Sort by creation date
        backtests.sort(key=lambda x: x['created_at'], reverse=True)
        
        return backtests
    
    def export_results(self, backtest_id: str, format_type: str = "excel") -> str:
        """
        Export backtest results to various formats
        
        Args:
            backtest_id: Backtest ID
            format_type: Export format ('excel', 'csv', 'json')
            
        Returns:
            Path to exported file
        """
        # Load data
        metadata = self.load_metadata(backtest_id)
        equity_curve = self.load_equity_curve(backtest_id)
        trade_log = self.load_trade_log(backtest_id)
        
        # Convert to DataFrames
        equity_df = pd.DataFrame([asdict(point) for point in equity_curve])
        trade_df = pd.DataFrame([entry.to_dict() for entry in trade_log])
        
        # Export based on format
        export_dir = self.base_path / "exports"
        
        if format_type == "excel":
            filename = f"backtest_export_{backtest_id}.xlsx"
            filepath = export_dir / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Write metadata as a summary sheet
                metadata_df = pd.Series(metadata.to_dict()).to_frame('Value')
                metadata_df.to_excel(writer, sheet_name='Summary')
                
                # Write equity curve
                equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
                
                # Write trade log
                trade_df.to_excel(writer, sheet_name='Trade Log', index=False)
        
        elif format_type == "csv":
            # Create a folder for CSV files
            csv_folder = export_dir / f"backtest_csv_{backtest_id}"
            csv_folder.mkdir(exist_ok=True)
            
            # Save individual CSV files
            metadata_df = pd.Series(metadata.to_dict()).to_frame('Value')
            metadata_df.to_csv(csv_folder / "summary.csv")
            equity_df.to_csv(csv_folder / "equity_curve.csv", index=False)
            trade_df.to_csv(csv_folder / "trade_log.csv", index=False)
            
            filepath = csv_folder
        
        elif format_type == "json":
            filename = f"backtest_export_{backtest_id}.json"
            filepath = export_dir / filename
            
            export_data = {
                'metadata': metadata.to_dict(),
                'equity_curve': equity_df.to_dict('records'),
                'trade_log': trade_df.to_dict('records')
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        logger.info("Exported backtest %s to %s (%s format)", backtest_id, filepath, format_type)
        return str(filepath)

# =====================================================================================
# EQUITY CURVE TRACKER
# =====================================================================================

class EquityCurveTracker:
    """
    Tracks equity curve in real-time during backtesting
    """
    
    def __init__(self, initial_capital: float = 1000000.0):
        """
        Initialize equity curve tracker
        
        Args:
            initial_capital: Starting portfolio value
        """
        self.initial_capital = initial_capital
        self.equity_curve: List[EquityCurvePoint] = []
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize first point
        initial_point = EquityCurvePoint(
            timestamp=datetime.now(),
            portfolio_value=initial_capital,
            cash=initial_capital,
            positions_value=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_commission=0.0,
            active_positions=0
        )
        self.equity_curve.append(initial_point)
        
        logger.info("Initialized EquityCurveTracker with ₹%.0f capital", initial_capital)
    
    def update(self, timestamp: datetime, 
               cash: float,
               positions: Dict[str, Dict[str, Any]],
               realized_pnl: float = 0.0,
               commission: float = 0.0) -> None:
        """
        Update equity curve with current portfolio state
        
        Args:
            timestamp: Current timestamp
            cash: Available cash
            positions: Current positions dict
            realized_pnl: Realized P&L from closed trades
            commission: Commission costs
        """
        # Calculate positions value and unrealized P&L
        positions_value = 0.0
        unrealized_pnl = 0.0
        
        for symbol, position in positions.items():
            quantity = position.get('quantity', 0)
            current_price = position.get('current_price', 0)
            avg_cost = position.get('avg_cost', 0)
            
            market_value = quantity * current_price
            cost_basis = quantity * avg_cost
            
            positions_value += market_value
            unrealized_pnl += (market_value - cost_basis)
        
        # Calculate total portfolio value
        portfolio_value = cash + positions_value
        
        # Calculate daily return
        daily_return = None
        cumulative_return = None
        drawdown = None
        
        if len(self.equity_curve) > 0:
            prev_value = self.equity_curve[-1].portfolio_value
            if prev_value > 0:
                daily_return = (portfolio_value - prev_value) / prev_value
                cumulative_return = (portfolio_value - self.initial_capital) / self.initial_capital
                
                # Calculate drawdown
                peak_value = max(point.portfolio_value for point in self.equity_curve)
                if peak_value > 0:
                    drawdown = (portfolio_value - peak_value) / peak_value
        
        # Create new equity curve point
        point = EquityCurvePoint(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            cash=cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_commission=self.get_total_commission() + commission,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            drawdown=drawdown,
            active_positions=len(positions),
            position_details=deepcopy(positions)
        )
        
        self.equity_curve.append(point)
        self.current_positions = deepcopy(positions)
    
    def get_total_commission(self) -> float:
        """Get total commission paid so far"""
        if self.equity_curve:
            return self.equity_curve[-1].total_commission
        return 0.0
    
    def get_current_value(self) -> float:
        """Get current portfolio value"""
        if self.equity_curve:
            return self.equity_curve[-1].portfolio_value
        return self.initial_capital
    
    def get_returns_series(self) -> pd.Series:
        """Get daily returns as pandas Series"""
        if len(self.equity_curve) < 2:
            return pd.Series(dtype=float)
        
        returns = []
        timestamps = []
        
        for i in range(1, len(self.equity_curve)):
            point = self.equity_curve[i]
            if point.daily_return is not None:
                returns.append(point.daily_return)
                timestamps.append(point.timestamp)
        
        return pd.Series(returns, index=timestamps)
    
    def get_equity_series(self) -> pd.Series:
        """Get equity curve as pandas Series"""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        
        values = [point.portfolio_value for point in self.equity_curve]
        timestamps = [point.timestamp for point in self.equity_curve]
        
        return pd.Series(values, index=timestamps)

# =====================================================================================
# CONFIGURATION VERSIONING
# =====================================================================================

def calculate_config_hash(config: Dict[str, Any]) -> str:
    """
    Calculate deterministic hash of configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SHA-256 hash of configuration
    """
    # Convert to JSON string with sorted keys for deterministic hashing
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]

def create_environment_snapshot() -> Dict[str, Any]:
    """
    Create snapshot of current environment
    
    Returns:
        Dictionary with environment information
    """
    import platform
    import sys
    import pkg_resources
    
    # Get package versions
    packages = {}
    try:
        for package in pkg_resources.working_set:
            packages[package.project_name] = package.version
    except Exception as e:
        logger.warning("Could not collect package versions: %s", e)
    
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'packages': packages
    }

# =====================================================================================
# EXAMPLE USAGE
# =====================================================================================

if __name__ == "__main__":
    # Example usage of reproducibility and persistence
    
    # Initialize components
    seed_manager = SeedManager(master_seed=12345)
    persistence = BacktestPersistence("example_backtest_results")
    equity_tracker = EquityCurveTracker(initial_capital=1000000)
    
    # Create sample metadata
    metadata = BacktestMetadata(
        name="Example Backtest",
        description="Sample backtest for testing persistence",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        symbols=["RELIANCE.NS", "TCS.NS"],
        random_seed=seed_manager.master_seed
    )
    
    # Generate some sample data
    rng = seed_manager.get_generator('simulation')
    
    # Sample equity curve
    base_date = datetime(2023, 1, 1)
    for i in range(100):
        date = base_date + timedelta(days=i)
        
        # Simulate portfolio changes
        daily_return = rng.normal(0.001, 0.02)  # 0.1% daily return with 2% volatility
        new_value = equity_tracker.get_current_value() * (1 + daily_return)
        
        # Mock positions
        positions = {
            'RELIANCE.NS': {
                'quantity': 100,
                'current_price': 2500 + rng.normal(0, 50),
                'avg_cost': 2500
            }
        }
        
        equity_tracker.update(
            timestamp=date,
            cash=new_value * 0.7,  # 70% cash
            positions=positions,
            commission=rng.uniform(0, 100)
        )
    
    # Sample trade log
    trade_log = []
    for i in range(10):
        trade = TradeLogEntry(
            trade_id=f"trade_{i}",
            symbol="RELIANCE.NS",
            side="buy" if i % 2 == 0 else "sell",
            quantity=100,
            signal_timestamp=base_date + timedelta(days=i*10),
            order_timestamp=base_date + timedelta(days=i*10, hours=1),
            fill_timestamp=base_date + timedelta(days=i*10, hours=2),
            signal_price=2500.0,
            order_price=2501.0,
            fill_price=2502.0,
            commission=50.0,
            slippage=2.0,
            total_cost=250250.0
        )
        trade_log.append(trade)
    
    # Save everything
    backtest_id = persistence.generate_run_id("example")
    metadata.backtest_id = backtest_id
    
    # Update metadata with results
    returns_series = equity_tracker.get_returns_series()
    if len(returns_series) > 0:
        metadata.total_return = (equity_tracker.get_current_value() - equity_tracker.initial_capital) / equity_tracker.initial_capital
        metadata.total_trades = len(trade_log)
    
    # Save all components
    persistence.save_metadata(metadata)
    persistence.save_equity_curve(backtest_id, equity_tracker.equity_curve)
    persistence.save_trade_log(backtest_id, trade_log)
    persistence.save_config(backtest_id, {'example': 'config'})
    
    # Export results
    export_path = persistence.export_results(backtest_id, "excel")
    
    print("📊 Backtest Results Saved")
    print(f"ID: {backtest_id}")
    print(f"Equity Points: {len(equity_tracker.equity_curve)}")
    print(f"Trades: {len(trade_log)}")
    print(f"Exported to: {export_path}")
    
    # List all backtests
    all_backtests = persistence.list_backtests()
    print(f"\nAll Backtests: {len(all_backtests)} found")
    
    # Demonstrate reproducibility
    print("\nRandom Seeds:")
    print(f"Master: {seed_manager.master_seed}")
    for component, seed in seed_manager.component_seeds.items():
        print(f"  {component}: {seed}")
"""
Advanced Order Execution Engine for Realistic Backtesting
=========================================================
This module implements a sophisticated order execution simulation that models:
- Multiple order types (market, limit, stop-limit)
- Realistic commission structures for Indian markets
- Volume-dependent slippage and market impact
- Partial fills based on liquidity constraints
- Bid-ask spread simulation
- Queue position modeling for limit orders
Designed to provide faithful simulation of real-world execution for FS.6 compliance.
"""
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid
import random
logger = logging.getLogger(__name__)

# =====================================================================================
# ENUMS AND DATA STRUCTURES

# =====================================================================================
class OrderType(Enum):
    """Supported order types for execution simulation"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"
class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "buy"
    SELL = "sell"
class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
class FillReason(Enum):
    """Reason for order fill"""
    MARKET_ORDER = "market_order"
    LIMIT_HIT = "limit_hit"
    STOP_TRIGGERED = "stop_triggered"
    PARTIAL_LIQUIDITY = "partial_liquidity"
@dataclass
class Fill:
    """Individual fill record"""
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    total_cost: float = 0.0
  # Including commission
    fill_reason: FillReason = FillReason.MARKET_ORDER
    market_data: dict[str, Any] = field(default_factory=dict[str, Any])
@dataclass
class Order:
    """Order representation with execution tracking"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: Optional[float] = None
  # For limit orders
    stop_price: Optional[float] = None
  # For stop orders
    time_in_force: str = "DAY"
  # DAY, GTC, IOC, FOK
    # Order status tracking
    status: OrderStatus = OrderStatus.PENDING
    submitted_time: datetime = field(default_factory=datetime.now)
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0.0
    total_commission: float = 0.0

    # Execution details
    fills: list[Fill] = field(default_factory=list[str])
    rejection_reason: str = ""
    def __post_init__(self):
        """Initialize remaining quantity"""
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED,
                              OrderStatus.REJECTED, OrderStatus.EXPIRED]
    @property
    def fill_ratio(self) -> float:
        """Calculate fill ratio (0-1)"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0
    def add_fill(self, fill: Fill) -> None:
        """Add a fill to the order"""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        self.total_commission += fill.commission

        # Update average fill price
        total_value = sum(f.quantity * f.price for f in self.fills)
        self.avg_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0.0

        # Update status
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
@dataclass
class MarketData:
    """Market data snapshot for execution simulation"""
    timestamp: datetime
    symbol: str
    open_price: float
    high: float
    low: float
    close: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    vwap: Optional[float] = None
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bid and self.ask:
            return self.ask - self.bid
        return 0.0
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points"""
        if self.bid and self.ask and self.ask > 0:
            return (self.spread / self.ask) * 10000
        return 0.0
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.close

# =====================================================================================
# EXECUTION MODELS

# =====================================================================================
class ExecutionModel(ABC):
    """Abstract base class for execution models"""
    @abstractmethod
    def execute_order(self, order: Order, market_data: MarketData,
                     execution_context: dict[str, Any]) -> list[Fill]:
        """Execute an order against market data"""
        pass
class IndianMarketExecutionModel(ExecutionModel):
    """
    Execution model tailored for Indian equity markets with NSE-specific characteristics:
    - Commission structure (brokerage, STT, GST, stamp duty)
    - Slippage modeling based on order size vs average volume
    - Partial fills for large orders
    - Realistic bid-ask spread simulation
    """
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize execution model with configuration
        Args:
            config: Dictionary containing execution parameters
        """

        # Commission structure
        self.brokerage_rate = config.get('brokerage_rate', 0.0003)
        self.stt_rate = config.get('stt_rate', 0.00025)
        self.exchange_charges = config.get('exchange_charges', 0.0000345)
        self.gst_rate = config.get('gst_rate', 0.18)
        self.stamp_duty_rate = config.get('stamp_duty_rate', 0.00015)

        # Slippage parameters
        self.slippage_model = config.get('slippage_model', 'adaptive')
        self.base_slippage_bps = config.get('base_slippage_bps', 2.0)
        self.market_impact_factor = config.get('market_impact_factor', 0.1)
        self.liquidity_impact_threshold = config.get('liquidity_impact_threshold', 0.01)

        # Execution parameters
        self.partial_fill_enabled = config.get('partial_fill_enabled', True)
        self.min_fill_ratio = config.get('min_fill_ratio', 0.5)
        self.max_spread_bps = config.get('max_spread_bps', 50.0)

        # Random seed for reproducibility
        self.random_seed = config.get('random_seed', 42)
        self.rng = np.random.default_rng(self.random_seed)
        logger.info(f"Initialized IndianMarketExecutionModel with {self.slippage_model} slippage")
    def calculate_commission(self, trade_value: float,
                           is_buy: bool = True) -> dict[str, float]:
        """
        Calculate detailed commission structure for Indian markets
        Args:
            trade_value: Total value of trade (quantity * price)
            is_buy: Whether this is a buy order
        Returns:
            Dictionary with breakdown of all costs
        """
        commission_breakdown = {}

        # 1. Brokerage (per side)
        brokerage = trade_value * self.brokerage_rate
        commission_breakdown['brokerage'] = brokerage

        # 2. STT - Securities Transaction Tax
        stt = trade_value * self.stt_rate
        commission_breakdown['stt'] = stt

        # 3. Exchange charges
        exchange_charges = trade_value * self.exchange_charges
        commission_breakdown['exchange_charges'] = exchange_charges

        # 4. GST on (Brokerage + Exchange charges)
        gst_base = brokerage + exchange_charges
        gst = gst_base * self.gst_rate
        commission_breakdown['gst'] = gst

        # 5. Stamp duty (buy side only)
        if is_buy:
            stamp_duty = trade_value * self.stamp_duty_rate
            commission_breakdown['stamp_duty'] = stamp_duty
        else:
            commission_breakdown['stamp_duty'] = 0.0

        # Total commission
        total_commission = sum(commission_breakdown.values())
        commission_breakdown['total'] = total_commission
        return commission_breakdown
    def calculate_slippage(self, order: Order, market_data: MarketData,
                          avg_volume: Optional[float] = None) -> float:
        """
        Calculate slippage based on order characteristics and market conditions
        Args:
            order: Order to execute
            market_data: Current market data
            avg_volume: Average daily trading volume
        Returns:
            Slippage amount per share
        """
        base_price = market_data.mid_price or market_data.close
        if self.slippage_model == "fixed":

            # Fixed slippage per share
            return self.base_slippage_bps / 10000 * base_price
        elif self.slippage_model == "adaptive":

            # Adaptive slippage based on order size
            base_slippage = self.base_slippage_bps / 10000 * base_price

            # Adjust for order size if volume data available
            size_multiplier = 1.0
            if avg_volume and avg_volume > 0:
                order_participation = order.quantity / avg_volume
                if order_participation > self.liquidity_impact_threshold:

                    # Larger orders get higher slippage
                    size_multiplier = 1.0 + (order_participation * self.market_impact_factor)
            return base_slippage * size_multiplier
        elif self.slippage_model == "liquidity_based":

            # Advanced market impact model
            if avg_volume and avg_volume > 0:
                order_participation = order.quantity / avg_volume

                # Square root market impact model
                impact_factor = self.market_impact_factor * (order_participation ** 0.5)
                return base_price * impact_factor
            else:

                # Fallback to adaptive
                return self.base_slippage_bps / 10000 * base_price
        else:

            # Default to adaptive
            return self.base_slippage_bps / 10000 * base_price
    def simulate_bid_ask_spread(self, market_data: MarketData) -> tuple[float, float]:
        """
        Simulate realistic bid-ask spread if not provided in market data
        Args:
            market_data: Market data snapshot
        Returns:
            tuple[str, ...] of (bid, ask) prices
        """
        if market_data.bid and market_data.ask:
            return market_data.bid, market_data.ask

        # Simulate spread based on volatility and liquidity
        base_price = market_data.close

        # Estimate spread based on volume (inverse relationship)
        if market_data.volume > 0:

            # Higher volume = tighter spread
            volume_factor = max(0.1, min(1.0, 1000000 / market_data.volume))
        else:
            volume_factor = 1.0

        # Base spread in basis points (wider for lower volume)
        spread_bps = max(1.0, min(self.max_spread_bps, 5.0 * volume_factor))
        spread_amount = (spread_bps / 10000) * base_price
        bid = base_price - spread_amount / 2
        ask = base_price + spread_amount / 2
        return bid, ask
    def execute_market_order(self, order: Order, market_data: MarketData,
                           execution_context: dict[str, Any]) -> list[Fill]:
        """Execute market order with realistic slippage and partial fills"""
        fills = []

        # Get average volume for slippage calculation
        avg_volume = execution_context.get('avg_volume', market_data.volume)

        # Calculate slippage
        slippage_per_share = self.calculate_slippage(order, market_data, avg_volume)

        # Determine execution price
        bid, ask = self.simulate_bid_ask_spread(market_data)
        if order.side == OrderSide.BUY:
            base_price = ask
  # Buy at ask
            execution_price = base_price + slippage_per_share
  # Additional slippage
        else:
            base_price = bid
  # Sell at bid
            execution_price = base_price - slippage_per_share
  # Slippage reduces price
        # Simulate partial fills for large orders
        remaining_quantity = order.remaining_quantity
        if self.partial_fill_enabled and avg_volume:
            participation_rate = remaining_quantity / avg_volume
            if participation_rate > self.liquidity_impact_threshold:

                # Large order - may need partial fills
                max_fill_pct = max(self.min_fill_ratio,
                                 1.0 - min(0.5, participation_rate * 0.8))
                fill_quantity = int(remaining_quantity * max_fill_pct)
            else:
                fill_quantity = remaining_quantity
        else:
            fill_quantity = remaining_quantity
        if fill_quantity > 0:

            # Calculate trade value and commission
            trade_value = fill_quantity * execution_price
            commission_breakdown = self.calculate_commission(
                trade_value, order.side == OrderSide.BUY
            )

            # Create fill
            fill = Fill(
                timestamp=market_data.timestamp,
                quantity=fill_quantity,
                price=execution_price,
                commission=commission_breakdown['total'],
                total_cost=trade_value + commission_breakdown['total'],
                fill_reason=FillReason.MARKET_ORDER,
                market_data={
                    'bid': bid,
                    'ask': ask,
                    'slippage_per_share': slippage_per_share,
                    'commission_breakdown': commission_breakdown,
                    'avg_volume': avg_volume,
                    'participation_rate': remaining_quantity / avg_volume if avg_volume else 0
                }
            )
            fills.append(fill)
        return fills
    def execute_limit_order(self, order: Order, market_data: MarketData,
                          execution_context: dict[str, Any]) -> list[Fill]:
        """Execute limit order with queue position simulation"""
        fills = []
        if not order.price:
            return fills
  # No limit price specified
        bid, ask = self.simulate_bid_ask_spread(market_data)

        # Check if limit order can be executed and get execution price
        execution_details = self._check_limit_execution(order, bid, ask)
        if not execution_details['can_execute']:
            return fills
        execution_price = execution_details['execution_price']
        remaining_quantity = order.remaining_quantity

        # Simple queue simulation - assume we're somewhere in the queue
        queue_position_factor = self.rng.uniform(0.7, 1.0)
  # 70-100% of order filled
        fill_quantity = int(remaining_quantity * queue_position_factor)
        if fill_quantity > 0:

            # Calculate commission
            trade_value = fill_quantity * execution_price
            commission_breakdown = self.calculate_commission(
                trade_value, order.side == OrderSide.BUY
            )

            # Create fill
            fill = Fill(
                timestamp=market_data.timestamp,
                quantity=fill_quantity,
                price=execution_price,
                commission=commission_breakdown['total'],
                total_cost=trade_value + commission_breakdown['total'],
                fill_reason=FillReason.LIMIT_HIT,
                market_data={
                    'bid': bid,
                    'ask': ask,
                    'limit_price': order.price,
                    'queue_position_factor': queue_position_factor,
                    'commission_breakdown': commission_breakdown
                }
            )
            fills.append(fill)
        return fills
    def _check_limit_execution(self, order: Order, bid: float, ask: float) -> dict[str, Any]:
        """Helper method to check if limit order can execute and determine price"""
        can_execute = False
        execution_price = order.price
        if order.side == OrderSide.BUY:

            # Buy limit: can execute if ask <= limit price
            if ask <= order.price:
                can_execute = True
                execution_price = min(ask, order.price)
  # Price improvement possible
        else:

            # Sell limit: can execute if bid >= limit price
            if bid >= order.price:
                can_execute = True
                execution_price = max(bid, order.price)
  # Price improvement possible
        return {
            'can_execute': can_execute,
            'execution_price': execution_price
        }
    def execute_stop_order(self, order: Order, market_data: MarketData,
                         execution_context: dict[str, Any]) -> list[Fill]:
        """Execute stop order when triggered"""
        fills = []
        if not order.stop_price:
            return fills

        # Check if stop is triggered
        if not self._is_stop_triggered(order, market_data):
            return fills

        # Execute based on stop order type
        if order.order_type == OrderType.STOP_MARKET:
            fills = self.execute_market_order(order, market_data, execution_context)
        elif order.order_type == OrderType.STOP_LIMIT and order.price:
            fills = self.execute_limit_order(order, market_data, execution_context)

        # Update fill reason for all fills
        for fill in fills:
            fill.fill_reason = FillReason.STOP_TRIGGERED
        return fills
    def _is_stop_triggered(self, order: Order, market_data: MarketData) -> bool:
        """Helper method to check if stop order is triggered"""
        if order.side == OrderSide.BUY:

            # Buy stop: triggered when price goes above stop price
            return market_data.high >= order.stop_price
        else:

            # Sell stop: triggered when price goes below stop price
            return market_data.low <= order.stop_price
    def execute_order(self, order: Order, market_data: MarketData,
                     execution_context: dict[str, Any]) -> list[Fill]:
        """
        Main order execution entry point
        Args:
            order: Order to execute
            market_data: Current market data
            execution_context: Additional context (avg_volume, etc.)
        Returns:
            list[str] of fills generated from this execution attempt
        """
        if order.is_complete:
            return []
        if market_data.symbol != order.symbol:
            logger.warning(f"Symbol mismatch: order={order.symbol}, data={market_data.symbol}")
            return []
        try:
            if order.order_type == OrderType.MARKET:
                return self.execute_market_order(order, market_data, execution_context)
            elif order.order_type == OrderType.LIMIT:
                return self.execute_limit_order(order, market_data, execution_context)
            elif order.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
                return self.execute_stop_order(order, market_data, execution_context)
            else:
                logger.warning(f"Unsupported order type: {order.order_type}")
                return []
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            return []

# =====================================================================================
# EXECUTION ENGINE

# =====================================================================================
class ExecutionEngine:
    """
    Main execution engine that manages orders and coordinates with execution models
    """
    def __init__(self, execution_model: ExecutionModel) -> None:
        """
        Initialize execution engine
        Args:
            execution_model: Execution model to use for order processing
        """
        self.execution_model = execution_model
        self.pending_orders: dict[str, Order] = {}
        self.completed_orders: dict[str, Order] = {}
        self.execution_history: list[tuple[datetime, str, list[Fill]]] = []
        logger.info(f"Initialized ExecutionEngine with {type(execution_model).__name__}")
    def submit_order(self, order: Order) -> str:
        """
        Submit an order for execution
        Args:
            order: Order to submit
        Returns:
            Order ID
        """
        self.pending_orders[order.order_id] = order
        logger.debug(f"Submitted order {order.order_id}: {order.order_type.value} "
                    f"{order.side.value} {order.quantity} {order.symbol}")
        return order.order_id
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order
        Args:
            order_id: Order to cancel
        Returns:
            True if successfully cancelled
        """
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            self.completed_orders[order_id] = self.pending_orders.pop(order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        return False
    def process_market_data(self, market_data: MarketData,
                          execution_context: Optional[dict[str, Any]] = None) -> dict[str, list[Fill]]:
        """
        Process market data against all pending orders
        Args:
            market_data: Market data snapshot
            execution_context: Additional context for execution
        Returns:
            Dictionary mapping order_id to list[str] of fills
        """
        if execution_context is None:
            execution_context = {}
        all_fills = {}
        orders_to_remove = []

        # Process orders for this symbol
        for order_id, order in self.pending_orders.items():
            if order.symbol == market_data.symbol:
                fills = self.execution_model.execute_order(order, market_data, execution_context)
                if fills:
                    all_fills[order_id] = fills

                    # Add fills to order
                    for fill in fills:
                        order.add_fill(fill)

                    # Record execution history
                    self.execution_history.append((market_data.timestamp, order_id, fills))
                    logger.debug(f"Order {order_id} executed: {len(fills)} fills, "
                               f"filled {order.filled_quantity}/{order.quantity}")

                # Move completed orders
                if order.is_complete:
                    orders_to_remove.append(order_id)

        # Clean up completed orders
        for order_id in orders_to_remove:
            self.completed_orders[order_id] = self.pending_orders.pop(order_id)
        return all_fills
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current status of an order"""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id]
        return None
    def get_pending_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all pending orders, optionally filtered by symbol"""
        orders = list[str](self.pending_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of execution activity"""
        total_orders = len(self.completed_orders) + len(self.pending_orders)
        completed_orders = len(self.completed_orders)
        total_fills = sum(len(order.fills) for order in self.completed_orders.values())
        return {
            'total_orders_submitted': total_orders,
            'completed_orders': completed_orders,
            'pending_orders': len(self.pending_orders),
            'total_fills': total_fills,
            'execution_history_length': len(self.execution_history)
        }

# =====================================================================================
# FACTORY FUNCTIONS

# =====================================================================================
def create_indian_market_execution_engine(config: dict[str, Any]) -> ExecutionEngine:
    """
    Factory function to create execution engine configured for Indian markets
    Args:
        config: Configuration dictionary
    Returns:
        Configured ExecutionEngine
    """
    execution_model = IndianMarketExecutionModel(config)
    return ExecutionEngine(execution_model)
def create_execution_context(avg_volume: Optional[float] = None,
                           liquidity_tier: str = "normal",
                           market_hours: bool = True) -> dict[str, Any]:
    """
    Create execution context for order processing
    Args:
        avg_volume: Average daily trading volume
        liquidity_tier: Liquidity classification (high/normal/low)
        market_hours: Whether market is in regular trading hours
    Returns:
        Execution context dictionary
    """
    return {
        'avg_volume': avg_volume,
        'liquidity_tier': liquidity_tier,
        'market_hours': market_hours,
        'timestamp': datetime.now()
    }

# =====================================================================================
# EXAMPLE USAGE

# =====================================================================================
if __name__ == "__main__":

    # Example configuration for Indian markets
    config = {
        'brokerage_rate': 0.0003,
        'stt_rate': 0.00025,
        'slippage_model': 'adaptive',
        'base_slippage_bps': 2.0,
        'partial_fill_enabled': True,
        'random_seed': 42
    }

    # Create execution engine
    engine = create_indian_market_execution_engine(config)

    # Create sample order
    order = Order(
        symbol="RELIANCE.NS",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=100
    )

    # Submit order
    order_id = engine.submit_order(order)

    # Create sample market data
    market_data = MarketData(
        timestamp=datetime.now(),
        symbol="RELIANCE.NS",
        open_price=2500.0,
        high=2520.0,
        low=2495.0,
        close=2510.0,
        volume=1000000
    )

    # Process execution
    execution_context = create_execution_context(avg_volume=800000)
    fills = engine.process_market_data(market_data, execution_context)

    # Print results
    if order_id in fills:
        for fill in fills[order_id]:
            print(f"Fill: {fill.quantity} @ ₹{fill.price:.2f}, "
                  f"Commission: ₹{fill.commission:.2f}")
    print(f"Execution Summary: {engine.get_execution_summary()}")

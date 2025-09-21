#!/usr/bin/env python3
"""
NSE Stock Screener - Real-Time Event-Driven Automation
Automated triggers based on market events, news, and custom conditions
"""

import sys
import json
import time
import asyncio
import websockets
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import requests
from dataclasses import dataclass
import argparse
import signal

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from automation_manager import AutomationManager

@dataclass
class MarketEvent:
    """Represents a market event that can trigger automation"""
    event_type: str
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    severity: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    source: str = "UNKNOWN"

@dataclass
class TriggerCondition:
    """Defines conditions that trigger automated actions"""
    condition_id: str
    name: str
    condition_type: str  # price_movement, volume_spike, news_alert, technical_signal
    parameters: Dict[str, Any]
    action: str
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 30

class EventDrivenAutomation:
    """Real-time event-driven automation system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.project_root = PROJECT_ROOT
        self.config = self.load_config(config_file)
        self.automation_manager = AutomationManager()
        
        # Event processing
        self.event_queue = asyncio.Queue()
        self.event_history = []
        self.trigger_conditions = []
        self.active_monitors = {}
        
        # Control flags
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Load trigger conditions
        self.load_trigger_conditions()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_config(self, config_file: Optional[str]) -> Dict:
        """Load event-driven automation configuration"""
        default_config = {
            "event_monitoring": {
                "price_monitoring": {
                    "enabled": True,
                    "update_interval": 30,
                    "symbols": ["NIFTY", "SENSEX", "BANKNIFTY"]
                },
                "volume_monitoring": {
                    "enabled": True,
                    "volume_threshold_multiplier": 3.0,
                    "update_interval": 60
                },
                "news_monitoring": {
                    "enabled": True,
                    "sources": ["economic_times", "business_standard", "moneycontrol"],
                    "keywords": ["earnings", "merger", "acquisition", "results", "guidance"]
                },
                "technical_monitoring": {
                    "enabled": True,
                    "indicators": ["rsi_overbought", "rsi_oversold", "macd_crossover", "breakout"],
                    "update_interval": 300
                }
            },
            "trigger_thresholds": {
                "price_movement": {
                    "minor": 2.0,    # 2% movement
                    "major": 5.0,    # 5% movement
                    "extreme": 10.0  # 10% movement
                },
                "volume_spike": {
                    "minor": 2.0,    # 2x average volume
                    "major": 5.0,    # 5x average volume
                    "extreme": 10.0  # 10x average volume
                },
                "rsi_levels": {
                    "oversold": 30,
                    "overbought": 70
                }
            },
            "automation_actions": {
                "immediate_scan": True,
                "detailed_analysis": True,
                "alert_generation": True,
                "portfolio_review": True,
                "risk_assessment": True
            },
            "rate_limiting": {
                "max_events_per_minute": 100,
                "max_actions_per_hour": 50,
                "cooldown_between_same_events": 300
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
    
    def load_trigger_conditions(self):
        """Load predefined trigger conditions"""
        
        # Price movement triggers
        self.trigger_conditions.extend([
            TriggerCondition(
                condition_id="nifty_major_move",
                name="NIFTY Major Movement",
                condition_type="price_movement",
                parameters={"symbol": "^NSEI", "threshold": 2.0, "timeframe": "1d"},
                action="immediate_market_scan",
                cooldown_minutes=60
            ),
            TriggerCondition(
                condition_id="individual_stock_breakout",
                name="Individual Stock Breakout",
                condition_type="price_movement",
                parameters={"threshold": 5.0, "volume_confirmation": True},
                action="detailed_stock_analysis",
                cooldown_minutes=30
            )
        ])
        
        # Volume spike triggers
        self.trigger_conditions.extend([
            TriggerCondition(
                condition_id="volume_spike_alert",
                name="Volume Spike Alert",
                condition_type="volume_spike",
                parameters={"multiplier": 3.0, "min_price": 50},
                action="volume_breakout_analysis",
                cooldown_minutes=15
            )
        ])
        
        # Technical indicator triggers
        self.trigger_conditions.extend([
            TriggerCondition(
                condition_id="rsi_oversold_alert",
                name="RSI Oversold Alert",
                condition_type="technical_signal",
                parameters={"indicator": "rsi", "level": 30, "direction": "below"},
                action="oversold_opportunity_scan",
                cooldown_minutes=120
            ),
            TriggerCondition(
                condition_id="macd_bullish_crossover",
                name="MACD Bullish Crossover",
                condition_type="technical_signal",
                parameters={"indicator": "macd", "signal": "bullish_crossover"},
                action="momentum_opportunity_scan",
                cooldown_minutes=240
            )
        ])
        
        # News-based triggers
        self.trigger_conditions.extend([
            TriggerCondition(
                condition_id="earnings_announcement",
                name="Earnings Announcement",
                condition_type="news_alert",
                parameters={"keywords": ["earnings", "results", "quarterly"], "sentiment": "any"},
                action="earnings_impact_analysis",
                cooldown_minutes=60
            ),
            TriggerCondition(
                condition_id="merger_acquisition_news",
                name="M&A News Alert",
                condition_type="news_alert",
                parameters={"keywords": ["merger", "acquisition", "takeover"], "sentiment": "positive"},
                action="ma_opportunity_analysis",
                cooldown_minutes=30
            )
        ])
        
        print(f"[OK] Loaded {len(self.trigger_conditions)} trigger conditions")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.shutdown_event.set()
        self.stop_monitoring()
    
    async def start_monitoring(self):
        """Start real-time event monitoring"""
        print("[LAUNCH] Starting Event-Driven Automation...")
        self.is_running = True
        
        # Start different monitoring tasks
        tasks = []
        
        if self.config["event_monitoring"]["price_monitoring"]["enabled"]:
            tasks.append(self.monitor_price_movements())
        
        if self.config["event_monitoring"]["volume_monitoring"]["enabled"]:
            tasks.append(self.monitor_volume_spikes())
        
        if self.config["event_monitoring"]["news_monitoring"]["enabled"]:
            tasks.append(self.monitor_news_events())
        
        if self.config["event_monitoring"]["technical_monitoring"]["enabled"]:
            tasks.append(self.monitor_technical_signals())
        
        # Start event processor
        tasks.append(self.process_events())
        
        print(f"[OK] Started {len(tasks)} monitoring tasks")
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("📱 Monitoring tasks cancelled")
        except Exception as e:
            print(f"[ERROR] Monitoring error: {e}")
    
    async def monitor_price_movements(self):
        """Monitor real-time price movements"""
        print("[CHART] Starting price movement monitoring...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                symbols = self.config["event_monitoring"]["price_monitoring"]["symbols"]
                
                for symbol in symbols:
                    await self.check_price_movement(symbol)
                    
                    if self.shutdown_event.is_set():
                        break
                
                # Wait for next update cycle
                interval = self.config["event_monitoring"]["price_monitoring"]["update_interval"]
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"[ERROR] Price monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def check_price_movement(self, symbol: str):
        """Check price movement for a specific symbol"""
        try:
            # Simulate price data retrieval (in production, use real API)
            current_price = 18500.0  # Simulated NIFTY price
            previous_price = 18200.0  # Simulated previous price
            
            price_change_pct = ((current_price - previous_price) / previous_price) * 100
            
            # Check against thresholds
            thresholds = self.config["trigger_thresholds"]["price_movement"]
            
            severity = "LOW"
            if abs(price_change_pct) >= thresholds["extreme"]:
                severity = "CRITICAL"
            elif abs(price_change_pct) >= thresholds["major"]:
                severity = "HIGH"
            elif abs(price_change_pct) >= thresholds["minor"]:
                severity = "MEDIUM"
            
            if severity != "LOW":
                event = MarketEvent(
                    event_type="price_movement",
                    symbol=symbol,
                    timestamp=datetime.now(),
                    data={
                        "current_price": current_price,
                        "previous_price": previous_price,
                        "change_percent": price_change_pct,
                        "direction": "up" if price_change_pct > 0 else "down"
                    },
                    severity=severity,
                    source="price_monitor"
                )
                
                await self.event_queue.put(event)
                print(f"[ANALYSIS] Price event: {symbol} {price_change_pct:+.2f}% ({severity})")
        
        except Exception as e:
            print(f"[ERROR] Price check error for {symbol}: {e}")
    
    async def monitor_volume_spikes(self):
        """Monitor volume spikes across stocks"""
        print("[ANALYSIS] Starting volume spike monitoring...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                await self.scan_volume_spikes()
                
                interval = self.config["event_monitoring"]["volume_monitoring"]["update_interval"]
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"[ERROR] Volume monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def scan_volume_spikes(self):
        """Scan for volume spikes"""
        try:
            # Simulate volume spike detection
            volume_spikes = [
                {"symbol": "RELIANCE", "volume_ratio": 4.2, "price_change": 3.1},
                {"symbol": "HDFCBANK", "volume_ratio": 3.8, "price_change": -1.5},
            ]
            
            threshold = self.config["event_monitoring"]["volume_monitoring"]["volume_threshold_multiplier"]
            
            for spike in volume_spikes:
                if spike["volume_ratio"] >= threshold:
                    event = MarketEvent(
                        event_type="volume_spike",
                        symbol=spike["symbol"],
                        timestamp=datetime.now(),
                        data=spike,
                        severity="HIGH" if spike["volume_ratio"] >= 5.0 else "MEDIUM",
                        source="volume_monitor"
                    )
                    
                    await self.event_queue.put(event)
                    print(f"[CHART] Volume spike: {spike['symbol']} ({spike['volume_ratio']:.1f}x)")
        
        except Exception as e:
            print(f"[ERROR] Volume spike scan error: {e}")
    
    async def monitor_news_events(self):
        """Monitor news events and alerts"""
        print("📰 Starting news monitoring...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                await self.scan_news_events()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                print(f"[ERROR] News monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def scan_news_events(self):
        """Scan for relevant news events"""
        try:
            # Simulate news event detection
            news_events = [
                {
                    "title": "RELIANCE announces Q3 earnings beat",
                    "symbols": ["RELIANCE"],
                    "sentiment": "positive",
                    "keywords": ["earnings", "beat"]
                },
                {
                    "title": "TCS bags major US contract",
                    "symbols": ["TCS"],
                    "sentiment": "positive",
                    "keywords": ["contract", "deal"]
                }
            ]
            
            for news in news_events:
                for symbol in news["symbols"]:
                    event = MarketEvent(
                        event_type="news_alert",
                        symbol=symbol,
                        timestamp=datetime.now(),
                        data=news,
                        severity="MEDIUM",
                        source="news_monitor"
                    )
                    
                    await self.event_queue.put(event)
                    print(f"📰 News event: {symbol} - {news['title'][:50]}...")
        
        except Exception as e:
            print(f"[ERROR] News scan error: {e}")
    
    async def monitor_technical_signals(self):
        """Monitor technical indicator signals"""
        print("[CONFIG] Starting technical signal monitoring...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                await self.scan_technical_signals()
                
                interval = self.config["event_monitoring"]["technical_monitoring"]["update_interval"]
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"[ERROR] Technical monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def scan_technical_signals(self):
        """Scan for technical indicator signals"""
        try:
            # Simulate technical signal detection
            signals = [
                {"symbol": "INFY", "indicator": "rsi", "value": 28, "signal": "oversold"},
                {"symbol": "ICICIBANK", "indicator": "macd", "signal": "bullish_crossover"},
            ]
            
            for signal in signals:
                event = MarketEvent(
                    event_type="technical_signal",
                    symbol=signal["symbol"],
                    timestamp=datetime.now(),
                    data=signal,
                    severity="MEDIUM",
                    source="technical_monitor"
                )
                
                await self.event_queue.put(event)
                print(f"[CONFIG] Technical signal: {signal['symbol']} - {signal['indicator']} {signal['signal']}")
        
        except Exception as e:
            print(f"[ERROR] Technical scan error: {e}")
    
    async def process_events(self):
        """Process events from the queue and trigger actions"""
        print("⚙️ Starting event processor...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=5.0)
                    await self.handle_event(event)
                except asyncio.TimeoutError:
                    continue  # No events, continue monitoring
                
            except Exception as e:
                print(f"[ERROR] Event processing error: {e}")
                await asyncio.sleep(1)
    
    async def handle_event(self, event: MarketEvent):
        """Handle a specific market event"""
        try:
            print(f"[TARGET] Processing event: {event.event_type} for {event.symbol}")
            
            # Add to history
            self.event_history.append(event)
            
            # Check trigger conditions
            triggered_conditions = self.check_trigger_conditions(event)
            
            for condition in triggered_conditions:
                if self.should_trigger_action(condition):
                    await self.execute_trigger_action(condition, event)
                    condition.last_triggered = datetime.now()
        
        except Exception as e:
            print(f"[ERROR] Event handling error: {e}")
    
    def check_trigger_conditions(self, event: MarketEvent) -> List[TriggerCondition]:
        """Check which trigger conditions match the event"""
        triggered_conditions = []
        
        for condition in self.trigger_conditions:
            if not condition.enabled:
                continue
            
            if self.condition_matches_event(condition, event):
                triggered_conditions.append(condition)
        
        return triggered_conditions
    
    def condition_matches_event(self, condition: TriggerCondition, event: MarketEvent) -> bool:
        """Check if a condition matches an event"""
        
        # Match by event type
        if condition.condition_type != event.event_type:
            return False
        
        # Check specific parameters
        params = condition.parameters
        
        if condition.condition_type == "price_movement":
            required_threshold = params.get("threshold", 2.0)
            actual_change = abs(event.data.get("change_percent", 0))
            return actual_change >= required_threshold
        
        elif condition.condition_type == "volume_spike":
            required_multiplier = params.get("multiplier", 3.0)
            actual_ratio = event.data.get("volume_ratio", 0)
            return actual_ratio >= required_multiplier
        
        elif condition.condition_type == "technical_signal":
            required_indicator = params.get("indicator")
            actual_indicator = event.data.get("indicator")
            return required_indicator == actual_indicator
        
        elif condition.condition_type == "news_alert":
            required_keywords = params.get("keywords", [])
            actual_keywords = event.data.get("keywords", [])
            return any(keyword in actual_keywords for keyword in required_keywords)
        
        return False
    
    def should_trigger_action(self, condition: TriggerCondition) -> bool:
        """Check if an action should be triggered based on cooldown"""
        if condition.last_triggered is None:
            return True
        
        cooldown_delta = timedelta(minutes=condition.cooldown_minutes)
        return datetime.now() - condition.last_triggered >= cooldown_delta
    
    async def execute_trigger_action(self, condition: TriggerCondition, event: MarketEvent):
        """Execute the action for a triggered condition"""
        try:
            print(f"[LAUNCH] Executing action: {condition.action} for {event.symbol}")
            
            if condition.action == "immediate_market_scan":
                result = await self.run_immediate_market_scan(event)
            elif condition.action == "detailed_stock_analysis":
                result = await self.run_detailed_stock_analysis(event)
            elif condition.action == "volume_breakout_analysis":
                result = await self.run_volume_breakout_analysis(event)
            elif condition.action == "oversold_opportunity_scan":
                result = await self.run_oversold_opportunity_scan(event)
            elif condition.action == "momentum_opportunity_scan":
                result = await self.run_momentum_opportunity_scan(event)
            elif condition.action == "earnings_impact_analysis":
                result = await self.run_earnings_impact_analysis(event)
            elif condition.action == "ma_opportunity_analysis":
                result = await self.run_ma_opportunity_analysis(event)
            else:
                print(f"[WARNING] Unknown action: {condition.action}")
                return
            
            print(f"[OK] Action completed: {condition.action}")
            print(f"   Result: {result.get('summary', 'No summary available')}")
        
        except Exception as e:
            print(f"[ERROR] Action execution error: {e}")
    
    async def run_immediate_market_scan(self, event: MarketEvent) -> Dict:
        """Run immediate market-wide scan"""
        # Simulate immediate analysis
        await asyncio.sleep(2)  # Simulate processing time
        
        return {
            "action": "immediate_market_scan",
            "trigger_event": event.event_type,
            "symbol": event.symbol,
            "scan_time": datetime.now().isoformat(),
            "stocks_analyzed": 250,
            "opportunities_found": 12,
            "summary": f"Found 12 opportunities after {event.symbol} movement"
        }
    
    async def run_detailed_stock_analysis(self, event: MarketEvent) -> Dict:
        """Run detailed analysis on specific stock"""
        await asyncio.sleep(1)
        
        return {
            "action": "detailed_stock_analysis",
            "symbol": event.symbol,
            "analysis_time": datetime.now().isoformat(),
            "technical_score": 75,
            "risk_rating": "MEDIUM",
            "recommendation": "MONITOR",
            "summary": f"Detailed analysis completed for {event.symbol}"
        }
    
    async def run_volume_breakout_analysis(self, event: MarketEvent) -> Dict:
        """Analyze volume breakout opportunities"""
        await asyncio.sleep(1.5)
        
        return {
            "action": "volume_breakout_analysis",
            "symbol": event.symbol,
            "volume_ratio": event.data.get("volume_ratio", 0),
            "breakout_strength": "STRONG",
            "followup_recommendation": "IMMEDIATE_WATCH",
            "summary": f"Volume breakout analysis for {event.symbol}"
        }
    
    async def run_oversold_opportunity_scan(self, event: MarketEvent) -> Dict:
        """Scan for oversold opportunities"""
        await asyncio.sleep(2)
        
        return {
            "action": "oversold_opportunity_scan",
            "trigger_symbol": event.symbol,
            "oversold_stocks_found": 8,
            "quality_opportunities": 3,
            "recommendation": "REVIEW_LIST",
            "summary": "Found 3 quality oversold opportunities"
        }
    
    async def run_momentum_opportunity_scan(self, event: MarketEvent) -> Dict:
        """Scan for momentum opportunities"""
        await asyncio.sleep(1.5)
        
        return {
            "action": "momentum_opportunity_scan",
            "trigger_symbol": event.symbol,
            "momentum_stocks_found": 15,
            "strong_momentum": 6,
            "recommendation": "MOMENTUM_WATCH",
            "summary": "Found 6 strong momentum opportunities"
        }
    
    async def run_earnings_impact_analysis(self, event: MarketEvent) -> Dict:
        """Analyze earnings impact"""
        await asyncio.sleep(1)
        
        return {
            "action": "earnings_impact_analysis",
            "symbol": event.symbol,
            "earnings_sentiment": event.data.get("sentiment", "neutral"),
            "sector_impact": "POSITIVE",
            "related_stocks_analyzed": 12,
            "summary": f"Earnings impact analysis for {event.symbol}"
        }
    
    async def run_ma_opportunity_analysis(self, event: MarketEvent) -> Dict:
        """Analyze M&A opportunities"""
        await asyncio.sleep(1)
        
        return {
            "action": "ma_opportunity_analysis",
            "symbol": event.symbol,
            "ma_type": "acquisition",
            "sector_analysis": "POSITIVE",
            "related_opportunities": 5,
            "summary": f"M&A opportunity analysis for {event.symbol}"
        }
    
    def stop_monitoring(self):
        """Stop all monitoring"""
        print("⏹️ Stopping event-driven automation...")
        self.is_running = False
    
    def get_status(self) -> Dict:
        """Get current automation status"""
        return {
            "running": self.is_running,
            "active_conditions": len([c for c in self.trigger_conditions if c.enabled]),
            "events_processed": len(self.event_history),
            "recent_events": len([e for e in self.event_history 
                                if e.timestamp > datetime.now() - timedelta(hours=1)])
        }
    
    def print_status(self):
        """Print current status"""
        status = self.get_status()
        
        print("\n[TARGET] EVENT-DRIVEN AUTOMATION STATUS")
        print("=" * 45)
        print(f"Status: {'🟢 Running' if status['running'] else '🔴 Stopped'}")
        print(f"Active Conditions: {status['active_conditions']}")
        print(f"Events Processed: {status['events_processed']}")
        print(f"Recent Events (1h): {status['recent_events']}")
        
        if self.event_history:
            print(f"\n[LIST] Recent Events:")
            recent_events = sorted(self.event_history, key=lambda x: x.timestamp, reverse=True)[:5]
            for i, event in enumerate(recent_events, 1):
                print(f"  {i}. {event.event_type} - {event.symbol} ({event.severity})")

def main():
    """Main entry point for event-driven automation"""
    parser = argparse.ArgumentParser(description='Event-Driven Real-Time Automation')
    
    parser.add_argument('--start', action='store_true',
                        help='Start event monitoring')
    parser.add_argument('--status', action='store_true',
                        help='Show current status')
    parser.add_argument('--config', type=str,
                        help='Configuration file path')
    parser.add_argument('--test-event', type=str,
                        help='Test with a simulated event type')
    
    args = parser.parse_args()
    
    try:
        # Create automation system
        automation = EventDrivenAutomation(config_file=args.config)
        
        if args.status:
            automation.print_status()
            return
        
        if args.test_event:
            print(f"🧪 Testing with simulated {args.test_event} event...")
            # Implementation for testing specific event types
            return
        
        if args.start:
            print("[LAUNCH] Starting Event-Driven Automation...")
            try:
                asyncio.run(automation.start_monitoring())
            except KeyboardInterrupt:
                print("\n⏹️ Shutting down...")
                automation.stop_monitoring()
        else:
            print("Use --start to begin monitoring or --status to check status")
            parser.print_help()
    
    except Exception as e:
        print(f"[ERROR] Event automation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
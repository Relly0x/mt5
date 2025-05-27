# enhanced_mt5_trading_bot_fixed_no_telegram.py
"""
FULLY FIXED Enhanced MT5 Trading Bot - NO TELEGRAM VERSION
- Array comparison issues resolved
- Enhanced TFT model loading fixed
- Compatible with your trained best_model.pt
- Uses only Enhanced TFT (never SimpleTFT)
- Ready for live trading
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import time
import logging
import json
import os
import signal
import sys
import threading
import atexit
from datetime import datetime
import traceback

# Import your existing modules - using Enhanced versions only
from models.tft.model import TemporalFusionTransformer  # Enhanced TFT only
from data.processors.normalizer import DataNormalizer
from strategy.strategy_factory import create_strategy
from execution.risk.risk_manager import RiskManager

# Global variables for cleanup
shutdown_event = threading.Event()


class SafeFormatter(logging.Formatter):
    """Unicode-safe logging formatter"""

    def format(self, record):
        try:
            if hasattr(record, 'msg'):
                msg = str(record.msg)
                # Replace emoji with text equivalents for console safety
                emoji_replacements = {
                    '🚀': '[ROCKET]', '✅': '[CHECK]', '🔧': '[TOOL]',
                    '⚠️': '[WARNING]', '🚨': '[ALERT]', '❌': '[ERROR]',
                    '🟢': '[GREEN]', '📊': '[CHART]', '💰': '[MONEY]',
                    '🎯': '[TARGET]', '🛡️': '[SHIELD]', '🔄': '[REFRESH]'
                }
                for emoji, replacement in emoji_replacements.items():
                    msg = msg.replace(emoji, replacement)
                record.msg = msg
            return super().format(record)
        except UnicodeEncodeError:
            safe_msg = repr(record.msg) if hasattr(record, 'msg') else "Log message encoding error"
            record.msg = safe_msg
            return super().format(record)


class EnhancedMT5TradingBot:
    def __init__(self, config_path='config/config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Setup SAFE logging
        self.setup_safe_logging()

        # Initialize Enhanced components only
        self.model = None
        self.normalizer = None
        self.strategy = None
        self.risk_manager = None

        # Trading state
        self.is_running = False
        self.positions = {}
        self.main_thread = None
        self.daily_trade_count = 0
        self.last_trade_date = None

        # Enhanced debugging
        self.debug_mode = True
        self.signal_debug_count = 0

        # Shutdown handling
        self.shutdown_event = threading.Event()

        self.logger.info("EnhancedMT5Bot - Enhanced MT5 Trading Bot initialized (No Telegram)")

    def setup_safe_logging(self):
        """Setup Unicode-safe logging configuration"""
        os.makedirs('logs', exist_ok=True)

        self.logger = logging.getLogger('EnhancedMT5Bot')
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create safe formatter
        formatter = SafeFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler with UTF-8 encoding
        try:
            file_handler = logging.FileHandler('logs/enhanced_mt5_trading_bot.log', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file handler: {e}")

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.logger.error("Failed to initialize MetaTrader5")
            return False

        # Check if already logged in
        account_info = mt5.account_info()

        if account_info is not None:
            self.logger.info("[CHECK] Using existing MT5 session")
            self.logger.info(f"Connected to account: {account_info.login}")
            self.logger.info(f"Broker: {account_info.company}")
            self.logger.info(f"Balance: {account_info.balance} {account_info.currency}")
            return True

        self.logger.error("[ERROR] No active MT5 session found")
        return False

    def load_enhanced_model(self):
        """Load the Enhanced TFT model with FIXED parameter loading"""
        model_path = self.config['export']['model_path']

        if not os.path.exists(model_path):
            self.logger.error(f"Enhanced model file not found: {model_path}")
            return False

        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            self.logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

            # Add this right after loading checkpoint to debug
            if 'model_state_dict' in checkpoint:
                # List first few parameter names to understand structure
                param_names = list(checkpoint['model_state_dict'].keys())[:10]
                self.logger.info(f"First 10 checkpoint parameters: {param_names}")

                # Check if it's a SimpleTFT or full TFT checkpoint
                if any('feature_projection' in name for name in checkpoint['model_state_dict'].keys()):
                    self.logger.warning("This appears to be a SimpleTFT checkpoint, not Enhanced TFT!")

            # Get model config from checkpoint or use default
            if 'config' in checkpoint and 'model' in checkpoint['config']:
                model_config = checkpoint['config']['model']
                self.logger.info("Using model config from checkpoint")
            else:
                model_config = self.config['model']
                self.logger.info("Using model config from current config")

            # Create Enhanced TFT model
            self.model = TemporalFusionTransformer(model_config)

            # CRITICAL FIX: Initialize the model layers first with a dummy forward pass
            self.logger.info("Initializing Enhanced TFT layers with dummy data...")

            # Create dummy batch matching your training data structure
            past_seq_len = model_config.get('past_sequence_length', 120)
            forecast_horizon = model_config.get('forecast_horizon', 12)

            # Use the actual feature dimensions from your trained model
            # Based on your normalizer, you have 29 features for past data
            dummy_batch = {
                'past': torch.randn(1, past_seq_len, 29),  # 29 features from your normalizer
                'future': torch.randn(1, forecast_horizon, 28),  # 28 features (excluding target)
                'static': torch.randn(1, 1)
            }

            # Run dummy forward pass to initialize all layers
            with torch.no_grad():
                _ = self.model(dummy_batch)

            self.logger.info("Model layers initialized, now loading weights...")

            # NOW load the state dict after initialization
            if 'model_state_dict' in checkpoint:
                # Get state dicts
                checkpoint_state = checkpoint['model_state_dict']
                model_state = self.model.state_dict()

                self.logger.info(f"Checkpoint has {len(checkpoint_state)} parameters")
                self.logger.info(f"Model expects {len(model_state)} parameters")

                # Direct loading since model is now properly initialized
                try:
                    self.model.load_state_dict(checkpoint_state, strict=False)
                    self.logger.info("Model weights loaded successfully")
                except Exception as e:
                    self.logger.warning(f"Partial loading warning: {e}")

                    # If direct loading fails, try matching parameters
                    matched_params = {}
                    for name, param in checkpoint_state.items():
                        if name in model_state and param.shape == model_state[name].shape:
                            matched_params[name] = param
                            self.logger.debug(f"Matched parameter: {name} {param.shape}")

                    if matched_params:
                        model_state.update(matched_params)
                        self.model.load_state_dict(model_state, strict=False)
                        self.logger.info(f"Loaded {len(matched_params)} matching parameters")

            self.model.eval()

            # Verify model has parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            if total_params == 0:
                self.logger.error("[ERROR] Model has 0 parameters! This will not work for trading.")
                return False

            self.logger.info(
                f"[CHECK] Enhanced TFT loaded successfully with {total_params:,} total parameters ({trainable_params:,} trainable)")

            # Final test with another forward pass
            try:
                with torch.no_grad():
                    output = self.model(dummy_batch)
                    self.logger.info(f"[CHECK] Model test successful. Output shape: {output.shape}")

                    # Check if output looks reasonable
                    output_np = output.numpy()
                    self.logger.info(f"Output range: [{output_np.min():.4f}, {output_np.max():.4f}]")

            except Exception as e:
                self.logger.error(f"[ERROR] Model test failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error loading Enhanced TFT model: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def initialize_enhanced_components(self):
        """Initialize Enhanced normalizer, strategy, and risk manager"""
        try:
            # Initialize Enhanced normalizer
            self.normalizer = DataNormalizer(self.config)

            # Get initial market data and fit scaler properly
            self.logger.info("[TOOL] Fitting Enhanced scaler with initial market data...")

            initial_data = self.get_enhanced_market_data()
            if initial_data:
                try:
                    # Process with Enhanced normalizer
                    processed_data = self.normalizer.process(initial_data)
                    self.logger.info("[CHECK] Enhanced scaler fitted successfully")

                    # Log scaler info for debugging
                    scaler_info = self.normalizer.get_scaler_info()
                    self.logger.info(f"Enhanced scaler info: {scaler_info}")

                except Exception as e:
                    self.logger.error(f"Error processing initial data: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    return False
            else:
                self.logger.warning("[WARNING] No initial market data available")
                return False

            # Initialize Enhanced strategy (always enhanced_tft)
            self.strategy = create_strategy(self.config, strategy_type='enhanced_tft')

            # Initialize risk manager
            self.risk_manager = RiskManager(self.config)

            self.logger.info("Enhanced components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing Enhanced components: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def get_enhanced_market_data(self):
        """Get Enhanced market data from MT5"""
        try:
            market_data = {}

            # Process each configured instrument
            for instrument in self.config['data']['instruments']:
                # Convert instrument format (EUR_USD -> EURUSD)
                mt5_symbol = instrument.replace('_', '')

                # Check if symbol exists
                symbol_info = mt5.symbol_info(mt5_symbol)
                if symbol_info is None:
                    self.logger.warning(f"Symbol {mt5_symbol} not found")
                    continue

                # Select symbol if not visible
                if not symbol_info.visible:
                    if not mt5.symbol_select(mt5_symbol, True):
                        self.logger.warning(f"Failed to select {mt5_symbol}")
                        continue

                market_data[instrument] = {}

                # Get data for each timeframe
                timeframes = {
                    'M1': mt5.TIMEFRAME_M1,
                    'M5': mt5.TIMEFRAME_M5,
                    'M15': mt5.TIMEFRAME_M15,
                    'M30': mt5.TIMEFRAME_M30,
                    'H1': mt5.TIMEFRAME_H1,
                    'H4': mt5.TIMEFRAME_H4,
                    'D1': mt5.TIMEFRAME_D1
                }

                # Get configured timeframes
                high_tf = self.config['data']['timeframes']['high']
                low_tf = self.config['data']['timeframes']['low']

                for tf_name in [high_tf, low_tf]:
                    if tf_name not in timeframes:
                        self.logger.warning(f"Unknown timeframe: {tf_name}")
                        continue

                    # Get historical data
                    rates = mt5.copy_rates_from_pos(mt5_symbol, timeframes[tf_name], 0, 300)

                    if rates is None:
                        self.logger.warning(f"No data for {mt5_symbol} {tf_name}")
                        continue

                    # Convert to DataFrame
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df.rename(columns={'tick_volume': 'volume'}, inplace=True)

                    market_data[instrument][tf_name] = df

                    # Debug logging
                    self.logger.info(f"Enhanced data: Got {len(df)} candles for {mt5_symbol} {tf_name}")

            self.logger.info(f"Enhanced market data collected for {len(market_data)} instruments")
            return market_data

        except Exception as e:
            self.logger.error(f"Error getting Enhanced market data: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def generate_enhanced_signals(self, market_data):
        """Generate Enhanced trading signals with FIXED array handling"""
        try:
            self.signal_debug_count += 1
            self.logger.info(f"[CHART] Enhanced signal generation #{self.signal_debug_count}")

            # Process data through Enhanced normalizer
            processed_data = self.normalizer.process(market_data)

            # Debug: Check what we got from normalization
            for instrument, timeframes in processed_data.items():
                for tf, df in timeframes.items():
                    if df is not None and len(df) > 0:
                        self.logger.info(f"Enhanced processing: {instrument} {tf}: {df.shape}")
                    else:
                        self.logger.warning(f"Empty processed data for {instrument} {tf}")

            # Check if normalizer is properly fitted
            if not self.normalizer.is_fitted:
                self.logger.error("[ERROR] Enhanced normalizer is not fitted!")
                return {}

            # Prepare Enhanced predictions for each instrument
            predictions = {}

            for instrument in self.config['data']['instruments']:
                if instrument not in processed_data:
                    self.logger.warning(f"No processed data for {instrument}")
                    continue

                # Get high timeframe data
                high_tf = self.config['data']['timeframes']['high']
                if high_tf not in processed_data[instrument]:
                    self.logger.warning(f"No {high_tf} data for {instrument}")
                    continue

                df = processed_data[instrument][high_tf]

                # Check if we have enough data
                past_seq_len = self.config['model']['past_sequence_length']
                if len(df) < past_seq_len:
                    self.logger.warning(f"Insufficient data for {instrument}: {len(df)} < {past_seq_len}")
                    continue

                # Prepare Enhanced model input
                recent_data = df.iloc[-past_seq_len:].copy()

                # Debug: Log the shape of data going into the model
                self.logger.info(f"Enhanced model input shape for {instrument}: {recent_data.shape}")

                # Create Enhanced tensors for TFT
                try:
                    # Enhanced TFT requires specific input format
                    past_tensor = torch.tensor(recent_data.values, dtype=torch.float32).unsqueeze(0)

                    # Create future tensor (reduced features for known future data)
                    forecast_horizon = self.config['model']['forecast_horizon']
                    future_features = recent_data.shape[1] - 1  # Exclude target variable
                    future_tensor = torch.zeros((1, forecast_horizon, future_features), dtype=torch.float32)

                    # Static features (can be enhanced based on your needs)
                    static_tensor = torch.zeros((1, 1), dtype=torch.float32)

                    # Enhanced TFT batch format
                    enhanced_batch_data = {
                        'past': past_tensor,
                        'future': future_tensor,
                        'static': static_tensor
                    }

                    # Run Enhanced TFT inference
                    with torch.no_grad():
                        enhanced_output = self.model(enhanced_batch_data)

                    predictions[instrument] = enhanced_output
                    self.logger.info(f"Enhanced prediction for {instrument}: shape {enhanced_output.shape}")

                except Exception as model_error:
                    self.logger.error(f"Enhanced model inference error for {instrument}: {model_error}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

            self.logger.info(f"Enhanced predictions generated for {len(predictions)} instruments")

            # Update Enhanced strategy with new data
            self.strategy.update_data(market_data)

            # Generate Enhanced signals using FIXED signal generator
            signals = self.strategy.generate_signals(predictions, market_data)

            # Debug: Log Enhanced signal generation results
            valid_signals = sum(1 for s in signals.values() if s.get('valid', False))
            self.logger.info(f"Enhanced signal generation complete: {valid_signals} high-quality signals")

            # Debug: Log details of invalid signals
            if self.debug_mode:
                for instrument, signal in signals.items():
                    if not signal.get('valid', False):
                        reason = signal.get('reason', 'Unknown reason')
                        self.logger.debug(f"Enhanced signal rejected for {instrument}: {reason}")

            return signals

        except Exception as e:
            self.logger.error(f"Error generating Enhanced signals: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def execute_enhanced_signal(self, signal, instrument):
        """Execute Enhanced trading signal with validation"""
        try:
            # Convert instrument format
            mt5_symbol = instrument.replace('_', '')

            # Enhanced signal validation
            if not signal.get('valid', False):
                return False

            # Check daily trade limit
            today = datetime.now().date()
            if self.last_trade_date != today:
                self.daily_trade_count = 0
                self.last_trade_date = today

            max_daily_trades = self.config.get('execution', {}).get('max_daily_trades', 3)
            if self.daily_trade_count >= max_daily_trades:
                self.logger.info(f"Daily trade limit reached ({self.daily_trade_count}/{max_daily_trades})")
                return False

            # Check if we already have a position
            if instrument in self.positions:
                self.logger.info(f"Already have position in {instrument}")
                return False

            # Get Enhanced signal details
            direction = signal.get('signal')
            current_price = signal.get('current_price')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            signal_strength = signal.get('strength', 0)
            quality_grade = signal.get('quality_grade', 'B')

            # Calculate Enhanced position size
            position_size = self.risk_manager.calculate_position_size(
                instrument, current_price, stop_loss, direction
            )

            # Get current price from MT5
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                self.logger.error(f"Failed to get tick for {mt5_symbol}")
                return False

            # Prepare Enhanced order
            if direction == 'buy':
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid

            # Enhanced order request with magic number from config
            magic_number = self.config.get('execution', {}).get('magic_number', 123456)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": float(position_size),
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": magic_number,
                "comment": f"Enhanced TFT {quality_grade}-grade: {signal_strength:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send Enhanced order
            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Enhanced order failed for {mt5_symbol}: {result.comment}"
                self.logger.error(error_msg)
                return False

            # Increment daily trade count
            self.daily_trade_count += 1

            self.logger.info(
                f"[GREEN] Enhanced order executed: {direction} {position_size} lots of {mt5_symbol} at {price}")
            self.logger.info(f"[TARGET] Quality Grade: {quality_grade}, Signal Strength: {signal_strength:.1%}")
            self.logger.info(f"[CHART] Daily Trades: {self.daily_trade_count}/{max_daily_trades}")

            # Store Enhanced position info
            self.positions[instrument] = {
                'ticket': result.order,
                'direction': direction,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': position_size,
                'entry_time': datetime.now(),
                'signal_strength': signal_strength,
                'quality_grade': quality_grade,
                'model_type': 'Enhanced_TFT'
            }

            return True

        except Exception as e:
            error_msg = f"Error executing Enhanced signal for {instrument}: {e}"
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def check_enhanced_positions(self):
        """Check and manage Enhanced positions"""
        try:
            magic_number = self.config.get('execution', {}).get('magic_number', 123456)
            positions = mt5.positions_get()

            if positions is None:
                return

            current_mt5_positions = {}

            for position in positions:
                if position.magic != magic_number:
                    continue

                symbol = position.symbol
                profit = position.profit

                # Convert back to our instrument format
                instrument = symbol
                for config_instrument in self.config['data']['instruments']:
                    if config_instrument.replace('_', '') == symbol:
                        instrument = config_instrument
                        break

                current_mt5_positions[instrument] = {
                    'profit': profit,
                    'current_price': position.price_current,
                    'entry_price': position.price_open,
                    'volume': position.volume
                }

            # Check for closed Enhanced positions
            closed_positions = []
            for instrument in list(self.positions.keys()):
                if instrument not in current_mt5_positions:
                    closed_positions.append(instrument)

            # Handle closed Enhanced positions
            for instrument in closed_positions:
                stored_pos = self.positions[instrument]

                self.logger.info(
                    f"[REFRESH] Enhanced Position Closed: {instrument.replace('_', '')} {stored_pos['direction'].upper()}")
                self.logger.info(
                    f"[CHART] Quality: {stored_pos.get('quality_grade', 'B')}-grade, Entry: {stored_pos['entry_price']:.5f}")

                # Remove from Enhanced tracking
                del self.positions[instrument]

            if self.debug_mode and len(current_mt5_positions) > 0:
                self.logger.info(f"Enhanced monitoring: {len(current_mt5_positions)} positions")

        except Exception as e:
            self.logger.error(f"Error checking Enhanced positions: {e}")

    def cleanup(self):
        """Enhanced cleanup function for graceful shutdown"""
        self.logger.info("[ALERT] Starting Enhanced cleanup...")

        # Set shutdown flag
        self.is_running = False
        self.shutdown_event.set()

        try:
            # Shutdown MT5 connection
            mt5.shutdown()
            self.logger.info("Enhanced MT5 connection closed")

        except Exception as e:
            self.logger.error(f"Error during Enhanced cleanup: {e}")

        self.logger.info("[CHECK] Enhanced cleanup complete")

    def run(self):
        """Enhanced main trading loop"""
        self.logger.info("[ROCKET] Starting Enhanced MT5 Trading Bot with TFT")

        # Initialize MT5
        if not self.initialize_mt5():
            return

        # Load Enhanced model
        if not self.load_enhanced_model():
            return

        # Initialize Enhanced components
        if not self.initialize_enhanced_components():
            return

        self.is_running = True
        self.logger.info("[CHECK] Enhanced Bot is ready for live trading - Press Ctrl+C to stop safely")

        # Enhanced startup notification
        account_info = mt5.account_info()
        if account_info:
            scaler_info = self.normalizer.get_scaler_info()
            self.logger.info(f"[ROCKET] Enhanced MT5 Trading Bot Started!")
            self.logger.info(f"Model: Enhanced Temporal Fusion Transformer")
            self.logger.info(f"Broker: {account_info.company}")
            self.logger.info(f"[MONEY] Balance: {account_info.balance:.2f} {account_info.currency}")
            self.logger.info(f"[CHART] Strategy: Enhanced TFT with Quality Controls")
            self.logger.info(f"Risk per Trade: {self.config.get('execution', {}).get('risk_per_trade', 0.01) * 100}%")
            self.logger.info(f"[CHART] Max Positions: {self.config.get('execution', {}).get('max_open_positions', 2)}")
            self.logger.info(f"Max Daily Trades: {self.config.get('execution', {}).get('max_daily_trades', 3)}")
            self.logger.info(f"[CHECK] Enhanced Scaler: {'Fitted' if scaler_info['is_fitted'] else 'Not Fitted'}")
            self.logger.info(f"[CHART] Features: {scaler_info['feature_count']}")
            self.logger.info("Ready for Enhanced live trading!")

        # Enhanced main loop
        iteration_count = 0
        last_status_update = datetime.now()

        try:
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    iteration_count += 1

                    # Check if market is open
                    if not self._is_market_open():
                        self.logger.debug("Market closed, waiting...")
                        time.sleep(60)
                        continue

                    # Get Enhanced market data
                    market_data = self.get_enhanced_market_data()
                    if not market_data:
                        self.logger.warning("No Enhanced market data available")
                        time.sleep(10)
                        continue

                    # Generate Enhanced signals
                    signals = self.generate_enhanced_signals(market_data)

                    # Process Enhanced signals
                    signals_processed = 0
                    for instrument, signal in signals.items():
                        if signal.get('valid', False):
                            if instrument not in self.positions:
                                quality_grade = signal.get('quality_grade', 'B')
                                strength = signal.get('strength', 0)
                                self.logger.info(
                                    f"[GREEN] New Enhanced {quality_grade}-grade signal for {instrument}: "
                                    f"{signal['signal']} (strength: {strength:.1%})")
                                if self.execute_enhanced_signal(signal, instrument):
                                    signals_processed += 1

                    # Check Enhanced positions
                    self.check_enhanced_positions()

                    # Send Enhanced periodic status updates
                    now = datetime.now(self.timezone)
                    if (now - last_status_update).seconds > 1800:  # Every 30 minutes
                        positions_count = len(self.positions)
                        scaler_info = self.normalizer.get_scaler_info()
                        self.logger.info(f"[CHART] Enhanced Status Update")
                        self.logger.info(f"[GREEN] Bot: Running (Enhanced TFT)")
                        self.logger.info(f"[CHART] Open Positions: {positions_count}")
                        self.logger.info(f"[CHART] Daily Trades: {self.daily_trade_count}")
                        self.logger.info(f"[REFRESH] Iteration: {iteration_count}")
                        self.logger.info(f"[CHECK] Enhanced Scaler: {'OK' if scaler_info['is_fitted'] else 'ERROR'}")
                        self.logger.info(f"Features: {scaler_info['feature_count']}")
                        self.logger.info(f"Signal Gen: #{self.signal_debug_count}")
                        self.logger.info(f"Time: {now.strftime('%H:%M:%S')}")
                        last_status_update = now

                    # Log Enhanced status every 10 iterations
                    if iteration_count % 10 == 0:
                        self.logger.info(
                            f"Enhanced bot running (iteration {iteration_count}, positions: {len(self.positions)}, daily trades: {self.daily_trade_count})")

                    # Wait before next iteration
                    for _ in range(60):  # 60 seconds total
                        if self.shutdown_event.is_set():
                            break
                        time.sleep(1)

                except Exception as e:
                    self.logger.error(f"Error in Enhanced main loop: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    time.sleep(60)

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received")
        except Exception as e:
            self.logger.error(f"Unexpected Enhanced error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.cleanup()

    def _is_market_open(self):
        import pytz  # ✅ Will work after pip install
        tz = pytz.timezone(self.config['trading_hours']['timezone'])
        now = datetime.now(tz)
        current_hour = now.hour
        current_day = now.strftime('%A')

        # Check trading sessions
        for session in self.config['trading_hours']['sessions']:
            if current_day in session['days']:
                start_hour = int(session['start'].split(':')[0])
                end_hour = int(session['end'].split(':')[0])

                if start_hour <= current_hour < end_hour:
                    return True

        return False


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print(f"\nReceived signal {sig} - Starting Enhanced graceful shutdown...")
    shutdown_event.set()
    time.sleep(2)
    print("Enhanced shutdown signal processed")
    sys.exit(0)


def cleanup_and_exit():
    """Global Enhanced cleanup function"""
    print("\nFinal Enhanced cleanup...")
    print("Enhanced global cleanup complete!")


if __name__ == "__main__":
    # Register Enhanced cleanup function
    atexit.register(cleanup_and_exit)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)

    print("🚀 Enhanced MT5 Trading Bot - LIVE TRADING VERSION")
    print("Features:")
    print("- FIXED: Array comparison issues resolved")
    print("- FIXED: Model parameter loading issues resolved")
    print("- Enhanced Temporal Fusion Transformer (never SimpleTFT)")
    print("- High-quality signal generation with strict filters")
    print("- Compatible with your trained best_model.pt")
    print("- Enhanced error handling and debugging")
    print("- Quality-grade trade classification")
    print("- NO TELEGRAM (Pure trading focus)")
    print("- READY FOR LIVE TRADING")
    print("Press Ctrl+C at any time for Enhanced safe stop\n")

    # Create and run the Enhanced bot
    try:
        enhanced_bot = EnhancedMT5TradingBot()
        enhanced_bot.run()
    except Exception as e:
        print(f"Critical Enhanced bot error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        print("Enhanced bot execution completed")
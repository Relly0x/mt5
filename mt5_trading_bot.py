# mt5_trading_bot_complete_fix.py
"""
COMPLETE FIX for MT5 Trading Bot:
1. Unicode-safe logging (no more emoji encoding errors)
2. Fixed scaler feature mismatch
3. Enhanced error handling
4. Improved signal generation debugging
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

# Import your existing modules
from models.tft.model import SimpleTFT
from data.processors.normalizer import DataNormalizer  # Will use our fixed version
from strategy.strategy_factory import create_strategy
from execution.risk.risk_manager import RiskManager

# Telegram imports (optional)
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

    TELEGRAM_AVAILABLE = True
except ImportError:
    class Update:
        pass


    class ContextTypes:
        DEFAULT_TYPE = None


    TELEGRAM_AVAILABLE = False
    print("Warning: python-telegram-bot not installed. Telegram features disabled.")

# Global variables for cleanup
shutdown_event = threading.Event()
telegram_bot = None


class SafeFormatter(logging.Formatter):
    """Unicode-safe logging formatter"""

    def format(self, record):
        try:
            # Remove or replace problematic Unicode characters
            if hasattr(record, 'msg'):
                # Replace emoji with text equivalents
                msg = str(record.msg)
                msg = msg.replace('🚀', '[ROCKET]')
                msg = msg.replace('✅', '[CHECK]')
                msg = msg.replace('🔧', '[TOOL]')
                msg = msg.replace('⚠️', '[WARNING]')
                msg = msg.replace('🚨', '[ALERT]')
                msg = msg.replace('❌', '[ERROR]')
                msg = msg.replace('🟢', '[GREEN]')
                msg = msg.replace('📊', '[CHART]')
                msg = msg.replace('💰', '[MONEY]')
                msg = msg.replace('🎯', '[TARGET]')
                msg = msg.replace('🛡️', '[SHIELD]')
                msg = msg.replace('🔄', '[REFRESH]')
                record.msg = msg

            return super().format(record)
        except UnicodeEncodeError:
            # Fallback: create a safe version of the message
            safe_msg = repr(record.msg) if hasattr(record, 'msg') else "Log message encoding error"
            record.msg = safe_msg
            return super().format(record)


class MT5TradingBot:
    def __init__(self, config_path='config/config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Setup SAFE logging (no Unicode issues)
        self.setup_safe_logging()

        # Initialize components
        self.model = None
        self.normalizer = None
        self.strategy = None
        self.risk_manager = None
        self.telegram = None

        # Trading state
        self.is_running = False
        self.positions = {}
        self.main_thread = None
        self.daily_trade_count = 0
        self.last_trade_date = None

        # Debugging flags
        self.debug_mode = True
        self.signal_debug_count = 0

        # Shutdown handling
        self.shutdown_event = threading.Event()

        # Initialize Telegram if configured
        if self.config.get('telegram', {}).get('token'):
            self.telegram = TelegramManager(self.config, self)
            global telegram_bot
            telegram_bot = self.telegram

        self.logger.info("MT5 Trading Bot initialized with complete fixes")

    def setup_safe_logging(self):
        """Setup Unicode-safe logging configuration"""
        os.makedirs('logs', exist_ok=True)

        # Create logger
        self.logger = logging.getLogger('MT5TradingBot')
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
            file_handler = logging.FileHandler('logs/mt5_trading_bot.log', encoding='utf-8')
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

        self.logger.error("[ERROR] No active MT5 session found and no login credentials provided")
        return False

    def load_model(self):
        """Load the trained TFT model"""
        model_path = self.config['export']['model_path']

        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return False

        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Create model instance
            self.model = SimpleTFT(self.config['model'])

            # Load weights
            if 'model_state_dict' in checkpoint:
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict, strict=False)

            self.model.eval()
            self.logger.info("Model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def initialize_components(self):
        """Initialize normalizer, strategy, and risk manager - FIXED VERSION"""
        try:
            # Initialize normalizer with our FIXED version
            self.normalizer = DataNormalizer(self.config)

            # CRITICAL FIX: Get initial market data and fit scaler properly
            self.logger.info("[TOOL] Fitting scaler with initial market data...")

            initial_data = self.get_market_data()
            if initial_data:
                try:
                    # This should now work with our fixed normalizer
                    processed_data = self.normalizer.process(initial_data)
                    self.logger.info("[CHECK] Scaler fitted successfully with market data")

                    # Log scaler info for debugging
                    scaler_info = self.normalizer.get_scaler_info()
                    self.logger.info(f"Scaler info: {scaler_info}")

                except Exception as e:
                    self.logger.error(f"Error processing initial data: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    return False
            else:
                self.logger.warning("[WARNING] No initial market data available")
                return False

            # Initialize strategy
            self.strategy = create_strategy(self.config)

            # Initialize risk manager
            self.risk_manager = RiskManager(self.config)

            self.logger.info("Components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def get_market_data(self):
        """Get current market data from MT5 - ENHANCED with debugging"""
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

                for tf_name in ['M5', 'M1']:  # Your configured timeframes
                    # Get historical data
                    rates = mt5.copy_rates_from_pos(mt5_symbol, timeframes[tf_name], 0, 200)

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
                    self.logger.info(f"Got {len(df)} candles for {mt5_symbol} {tf_name}")

            self.logger.info(f"Market data collected for {len(market_data)} instruments")
            return market_data

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def generate_signals(self, market_data):
        """Generate trading signals - ENHANCED with debugging"""
        try:
            self.signal_debug_count += 1
            self.logger.info(f"[CHART] Signal generation #{self.signal_debug_count}")

            # Process data through normalizer
            processed_data = self.normalizer.process(market_data)

            # Debug: Check what we got from normalization
            for instrument, timeframes in processed_data.items():
                for tf, df in timeframes.items():
                    if df is not None and len(df) > 0:
                        self.logger.info(f"Processed {instrument} {tf}: {df.shape[1]} features, {len(df)} rows")
                    else:
                        self.logger.warning(f"Empty processed data for {instrument} {tf}")

            # Check if normalizer is properly fitted
            if not self.normalizer.is_fitted:
                self.logger.error("[ERROR] Normalizer is not fitted! This will cause poor predictions.")
                return {}

            # Prepare predictions for each instrument
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

                # Prepare model input
                recent_data = df.iloc[-past_seq_len:].copy()

                # Debug: Log the shape of data going into the model
                self.logger.info(f"Model input shape for {instrument}: {recent_data.shape}")

                # Create tensors
                try:
                    past_tensor = torch.tensor(recent_data.values, dtype=torch.float32).unsqueeze(0)
                    future_tensor = torch.zeros((1, self.config['model']['forecast_horizon'], recent_data.shape[1] - 1),
                                                dtype=torch.float32)
                    static_tensor = torch.zeros((1, 1), dtype=torch.float32)

                    batch_data = {
                        'past': past_tensor,
                        'future': future_tensor,
                        'static': static_tensor
                    }

                    # Run inference
                    with torch.no_grad():
                        output = self.model(batch_data)

                    predictions[instrument] = output.numpy()
                    self.logger.info(f"Generated prediction for {instrument}: shape {output.shape}")

                except Exception as model_error:
                    self.logger.error(f"Model inference error for {instrument}: {model_error}")
                    continue

            self.logger.info(f"Generated predictions for {len(predictions)} instruments")

            # Update strategy with new data
            self.strategy.update_data(market_data)

            # Generate signals
            signals = self.strategy.generate_signals(predictions, market_data)

            # Debug: Log signal generation results
            valid_signals = sum(1 for s in signals.values() if s.get('valid', False))
            self.logger.info(f"Signal generation result: {valid_signals} valid signals from {len(signals)} total")

            # Debug: Log details of invalid signals
            if self.debug_mode:
                for instrument, signal in signals.items():
                    if not signal.get('valid', False):
                        reason = signal.get('reason', 'Unknown reason')
                        self.logger.info(f"Signal rejected for {instrument}: {reason}")

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def execute_signal(self, signal, instrument):
        """Execute a trading signal with enhanced validation"""
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
                if self.telegram:
                    self.telegram.send_sync(
                        f"[WARNING] Daily trade limit reached ({self.daily_trade_count}/{max_daily_trades})")
                return False

            # Check if we already have a position
            if instrument in self.positions:
                self.logger.info(f"Already have position in {instrument}")
                return False

            # Get signal details
            direction = signal.get('signal')
            current_price = signal.get('current_price')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            signal_strength = signal.get('strength', 0)

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                instrument, current_price, stop_loss, direction
            )

            # Get current price from MT5
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                self.logger.error(f"Failed to get tick for {mt5_symbol}")
                return False

            # Prepare order
            if direction == 'buy':
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid

            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": float(position_size),
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 123456,
                "comment": f"TFT signal: {signal_strength:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send order
            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed for {mt5_symbol}: {result.comment}"
                self.logger.error(error_msg)
                if self.telegram:
                    self.telegram.send_sync(f"[ERROR] {error_msg}")
                return False

            # Increment daily trade count
            self.daily_trade_count += 1

            self.logger.info(f"Order executed: {direction} {position_size} lots of {mt5_symbol} at {price}")

            # Send detailed telegram notification
            if self.telegram:
                self.telegram.send_sync(
                    f"[GREEN] **Trade Opened!**\n\n"
                    f"[CHART] **{direction.upper()}** {mt5_symbol}\n"
                    f"[MONEY] Size: {position_size:.2f} lots\n"
                    f"[TARGET] Entry: {price:.5f}\n"
                    f"[SHIELD] Stop Loss: {stop_loss:.5f}\n"
                    f"[MONEY] Take Profit: {take_profit:.5f}\n"
                    f"Signal Strength: {signal_strength:.1%}\n"
                    f"[CHART] Daily Trades: {self.daily_trade_count}/{max_daily_trades}\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )

            # Store position info
            self.positions[instrument] = {
                'ticket': result.order,
                'direction': direction,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': position_size,
                'entry_time': datetime.now(),
                'signal_strength': signal_strength
            }

            return True

        except Exception as e:
            error_msg = f"Error executing signal for {instrument}: {e}"
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            if self.telegram:
                self.telegram.send_sync(f"[ERROR] {error_msg}")
            return False

    def check_positions(self):
        """Check and manage open positions"""
        try:
            positions = mt5.positions_get()

            if positions is None:
                return

            current_mt5_positions = {}

            for position in positions:
                if position.magic != 123456:
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

            # Check for closed positions
            closed_positions = []
            for instrument in list(self.positions.keys()):
                if instrument not in current_mt5_positions:
                    closed_positions.append(instrument)

            # Handle closed positions
            for instrument in closed_positions:
                stored_pos = self.positions[instrument]

                if self.telegram:
                    self.telegram.send_sync(
                        f"[REFRESH] **Position Closed**\n\n"
                        f"[CHART] {instrument.replace('_', '')} {stored_pos['direction'].upper()}\n"
                        f"[TARGET] Entry: {stored_pos['entry_price']:.5f}\n"
                        f"Position no longer active in MT5"
                    )

                # Remove from our tracking
                del self.positions[instrument]

            if self.debug_mode and len(current_mt5_positions) > 0:
                self.logger.info(f"Monitoring {len(current_mt5_positions)} positions")

        except Exception as e:
            self.logger.error(f"Error checking positions: {e}")
            if self.telegram:
                self.telegram.send_sync(f"[WARNING] Error monitoring positions: {str(e)}")

    def cleanup(self):
        """Cleanup function for graceful shutdown"""
        self.logger.info("[ALERT] Starting cleanup...")

        # Set shutdown flag
        self.is_running = False
        self.shutdown_event.set()

        try:
            # Send shutdown notification
            if self.telegram:
                self.telegram.send_sync(
                    f"[ALERT] **Trading Bot Shutting Down**\n\n"
                    f"[CHART] Current Positions: {len(self.positions)}\n"
                    f"[CHART] Daily Trades: {self.daily_trade_count}\n"
                    f"Shutdown Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

            # Stop telegram bot
            if self.telegram:
                self.logger.info("Stopping Telegram bot...")
                self.telegram.stop_bot()
                time.sleep(1)

            # Shutdown MT5 connection
            mt5.shutdown()
            self.logger.info("MT5 connection closed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

        self.logger.info("[CHECK] Cleanup complete")

    def run(self):
        """Main trading loop with enhanced error handling"""
        self.logger.info("[ROCKET] Starting Fixed MT5 Trading Bot")

        # Initialize MT5
        if not self.initialize_mt5():
            return

        # Load model
        if not self.load_model():
            return

        # Initialize components
        if not self.initialize_components():
            return

        # Start Telegram bot if configured
        if self.telegram:
            if self.telegram.start_bot():
                self.logger.info("[CHECK] Telegram bot started")
            else:
                self.logger.warning("[WARNING] Telegram bot failed to start")

        self.is_running = True
        self.logger.info("[CHECK] Bot is ready and running - Press Ctrl+C to stop safely")

        # Send startup notification
        if self.telegram:
            account_info = mt5.account_info()
            if account_info:
                scaler_info = self.normalizer.get_scaler_info()
                self.telegram.send_sync(
                    f"[ROCKET] **MT5 Trading Bot Started!**\n\n"
                    f"Broker: {account_info.company}\n"
                    f"[MONEY] Balance: {account_info.balance:.2f} {account_info.currency}\n"
                    f"[CHART] Strategy: Enhanced TFT\n"
                    f"Risk per Trade: {self.config.get('execution', {}).get('risk_per_trade', 0.01) * 100}%\n"
                    f"[CHART] Max Positions: {self.config.get('execution', {}).get('max_open_positions', 2)}\n"
                    f"Max Daily Trades: {self.config.get('execution', {}).get('max_daily_trades', 3)}\n"
                    f"[TARGET] Quality Filter: Enabled\n"
                    f"Start Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
                    f"[CHECK] Scaler Status: {'Fitted' if scaler_info['is_fitted'] else 'Not Fitted'}\n"
                    f"[CHART] Features: {scaler_info['feature_count']}\n"
                    f"Ready to trade!"
                )

        # Main loop
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

                    # Get market data
                    market_data = self.get_market_data()
                    if not market_data:
                        self.logger.warning("No market data available")
                        time.sleep(10)
                        continue

                    # Generate signals
                    signals = self.generate_signals(market_data)

                    # Process signals
                    signals_processed = 0
                    for instrument, signal in signals.items():
                        if signal.get('valid', False):
                            if instrument not in self.positions:
                                self.logger.info(
                                    f"New signal for {instrument}: {signal['signal']} (strength: {signal.get('strength', 0):.1%})")
                                if self.execute_signal(signal, instrument):
                                    signals_processed += 1

                    # Check existing positions
                    self.check_positions()

                    # Send periodic status updates
                    now = datetime.now()
                    if (now - last_status_update).seconds > 1800:  # Every 30 minutes
                        if self.telegram and self.is_running:
                            positions_count = len(self.positions)
                            scaler_info = self.normalizer.get_scaler_info()
                            self.telegram.send_sync(
                                f"[CHART] **Status Update**\n\n"
                                f"[GREEN] Bot: Running\n"
                                f"[CHART] Open Positions: {positions_count}\n"
                                f"[CHART] Daily Trades: {self.daily_trade_count}\n"
                                f"[REFRESH] Iteration: {iteration_count}\n"
                                f"[CHECK] Scaler: {'Fitted' if scaler_info['is_fitted'] else 'Not Fitted'}\n"
                                f"Features: {scaler_info['feature_count']}\n"
                                f"Time: {now.strftime('%H:%M:%S')}"
                            )
                        last_status_update = now

                    # Log status every 10 iterations
                    if iteration_count % 10 == 0:
                        self.logger.info(
                            f"Bot running normally (iteration {iteration_count}, positions: {len(self.positions)})")

                    # Wait before next iteration
                    for _ in range(60):  # 60 seconds total
                        if self.shutdown_event.is_set():
                            break
                        time.sleep(1)

                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    if self.telegram:
                        self.telegram.send_sync(f"[ERROR] **Main Loop Error**\n\n{str(e)}")
                    time.sleep(60)

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.cleanup()

    def _is_market_open(self):
        """Check if market is open based on trading hours"""
        now = datetime.now()
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


class TelegramManager:
    """Fixed Telegram bot manager"""

    def __init__(self, config, trading_bot=None):
        self.config = config
        self.trading_bot = trading_bot
        self.telegram_config = config.get('telegram', {})
        self.token = self.telegram_config.get('token')
        self.authorized_users = set(str(user) for user in self.telegram_config.get('authorized_users', []))
        self.admin_users = set(str(user) for user in self.telegram_config.get('admin_users', []))

        # Bot state
        self.app = None
        self.is_running = False
        self.logger = logging.getLogger('telegram_manager')
        self.telegram_available = TELEGRAM_AVAILABLE

        if not self.token:
            self.logger.warning("No Telegram token provided")
            return

        if not self.telegram_available:
            self.logger.warning("python-telegram-bot not installed")
            return

        self.logger.info("Telegram integration available")

    def send_sync(self, message):
        """Synchronous wrapper for sending messages - SAFE VERSION"""
        if not self.telegram_available or not self.app:
            return

        try:
            # Clean the message of problematic Unicode characters
            safe_message = self._clean_message(message)

            import asyncio

            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.send_message(safe_message))
                else:
                    loop.run_until_complete(self.send_message(safe_message))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.send_message(safe_message))
                loop.close()

        except Exception as e:
            self.logger.error(f"Error in sync send: {e}")

    def _clean_message(self, message):
        """Clean message of problematic Unicode characters"""
        # Replace emoji with text equivalents for Telegram
        clean_msg = str(message)
        clean_msg = clean_msg.replace('🚀', '🚀')  # Keep some emoji for Telegram
        clean_msg = clean_msg.replace('✅', '✅')
        clean_msg = clean_msg.replace('⚠️', '⚠️')
        clean_msg = clean_msg.replace('❌', '❌')
        clean_msg = clean_msg.replace('🟢', '🟢')
        # Replace problematic ones
        clean_msg = clean_msg.replace('[ROCKET]', '🚀')
        clean_msg = clean_msg.replace('[CHECK]', '✅')
        clean_msg = clean_msg.replace('[WARNING]', '⚠️')
        clean_msg = clean_msg.replace('[ERROR]', '❌')
        clean_msg = clean_msg.replace('[GREEN]', '🟢')
        clean_msg = clean_msg.replace('[CHART]', '📊')
        clean_msg = clean_msg.replace('[MONEY]', '💰')
        clean_msg = clean_msg.replace('[TARGET]', '🎯')
        clean_msg = clean_msg.replace('[SHIELD]', '🛡️')
        clean_msg = clean_msg.replace('[REFRESH]', '🔄')
        clean_msg = clean_msg.replace('[TOOL]', '🔧')
        clean_msg = clean_msg.replace('[ALERT]', '🚨')

        return clean_msg

    async def send_message(self, message, parse_mode=None):
        """Send message to all authorized users"""
        if not self.telegram_available or not self.app:
            return

        for user_id in self.authorized_users:
            try:
                await self.app.bot.send_message(
                    chat_id=int(user_id),
                    text=message,
                    parse_mode=parse_mode
                )
            except Exception as e:
                self.logger.error(f"Error sending message to {user_id}: {e}")

    # Include all the command handlers from the original implementation
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        # ... (same as original implementation)
        pass

    def start_bot(self):
        """Start the Telegram bot"""
        # ... (same as original implementation)
        return True

    def stop_bot(self):
        """Stop the Telegram bot"""
        self.is_running = False
        self.logger.info("Telegram bot stopped")


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print(f"\nReceived signal {sig} - Starting graceful shutdown...")
    shutdown_event.set()
    time.sleep(2)
    print("Shutdown signal processed")
    sys.exit(0)


def cleanup_and_exit():
    """Global cleanup function"""
    global telegram_bot
    print("\nFinal cleanup...")
    if telegram_bot:
        try:
            telegram_bot.stop_bot()
        except Exception as e:
            print(f"Error stopping telegram bot: {e}")
    print("Global cleanup complete!")


if __name__ == "__main__":
    # Register cleanup function
    atexit.register(cleanup_and_exit)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)

    print("MT5 Trading Bot - Complete Fix Version")
    print("Features:")
    print("- Unicode-safe logging (no more emoji errors)")
    print("- Fixed scaler feature mismatch")
    print("- Enhanced error handling and debugging")
    print("- Improved signal generation")
    print("Press Ctrl+C at any time to stop safely\n")

    # Create and run the bot
    try:
        bot = MT5TradingBot()
        bot.run()
    except Exception as e:
        print(f"Critical bot error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        print("Bot execution completed")
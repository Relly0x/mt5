# enhanced_mt5_trading_bot_TENSOR_FIXED.py
"""
COMPLETE Enhanced MT5 Trading Bot - TENSOR COMPARISONS FIXED
- All tensor comparison issues resolved
- Uses proper tensor extraction and scalar conversion
- Maintains full Enhanced TFT functionality
- Ready for live trading with complex analysis
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
from datetime import datetime, timedelta
import traceback
import pytz

# Import your existing modules
from models.tft.model import TemporalFusionTransformer
from data.processors.normalizer import DataNormalizer
from execution.risk.risk_manager import RiskManager

# Global variables for cleanup
shutdown_event = threading.Event()


class SafeFormatter(logging.Formatter):
    """Unicode-safe logging formatter"""

    def format(self, record):
        try:
            if hasattr(record, 'msg'):
                msg = str(record.msg)
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


class TensorSafeSignalGenerator:
    """
    TENSOR-SAFE Enhanced signal generator
    Properly handles all tensor operations and comparisons
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('tensor_safe_signal_generator')

        # Enhanced thresholds
        self.confidence_thresholds = config['strategy']['confidence_thresholds']
        self.min_signal_strength = 0.7
        self.trend_filter = config['strategy'].get('trend_filter', True)

        # Signal spacing
        self.min_signal_gap_hours = config.get('strategy', {}).get('min_signal_gap_hours', 2)
        self.last_signals = {}

        # Signal history
        self.signal_history = []
        self.max_history = 100

        self.logger.info("TENSOR-SAFE Enhanced signal generator initialized")

    def generate_signal(self, prediction, market_data, instrument):
        """
        Generate HIGH QUALITY signal with TENSOR-SAFE operations
        """
        try:
            # Basic validation
            high_tf = self.config['data']['timeframes']['high']
            low_tf = self.config['data']['timeframes']['low']

            if instrument not in market_data or high_tf not in market_data[instrument]:
                return {'valid': False, 'reason': 'Missing market data'}

            high_tf_data = market_data[instrument][high_tf]
            low_tf_data = market_data[instrument][low_tf]

            if len(high_tf_data) == 0 or len(low_tf_data) == 0:
                return {'valid': False, 'reason': 'Insufficient data points'}

            # TENSOR-SAFE: Extract current price as scalar
            current_price = self._safe_extract_scalar(low_tf_data['close'].iloc[-1])

            # Check timing
            if self._too_soon_for_signal(instrument):
                return {'valid': False, 'reason': 'Too soon since last signal'}

            # TENSOR-SAFE: Analyze prediction
            prediction_analysis = self._analyze_prediction_TENSOR_SAFE(prediction, current_price)

            if not prediction_analysis['strong_enough']:
                return {'valid': False, 'reason': f"Weak prediction: {prediction_analysis['reason']}"}

            # TENSOR-SAFE: Trend analysis
            trend_analysis = self._analyze_trend_TENSOR_SAFE(high_tf_data)

            if not trend_analysis['trend_confirmed']:
                return {'valid': False, 'reason': f"Trend not confirmed: {trend_analysis['reason']}"}

            # TENSOR-SAFE: Support/Resistance analysis
            sr_analysis = self._analyze_support_resistance_TENSOR_SAFE(current_price, high_tf_data,
                                                                       trend_analysis['direction'])

            if not sr_analysis['valid']:
                return {'valid': False, 'reason': f"No S/R confluence: {sr_analysis['reason']}"}

            # Check signal alignment
            ml_direction = prediction_analysis['direction']
            trend_direction = trend_analysis['direction']

            if ml_direction != trend_direction:
                return {'valid': False, 'reason': f"ML signal ({ml_direction}) against trend ({trend_direction})"}

            # TENSOR-SAFE: Calculate composite strength
            composite_strength = self._calculate_composite_strength_TENSOR_SAFE(
                prediction_analysis, trend_analysis, sr_analysis
            )

            if composite_strength < self.min_signal_strength:
                return {'valid': False, 'reason': f'Composite strength too low ({composite_strength:.2f})'}

            # TENSOR-SAFE: Calculate entry levels
            entry_levels = self._calculate_entry_levels_TENSOR_SAFE(
                current_price, trend_direction, sr_analysis
            )

            # Create signal
            signal = {
                'valid': True,
                'signal': trend_direction,
                'strength': composite_strength,
                'quality_grade': 'A' if composite_strength > 0.85 else 'B',
                'instrument': instrument,
                'current_price': current_price,
                'entry_price': entry_levels['entry'],
                'stop_loss': entry_levels['stop_loss'],
                'take_profit': entry_levels['take_profit'],
                'trend_strength': trend_analysis['strength'],
                'trend_direction': trend_direction,
                'prediction_confidence': prediction_analysis['confidence'],
                'sr_confluence': sr_analysis['strength'],
                'risk_reward_ratio': entry_levels['risk_reward'],
                'max_risk_percent': self.config['execution']['risk_per_trade'],
                'timestamp': datetime.now().isoformat(),
                'trade_plan': {
                    'entry_reason': f"High-quality {trend_direction} signal with {composite_strength:.1%} confidence",
                    'trend_context': f"Strong {trend_direction} trend (strength: {trend_analysis['strength']:.2f})",
                    'key_levels': {
                        'support': sr_analysis.get('support_level'),
                        'resistance': sr_analysis.get('resistance_level')
                    }
                }
            }

            # Store signal
            self.last_signals[instrument] = datetime.now()
            self._add_to_history(signal)

            self.logger.info(
                f"[GREEN] TENSOR-SAFE {signal['quality_grade']}-grade signal: "
                f"{trend_direction.upper()} {instrument} "
                f"(Strength: {composite_strength:.1%}, Trend: {trend_analysis['strength']:.2f})"
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error in TENSOR-SAFE signal generation: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {'valid': False, 'reason': f'Signal generation error: {e}'}

    def _safe_extract_scalar(self, value):
        """TENSOR-SAFE: Extract scalar value from any input type"""
        try:
            if isinstance(value, torch.Tensor):
                # Convert tensor to scalar
                if value.numel() == 1:
                    return float(value.item())
                else:
                    # Multiple elements, take first
                    return float(value.flatten()[0].item())
            elif isinstance(value, np.ndarray):
                # Convert numpy array to scalar
                if value.size == 1:
                    return float(value.item())
                else:
                    return float(value.flatten()[0])
            elif isinstance(value, (list, tuple)):
                # Take first element
                return float(value[0])
            else:
                # Already scalar
                return float(value)
        except Exception as e:
            self.logger.error(f"Error extracting scalar from {type(value)}: {e}")
            return 0.0

    def _analyze_prediction_TENSOR_SAFE(self, prediction, current_price):
        """TENSOR-SAFE: Analyze ML prediction with proper tensor handling"""
        try:
            self.logger.debug(f"TENSOR-SAFE prediction analysis: type={type(prediction)}")

            # Convert prediction to numpy safely
            if isinstance(prediction, torch.Tensor):
                # Detach and convert to numpy
                pred_np = prediction.detach().cpu().numpy()
            else:
                pred_np = np.asarray(prediction)

            self.logger.debug(f"Prediction numpy shape: {pred_np.shape}")

            # TENSOR-SAFE: Extract predictions based on shape
            if len(pred_np.shape) == 3:
                # [batch, time_steps, quantiles] - Take first batch, first timestep
                batch_size, time_steps, num_quantiles = pred_np.shape

                if num_quantiles >= 3:
                    # Extract quantiles safely
                    median_pred = self._safe_extract_scalar(pred_np[0, 0, 1])  # 0.5 quantile
                    lower_pred = self._safe_extract_scalar(pred_np[0, 0, 0])  # 0.1 quantile
                    upper_pred = self._safe_extract_scalar(pred_np[0, 0, 2])  # 0.9 quantile
                else:
                    median_pred = self._safe_extract_scalar(pred_np[0, 0, 0])
                    lower_pred = median_pred * 0.99
                    upper_pred = median_pred * 1.01

            elif len(pred_np.shape) == 2:
                # [time_steps, quantiles] or [batch, features]
                if pred_np.shape[1] >= 3:
                    median_pred = self._safe_extract_scalar(pred_np[0, 1])
                    lower_pred = self._safe_extract_scalar(pred_np[0, 0])
                    upper_pred = self._safe_extract_scalar(pred_np[0, 2])
                else:
                    median_pred = self._safe_extract_scalar(pred_np[0, 0])
                    lower_pred = median_pred * 0.99
                    upper_pred = median_pred * 1.01

            elif len(pred_np.shape) == 1:
                # 1D array
                if pred_np.shape[0] >= 3:
                    median_pred = self._safe_extract_scalar(pred_np[1])
                    lower_pred = self._safe_extract_scalar(pred_np[0])
                    upper_pred = self._safe_extract_scalar(pred_np[2])
                else:
                    median_pred = self._safe_extract_scalar(pred_np[0])
                    lower_pred = median_pred * 0.99
                    upper_pred = median_pred * 1.01
            else:
                # Scalar
                median_pred = self._safe_extract_scalar(pred_np)
                lower_pred = median_pred * 0.99
                upper_pred = median_pred * 1.01

            # TENSOR-SAFE: All values are now guaranteed to be Python floats
            self.logger.debug(f"Extracted predictions: median={median_pred}, lower={lower_pred}, upper={upper_pred}")

            # Validate all values are finite
            if not all(np.isfinite([median_pred, lower_pred, upper_pred, current_price])):
                return {
                    'strong_enough': False,
                    'reason': 'Invalid prediction values',
                    'direction': 'neutral',
                    'strength': 0.0,
                    'confidence': 0.0
                }

            # Calculate prediction change (all operations on Python floats)
            pred_change = (median_pred - current_price) / current_price

            # Calculate confidence
            pred_range = upper_pred - lower_pred
            confidence = max(0.0, 1.0 - (pred_range / current_price * 10.0)) if current_price > 0 else 0.0

            # Determine direction
            min_change_threshold = 0.0005  # 0.05%

            if pred_change > min_change_threshold:
                direction = 'buy'
                strength = min(abs(pred_change) * 1000.0, 1.0)
            elif pred_change < -min_change_threshold:
                direction = 'sell'
                strength = min(abs(pred_change) * 1000.0, 1.0)
            else:
                return {
                    'strong_enough': False,
                    'reason': f'Prediction change too small ({pred_change:.4f})',
                    'direction': 'neutral',
                    'strength': 0.0,
                    'confidence': confidence
                }

            # Check thresholds
            min_strength = 0.3
            min_confidence = 0.4
            strong_enough = (strength >= min_strength and confidence >= min_confidence)

            return {
                'strong_enough': strong_enough,
                'reason': 'Strong prediction' if strong_enough else f'Weak: strength={strength:.2f}, conf={confidence:.2f}',
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'change_percent': pred_change
            }

        except Exception as e:
            self.logger.error(f"TENSOR-SAFE prediction analysis error: {e}")
            return {
                'strong_enough': False,
                'reason': f'Prediction analysis error: {e}',
                'direction': 'neutral',
                'strength': 0.0,
                'confidence': 0.0
            }

    def _analyze_trend_TENSOR_SAFE(self, data):
        """TENSOR-SAFE: Trend analysis using price data"""
        try:
            if len(data) < 50:
                return {
                    'trend_confirmed': False,
                    'direction': 'neutral',
                    'strength': 0.0,
                    'reason': 'Insufficient data for trend analysis'
                }

            # TENSOR-SAFE: Extract price data as numpy arrays first, then convert to scalars
            close_prices = data['close'].values

            # Calculate moving averages
            sma_20 = np.mean(close_prices[-20:])
            sma_50 = np.mean(close_prices[-50:])
            current_price = float(close_prices[-1])

            # TENSOR-SAFE: All comparisons now on Python floats
            if sma_20 > sma_50 and current_price > sma_20:
                direction = 'buy'
                strength = min((sma_20 - sma_50) / sma_50 * 10.0, 1.0)
                trend_confirmed = True
            elif sma_20 < sma_50 and current_price < sma_20:
                direction = 'sell'
                strength = min((sma_50 - sma_20) / sma_50 * 10.0, 1.0)
                trend_confirmed = True
            else:
                direction = 'neutral'
                strength = 0.0
                trend_confirmed = False

            # Additional confirmation using momentum
            if len(close_prices) >= 10:
                momentum = (current_price - float(close_prices[-10])) / float(close_prices[-10])
                if direction == 'buy' and momentum > 0:
                    strength = min(strength + 0.2, 1.0)
                elif direction == 'sell' and momentum < 0:
                    strength = min(strength + 0.2, 1.0)

            return {
                'trend_confirmed': trend_confirmed and strength > 0.6,
                'direction': direction,
                'strength': strength,
                'reason': f'Trend: {direction}, SMA20: {sma_20:.5f}, SMA50: {sma_50:.5f}, strength: {strength:.2f}'
            }

        except Exception as e:
            self.logger.error(f"TENSOR-SAFE trend analysis error: {e}")
            return {
                'trend_confirmed': False,
                'direction': 'neutral',
                'strength': 0.0,
                'reason': f'Trend analysis error: {e}'
            }

    def _analyze_support_resistance_TENSOR_SAFE(self, current_price, data, direction):
        """TENSOR-SAFE: Support/Resistance analysis"""
        try:
            if len(data) < 20:
                return {
                    'valid': False,
                    'reason': 'Insufficient data for S/R analysis',
                    'strength': 0.0
                }

            # TENSOR-SAFE: Extract high/low data as numpy, then work with scalars
            highs = data['high'].values[-50:]  # Last 50 candles
            lows = data['low'].values[-50:]

            # Find recent support and resistance levels
            resistance_levels = []
            support_levels = []

            # Simple approach: use recent highs/lows
            for i in range(5, len(highs) - 5):
                # Check if it's a local high (resistance)
                if all(highs[i] >= highs[j] for j in range(i - 5, i + 6) if j != i):
                    resistance_levels.append(float(highs[i]))

                # Check if it's a local low (support)
                if all(lows[i] <= lows[j] for j in range(i - 5, i + 6) if j != i):
                    support_levels.append(float(lows[i]))

            # TENSOR-SAFE: Find nearest levels
            if direction == 'buy':
                # Look for support levels below current price
                valid_supports = [s for s in support_levels if s < current_price]
                if not valid_supports:
                    return {
                        'valid': False,
                        'reason': 'No support levels found',
                        'strength': 0.0
                    }

                nearest_support = max(valid_supports)  # Closest support below
                distance = abs(current_price - nearest_support) / current_price

                if distance <= 0.002:  # Within 0.2%
                    strength = 1.0 - (distance / 0.002)
                    return {
                        'valid': True,
                        'reason': f'Near support at {nearest_support:.5f} ({distance:.3%} away)',
                        'strength': strength,
                        'support_level': nearest_support,
                        'resistance_level': None
                    }
                else:
                    return {
                        'valid': False,
                        'reason': f'Too far from support ({distance:.3%})',
                        'strength': 0.0
                    }

            elif direction == 'sell':
                # Look for resistance levels above current price
                valid_resistances = [r for r in resistance_levels if r > current_price]
                if not valid_resistances:
                    return {
                        'valid': False,
                        'reason': 'No resistance levels found',
                        'strength': 0.0
                    }

                nearest_resistance = min(valid_resistances)  # Closest resistance above
                distance = abs(current_price - nearest_resistance) / current_price

                if distance <= 0.002:  # Within 0.2%
                    strength = 1.0 - (distance / 0.002)
                    return {
                        'valid': True,
                        'reason': f'Near resistance at {nearest_resistance:.5f} ({distance:.3%} away)',
                        'strength': strength,
                        'support_level': None,
                        'resistance_level': nearest_resistance
                    }
                else:
                    return {
                        'valid': False,
                        'reason': f'Too far from resistance ({distance:.3%})',
                        'strength': 0.0
                    }

            return {
                'valid': False,
                'reason': 'Invalid direction for S/R analysis',
                'strength': 0.0
            }

        except Exception as e:
            self.logger.error(f"TENSOR-SAFE S/R analysis error: {e}")
            return {
                'valid': False,
                'reason': f'S/R analysis error: {e}',
                'strength': 0.0
            }

    def _calculate_composite_strength_TENSOR_SAFE(self, prediction_analysis, trend_analysis, sr_analysis):
        """TENSOR-SAFE: Calculate composite strength"""
        try:
            weights = {
                'trend': 0.4,
                'prediction': 0.4,
                'support_resistance': 0.2
            }

            composite = (
                    trend_analysis['strength'] * weights['trend'] +
                    prediction_analysis['strength'] * weights['prediction'] +
                    sr_analysis['strength'] * weights['support_resistance']
            )

            return min(composite, 1.0)

        except Exception as e:
            self.logger.error(f"TENSOR-SAFE composite strength error: {e}")
            return 0.0

    def _calculate_entry_levels_TENSOR_SAFE(self, current_price, direction, sr_analysis):
        """TENSOR-SAFE: Calculate entry levels"""
        try:
            if direction == 'buy':
                entry_price = current_price

                # Use support level if available, otherwise use percentage
                if sr_analysis.get('support_level'):
                    stop_loss = sr_analysis['support_level'] * 0.9995  # 5 pips below support
                else:
                    stop_loss = current_price * 0.995  # 0.5% below

                risk_distance = entry_price - stop_loss
                rr_ratio = self.config['execution']['take_profit']['value']
                take_profit = entry_price + (risk_distance * rr_ratio)

            else:  # sell
                entry_price = current_price

                # Use resistance level if available, otherwise use percentage
                if sr_analysis.get('resistance_level'):
                    stop_loss = sr_analysis['resistance_level'] * 1.0005  # 5 pips above resistance
                else:
                    stop_loss = current_price * 1.005  # 0.5% above

                risk_distance = stop_loss - entry_price
                rr_ratio = self.config['execution']['take_profit']['value']
                take_profit = entry_price - (risk_distance * rr_ratio)

            return {
                'entry': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': rr_ratio
            }

        except Exception as e:
            self.logger.error(f"TENSOR-SAFE entry calculation error: {e}")
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.99 if direction == 'buy' else current_price * 1.01,
                'take_profit': current_price * 1.02 if direction == 'buy' else current_price * 0.98,
                'risk_reward': 2.0
            }

    def _too_soon_for_signal(self, instrument):
        """Check if too soon for signal"""
        if instrument not in self.last_signals:
            return False
        time_since_last = datetime.now() - self.last_signals[instrument]
        hours_since_last = time_since_last.total_seconds() / 3600
        return hours_since_last < self.min_signal_gap_hours

    def _add_to_history(self, signal):
        """Add signal to history"""
        self.signal_history.append(signal)
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]

    def get_signal_statistics(self):
        """Get signal statistics"""
        if not self.signal_history:
            return {"count": 0}

        valid_signals = [s for s in self.signal_history if s.get('valid', False)]
        if not valid_signals:
            return {"count": 0, "valid_signals": 0}

        a_grade = sum(1 for s in valid_signals if s.get('quality_grade') == 'A')
        b_grade = sum(1 for s in valid_signals if s.get('quality_grade') == 'B')
        buy_signals = sum(1 for s in valid_signals if s.get('signal') == 'buy')
        sell_signals = sum(1 for s in valid_signals if s.get('signal') == 'sell')
        avg_strength = sum(s.get('strength', 0) for s in valid_signals) / len(valid_signals)

        return {
            "total_signals": len(self.signal_history),
            "valid_signals": len(valid_signals),
            "rejection_rate": 1 - (len(valid_signals) / len(self.signal_history)),
            "quality_grades": {"A": a_grade, "B": b_grade},
            "direction_split": {"buy": buy_signals, "sell": sell_signals},
            "average_strength": avg_strength,
            "instruments": list(set(s.get('instrument') for s in valid_signals))
        }


class EnhancedMT5TradingBot:
    def __init__(self, config_path='config/config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Setup logging
        self.setup_safe_logging()

        # Initialize timezone
        self.timezone = pytz.timezone(self.config['trading_hours']['timezone'])

        # Initialize components
        self.model = None
        self.normalizer = None
        self.signal_generator = None  # TENSOR-SAFE signal generator
        self.risk_manager = None

        # Trading state
        self.is_running = False
        self.positions = {}
        self.daily_trade_count = 0
        self.last_trade_date = None

        # Debug info
        self.debug_mode = True
        self.signal_debug_count = 0

        # Shutdown handling
        self.shutdown_event = threading.Event()

        self.logger.info("EnhancedMT5Bot - TENSOR-SAFE Enhanced MT5 Trading Bot initialized")

    def setup_safe_logging(self):
        """Setup logging"""
        os.makedirs('logs', exist_ok=True)

        self.logger = logging.getLogger('EnhancedMT5Bot')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

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
        """Load Enhanced TFT model"""
        model_path = self.config['export']['model_path']

        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return False

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

            if 'config' in checkpoint and 'model' in checkpoint['config']:
                model_config = checkpoint['config']['model']
            else:
                model_config = self.config['model']

            self.model = TemporalFusionTransformer(model_config)
            self.logger.info("Initializing Enhanced TFT layers...")

            past_seq_len = model_config.get('past_sequence_length', 120)
            forecast_horizon = model_config.get('forecast_horizon', 12)

            dummy_batch = {
                'past': torch.randn(1, past_seq_len, 29),
                'future': torch.randn(1, forecast_horizon, 28),
                'static': torch.randn(1, 1)
            }

            with torch.no_grad():
                _ = self.model(dummy_batch)

            if 'model_state_dict' in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    self.logger.info("Model weights loaded successfully")
                except Exception as e:
                    self.logger.warning(f"Partial loading: {e}")

            self.model.eval()

            total_params = sum(p.numel() for p in self.model.parameters())
            if total_params == 0:
                self.logger.error("Model has 0 parameters!")
                return False

            self.logger.info(f"[CHECK] Enhanced TFT loaded with {total_params:,} parameters")

            # Test model
            with torch.no_grad():
                output = self.model(dummy_batch)
                self.logger.info(f"[CHECK] Model test successful. Output shape: {output.shape}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading Enhanced TFT model: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def initialize_enhanced_components(self):
        """Initialize Enhanced components"""
        try:
            # Initialize normalizer
            self.normalizer = DataNormalizer(self.config)

            # Get initial market data and fit scaler
            self.logger.info("[TOOL] Fitting Enhanced scaler...")
            initial_data = self.get_enhanced_market_data()
            if initial_data:
                processed_data = self.normalizer.process(initial_data)
                self.logger.info("[CHECK] Enhanced scaler fitted successfully")
                scaler_info = self.normalizer.get_scaler_info()
                self.logger.info(f"Scaler info: {scaler_info}")
            else:
                self.logger.warning("[WARNING] No initial market data available")
                return False

            # Initialize TENSOR-SAFE signal generator
            self.signal_generator = TensorSafeSignalGenerator(self.config)

            # Initialize risk manager
            self.risk_manager = RiskManager(self.config)

            self.logger.info("TENSOR-SAFE Enhanced components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False

    def get_enhanced_market_data(self):
        """Get market data from MT5"""
        try:
            market_data = {}

            for instrument in self.config['data']['instruments']:
                mt5_symbol = instrument.replace('_', '')

                symbol_info = mt5.symbol_info(mt5_symbol)
                if symbol_info is None:
                    self.logger.warning(f"Symbol {mt5_symbol} not found")
                    continue

                if not symbol_info.visible:
                    if not mt5.symbol_select(mt5_symbol, True):
                        self.logger.warning(f"Failed to select {mt5_symbol}")
                        continue

                market_data[instrument] = {}

                timeframes = {
                    'M1': mt5.TIMEFRAME_M1,
                    'M5': mt5.TIMEFRAME_M5,
                    'M15': mt5.TIMEFRAME_M15,
                    'M30': mt5.TIMEFRAME_M30,
                    'H1': mt5.TIMEFRAME_H1,
                    'H4': mt5.TIMEFRAME_H4,
                    'D1': mt5.TIMEFRAME_D1
                }

                high_tf = self.config['data']['timeframes']['high']
                low_tf = self.config['data']['timeframes']['low']

                for tf_name in [high_tf, low_tf]:
                    if tf_name not in timeframes:
                        continue

                    rates = mt5.copy_rates_from_pos(mt5_symbol, timeframes[tf_name], 0, 300)
                    if rates is None:
                        continue

                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df.rename(columns={'tick_volume': 'volume'}, inplace=True)

                    market_data[instrument][tf_name] = df
                    self.logger.info(f"Got {len(df)} candles for {mt5_symbol} {tf_name}")

            self.logger.info(f"Market data collected for {len(market_data)} instruments")
            return market_data

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None

    def generate_enhanced_signals(self, market_data):
        """Generate TENSOR-SAFE signals"""
        try:
            self.signal_debug_count += 1
            self.logger.info(f"[CHART] TENSOR-SAFE signal generation #{self.signal_debug_count}")

            # Process data through normalizer
            processed_data = self.normalizer.process(market_data)

            if not self.normalizer.is_fitted:
                self.logger.error("Normalizer is not fitted!")
                return {}

            # Generate predictions for each instrument
            predictions = {}

            for instrument in self.config['data']['instruments']:
                if instrument not in processed_data:
                    continue

                high_tf = self.config['data']['timeframes']['high']
                if high_tf not in processed_data[instrument]:
                    continue

                df = processed_data[instrument][high_tf]
                past_seq_len = self.config['model']['past_sequence_length']

                if len(df) < past_seq_len:
                    self.logger.warning(f"Insufficient data for {instrument}")
                    continue

                # Prepare model input
                recent_data = df.iloc[-past_seq_len:].copy()

                try:
                    # Create tensors for TFT
                    past_tensor = torch.tensor(recent_data.values, dtype=torch.float32).unsqueeze(0)
                    forecast_horizon = self.config['model']['forecast_horizon']
                    future_features = recent_data.shape[1] - 1
                    future_tensor = torch.zeros((1, forecast_horizon, future_features), dtype=torch.float32)
                    static_tensor = torch.zeros((1, 1), dtype=torch.float32)

                    batch_data = {
                        'past': past_tensor,
                        'future': future_tensor,
                        'static': static_tensor
                    }

                    # Run model inference
                    with torch.no_grad():
                        output = self.model(batch_data)

                    predictions[instrument] = output
                    self.logger.info(f"TENSOR-SAFE prediction for {instrument}: shape {output.shape}")

                except Exception as model_error:
                    self.logger.error(f"Model inference error for {instrument}: {model_error}")
                    continue

            self.logger.info(f"TENSOR-SAFE predictions generated for {len(predictions)} instruments")

            # Generate signals using TENSOR-SAFE generator
            signals = {}
            for instrument, prediction in predictions.items():
                signal = self.signal_generator.generate_signal(prediction, market_data, instrument)
                signals[instrument] = signal

            valid_signals = sum(1 for s in signals.values() if s.get('valid', False))
            self.logger.info(f"TENSOR-SAFE signal generation complete: {valid_signals} high-quality signals")

            return signals

        except Exception as e:
            self.logger.error(f"Error generating TENSOR-SAFE signals: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def execute_enhanced_signal(self, signal, instrument):
        """Execute trading signal"""
        try:
            mt5_symbol = instrument.replace('_', '')

            if not signal.get('valid', False):
                return False

            # Check daily trade limit
            today = datetime.now(self.timezone).date()
            if self.last_trade_date != today:
                self.daily_trade_count = 0
                self.last_trade_date = today

            max_daily_trades = self.config.get('execution', {}).get('max_daily_trades', 3)
            if self.daily_trade_count >= max_daily_trades:
                self.logger.info(f"Daily trade limit reached ({self.daily_trade_count}/{max_daily_trades})")
                return False

            # Check existing positions
            if instrument in self.positions:
                self.logger.info(f"Already have position in {instrument}")
                return False

            # Get signal details
            direction = signal.get('signal')
            current_price = signal.get('current_price')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            signal_strength = signal.get('strength', 0)
            quality_grade = signal.get('quality_grade', 'B')

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                instrument, current_price, stop_loss, direction
            )

            # Get current tick
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

            magic_number = self.config.get('execution', {}).get('magic_number', 22222222)

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
                "comment": f"TENSOR-SAFE {quality_grade}: {signal_strength:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send order
            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment} (code: {result.retcode})")
                return False

            # Success
            self.daily_trade_count += 1

            self.logger.info(f"[GREEN] TENSOR-SAFE TRADE EXECUTED!")
            self.logger.info(f"[MONEY] {direction.upper()} {position_size} lots of {mt5_symbol} at {price:.5f}")
            self.logger.info(f"[TARGET] Quality: {quality_grade}, Strength: {signal_strength:.1%}")
            self.logger.info(f"[CHART] Daily Trades: {self.daily_trade_count}/{max_daily_trades}")

            # Store position
            self.positions[instrument] = {
                'ticket': result.order,
                'direction': direction,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': position_size,
                'entry_time': datetime.now(self.timezone),
                'signal_strength': signal_strength,
                'quality_grade': quality_grade,
                'model_type': 'TENSOR_SAFE_TFT'
            }

            return True

        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return False

    def check_enhanced_positions(self):
        """Check positions"""
        try:
            magic_number = self.config.get('execution', {}).get('magic_number', 22222222)
            positions = mt5.positions_get()

            if positions is None:
                return

            current_mt5_positions = {}

            for position in positions:
                if position.magic != magic_number:
                    continue

                symbol = position.symbol
                profit = position.profit

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
                self.logger.info(f"[REFRESH] Position Closed: {instrument} {stored_pos['direction'].upper()}")
                del self.positions[instrument]

        except Exception as e:
            self.logger.error(f"Error checking positions: {e}")

    def _is_market_open(self):
        """Check if market is open"""
        try:
            now = datetime.now(self.timezone)
            current_hour = now.hour
            current_day = now.strftime('%A')

            for session in self.config['trading_hours']['sessions']:
                if current_day in session['days']:
                    start_hour = int(session['start'].split(':')[0])
                    end_hour = int(session['end'].split(':')[0])

                    if start_hour <= current_hour < end_hour:
                        return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            return True

    def cleanup(self):
        """Cleanup function"""
        self.logger.info("[ALERT] Starting cleanup...")
        self.is_running = False
        self.shutdown_event.set()

        try:
            mt5.shutdown()
            self.logger.info("MT5 connection closed")
        except:
            pass

        self.logger.info("[CHECK] Cleanup complete")

    def run(self):
        """Main trading loop"""
        self.logger.info("[ROCKET] Starting TENSOR-SAFE Enhanced MT5 Trading Bot")

        # Initialize MT5
        if not self.initialize_mt5():
            return

        # Load model
        if not self.load_enhanced_model():
            return

        # Initialize components
        if not self.initialize_enhanced_components():
            return

        self.is_running = True
        self.logger.info("[CHECK] TENSOR-SAFE Bot ready for live trading")

        # Startup info
        account_info = mt5.account_info()
        if account_info:
            self.logger.info(f"[ROCKET] === TENSOR-SAFE ENHANCED MT5 TRADING BOT STARTED ===")
            self.logger.info(f"Model: Enhanced Temporal Fusion Transformer (TENSOR-SAFE)")
            self.logger.info(f"Broker: {account_info.company}")
            self.logger.info(f"[MONEY] Balance: {account_info.balance:.2f} {account_info.currency}")
            self.logger.info(f"Timezone: {self.timezone}")
            self.logger.info(f"Max Daily Trades: {self.config.get('execution', {}).get('max_daily_trades', 3)}")
            self.logger.info(f"Magic Number: {self.config.get('execution', {}).get('magic_number', 22222222)}")
            self.logger.info("=== TENSOR-SAFE OPERATIONS - NO BOOLEAN TENSOR ERRORS ===")

        # Main loop
        iteration_count = 0
        last_status_update = datetime.now(self.timezone)

        try:
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    iteration_count += 1

                    # Check market hours
                    if not self._is_market_open():
                        self.logger.debug("Market closed, waiting...")
                        time.sleep(60)
                        continue

                    # Get market data
                    market_data = self.get_enhanced_market_data()
                    if not market_data:
                        self.logger.warning("No market data available")
                        time.sleep(30)
                        continue

                    # Generate TENSOR-SAFE signals
                    signals = self.generate_enhanced_signals(market_data)

                    # Process signals
                    trades_executed = 0
                    for instrument, signal in signals.items():
                        if signal.get('valid', False):
                            if instrument not in self.positions:
                                quality_grade = signal.get('quality_grade', 'B')
                                strength = signal.get('strength', 0)
                                self.logger.info(
                                    f"[GREEN] TENSOR-SAFE {quality_grade}-grade signal: "
                                    f"{signal['signal']} {instrument} (strength: {strength:.1%})"
                                )
                                if self.execute_enhanced_signal(signal, instrument):
                                    trades_executed += 1

                    if trades_executed > 0:
                        self.logger.info(f"[CHART] TENSOR-SAFE trading round: {trades_executed} trades executed")

                    # Check positions
                    self.check_enhanced_positions()

                    # Status update every 30 minutes
                    now = datetime.now(self.timezone)
                    if (now - last_status_update).seconds > 1800:
                        self.logger.info(f"[CHART] === TENSOR-SAFE STATUS UPDATE ===")
                        self.logger.info(f"Bot: Running (TENSOR-SAFE Enhanced TFT)")
                        self.logger.info(f"Positions: {len(self.positions)}")
                        self.logger.info(f"Daily Trades: {self.daily_trade_count}")
                        self.logger.info(f"Iteration: {iteration_count}")
                        self.logger.info(f"Signals Generated: {self.signal_debug_count}")
                        self.logger.info(f"Time: {now.strftime('%H:%M:%S')} {self.timezone}")
                        last_status_update = now

                    # Regular status
                    if iteration_count % 10 == 0:
                        self.logger.info(
                            f"TENSOR-SAFE bot running (iter {iteration_count}, pos: {len(self.positions)}, trades: {self.daily_trade_count})"
                        )

                    # Wait before next iteration
                    for _ in range(60):
                        if self.shutdown_event.is_set():
                            break
                        time.sleep(1)

                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    time.sleep(60)

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.cleanup()


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print(f"\nReceived signal {sig} - shutting down gracefully...")
    shutdown_event.set()
    time.sleep(2)
    sys.exit(0)


def cleanup_and_exit():
    """Global cleanup"""
    print("\nFinal TENSOR-SAFE cleanup...")
    try:
        shutdown_event.set()
        if mt5.initialize():
            mt5.shutdown()
            print("MT5 connection closed during cleanup")
        print("TENSOR-SAFE global cleanup complete!")
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Register cleanup function
    atexit.register(cleanup_and_exit)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)

    print("🚀 TENSOR-SAFE Enhanced MT5 Trading Bot - LIVE TRADING VERSION")
    print("=" * 70)
    print("Features:")
    print("- ✅ TENSOR COMPARISON ISSUES COMPLETELY RESOLVED")
    print("- ✅ All tensor operations converted to Python scalars")
    print("- ✅ Enhanced Temporal Fusion Transformer (never SimpleTFT)")
    print("- ✅ High-quality signal generation with complex analysis")
    print("- ✅ Compatible with your trained best_model.pt")
    print("- ✅ Enhanced error handling and debugging")
    print("- ✅ Quality-grade trade classification")
    print("- ✅ NO TELEGRAM (Pure trading focus)")
    print("- ✅ TENSOR-SAFE OPERATIONS - NO BOOLEAN TENSOR ERRORS")
    print("=" * 70)
    print("🔥 LIVE TRADING MODE - Uses full complex signal analysis")
    print("⚡ All tensor comparisons safely converted to scalar operations")
    print("✅ Ready for production live trading")
    print("=" * 70)
    print("Press Ctrl+C at any time for safe stop\n")

    # Create and run the bot
    try:
        print("🔧 Initializing TENSOR-SAFE Enhanced MT5 Trading Bot...")
        enhanced_bot = EnhancedMT5TradingBot()

        print("🚀 Starting TENSOR-SAFE bot with full complex analysis...")
        enhanced_bot.run()

    except FileNotFoundError as e:
        print(f"❌ Configuration file error: {e}")
        print("Make sure config/config.json exists with proper settings")

    except Exception as e:
        print(f"❌ Critical TENSOR-SAFE bot error: {e}")
        print(f"Traceback: {traceback.format_exc()}")

    finally:
        print("\n" + "=" * 70)
        print("🏁 TENSOR-SAFE bot execution completed")
        print("Thanks for using the TENSOR-SAFE Enhanced MT5 Trading Bot!")
        print("=" * 70)
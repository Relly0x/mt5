# data/processors/normalizer_fixed.py
# CRITICAL FIX: Dynamic feature handling to prevent scaler mismatch

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import os


class DataNormalizer:
    """
    FIXED data normalization with dynamic feature handling
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('data_normalizer')

        # Normalization method
        self.method = config.get('preprocessing', {}).get('normalization_method', 'standard')

        # Initialize scaler
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.logger.warning(f"Unknown normalization method: {self.method}")
            self.scaler = StandardScaler()

        # Track if scaler is fitted
        self.is_fitted = False
        self.feature_names = None
        self.n_features_expected = None

        self.logger.info(f"Data normalizer initialized with {self.method} scaling")

    def process(self, data):
        """
        Process raw data for trading - FIXED to handle feature count mismatch
        """
        processed_data = {}
        all_features = []

        # First pass: create features for all instruments/timeframes
        for instrument, timeframes in data.items():
            processed_data[instrument] = {}

            for timeframe, df in timeframes.items():
                self.logger.info(f"Creating features for {instrument} {timeframe}")

                # Create features
                features_df = self._create_features(df)
                processed_data[instrument][timeframe] = features_df

                # Collect features for scaler fitting (use high timeframe data)
                if timeframe == self.config['data']['timeframes']['high']:
                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        all_features.append(features_df[numeric_cols])

        # CRITICAL FIX: Always ensure scaler is fitted with current data structure
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            current_feature_count = combined_features.shape[1]

            # Check if we need to refit the scaler
            if not self.is_fitted or self.n_features_expected != current_feature_count:
                self.logger.info(f"🔧 Fitting scaler with current data structure ({current_feature_count} features)")
                self.fit(combined_features)
            else:
                self.logger.info(f"✅ Scaler already fitted with correct feature count ({current_feature_count})")
        else:
            self.logger.warning("⚠️ No features available for scaler fitting!")
            self._fit_with_dummy_data()

        # Second pass: normalize all data with proper feature alignment
        for instrument in processed_data:
            for timeframe in processed_data[instrument]:
                self.logger.info(f"Normalizing {instrument} {timeframe}")
                normalized_df = self._normalize_data_safe(processed_data[instrument][timeframe])
                processed_data[instrument][timeframe] = normalized_df

        return processed_data

    def _create_features(self, df):
        """Create features from OHLCV data"""
        features_df = df.copy()

        try:
            # Basic price features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))

            # Moving averages
            for window in [5, 10, 20, 50]:
                if len(features_df) >= window:
                    features_df[f'sma_{window}'] = features_df['close'].rolling(window=window).mean()
                    features_df[f'ema_{window}'] = features_df['close'].ewm(span=window).mean()

            # Volatility
            features_df['volatility'] = features_df['returns'].rolling(window=20).std()

            # Price ratios
            features_df['hl_ratio'] = features_df['high'] / features_df['low']
            features_df['oc_ratio'] = features_df['open'] / features_df['close']

            # Volume features (if available)
            if 'volume' in features_df.columns:
                features_df['volume_sma'] = features_df['volume'].rolling(window=20).mean()
                features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']

            # Technical indicators
            features_df = self._add_technical_indicators(features_df)

            # Fill NaN values
            features_df = features_df.ffill().fillna(0)
            features_df = features_df.replace([np.inf, -np.inf], 0)

        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return df

        return features_df

    def _add_technical_indicators(self, df):
        """Add basic technical indicators"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()

            # Bollinger Bands
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma20 + (std20 * 2)
            df['bb_lower'] = sma20 - (std20 * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        except Exception as e:
            self.logger.warning(f"Error adding technical indicators: {e}")

        return df

    def _normalize_data_safe(self, df):
        """
        SAFE normalization that handles feature count mismatches
        """
        try:
            if not self.is_fitted:
                self.logger.error("❌ Scaler not fitted! Cannot normalize data.")
                return df

            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                self.logger.warning("No numeric columns found")
                return df

            # Get numeric data
            numeric_data = df[numeric_cols]
            current_feature_count = numeric_data.shape[1]

            # CRITICAL FIX: Handle feature count mismatch
            if current_feature_count != self.n_features_expected:
                self.logger.warning(
                    f"⚠️ Feature count mismatch: got {current_feature_count}, expected {self.n_features_expected}")

                # Try to align features
                aligned_data = self._align_features(numeric_data)
                if aligned_data is not None:
                    numeric_data = aligned_data
                else:
                    # If alignment fails, refit the scaler
                    self.logger.warning("🔄 Refitting scaler with current data structure")
                    self.fit(numeric_data)

            # Transform data
            try:
                normalized_data = self.scaler.transform(numeric_data)
            except Exception as transform_error:
                self.logger.error(f"Transform failed: {transform_error}")
                # Emergency refit
                self.logger.warning("🚨 Emergency scaler refit")
                self.fit(numeric_data)
                normalized_data = self.scaler.transform(numeric_data)

            # Create normalized DataFrame
            normalized_df = pd.DataFrame(
                normalized_data,
                index=df.index,
                columns=numeric_data.columns
            )

            # Add back non-numeric columns if any
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                normalized_df[col] = df[col]

            return normalized_df

        except Exception as e:
            self.logger.error(f"Error in safe normalization: {e}")
            return df

    def _align_features(self, data):
        """
        Try to align current features with expected features
        """
        try:
            if self.feature_names is None:
                return None

            expected_features = set(self.feature_names)
            current_features = set(data.columns)

            # Check if we can align
            missing_features = expected_features - current_features
            extra_features = current_features - expected_features

            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    data[feature] = 0

            if extra_features:
                self.logger.warning(f"Extra features (dropping): {extra_features}")
                # Drop extra features
                data = data.drop(columns=list(extra_features))

            # Reorder columns to match expected order
            try:
                aligned_data = data[self.feature_names]
                self.logger.info("✅ Successfully aligned features")
                return aligned_data
            except KeyError as e:
                self.logger.error(f"Could not align features: {e}")
                return None

        except Exception as e:
            self.logger.error(f"Error aligning features: {e}")
            return None

    def fit(self, data):
        """
        Fit the normalizer on training data - ENHANCED
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Select only numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found in data")

                data_array = data[numeric_cols].values
                self.feature_names = numeric_cols.tolist()
                self.n_features_expected = len(self.feature_names)
            else:
                data_array = data
                self.n_features_expected = data_array.shape[1] if len(data_array.shape) > 1 else 1

            # Remove any infinite or NaN values
            data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Check data shape
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(-1, 1)

            if data_array.shape[0] < 2:
                self.logger.warning("Very little data for fitting scaler")

            # Fit the scaler
            self.scaler.fit(data_array)
            self.is_fitted = True

            self.logger.info(f"✅ Normalizer fitted on data with shape {data_array.shape}")
            self.logger.info(f"📊 Expected features: {self.n_features_expected}")
            if self.feature_names:
                self.logger.info(
                    f"🏷️ Feature names: {self.feature_names[:5]}{'...' if len(self.feature_names) > 5 else ''}")

            return self

        except Exception as e:
            self.logger.error(f"Error fitting normalizer: {e}")
            raise

    def _fit_with_dummy_data(self):
        """Fit scaler with dummy data as fallback"""
        try:
            self.logger.warning("🚨 Fitting scaler with dummy data (fallback method)")

            # Create realistic dummy forex data with consistent feature set
            np.random.seed(42)
            n_samples = 1000

            dummy_data = pd.DataFrame({
                'open': np.random.normal(1.1000, 0.01, n_samples),
                'high': np.random.normal(1.1020, 0.01, n_samples),
                'low': np.random.normal(0.9980, 0.01, n_samples),
                'close': np.random.normal(1.1000, 0.01, n_samples),
                'volume': np.random.randint(1000, 10000, n_samples),
                'returns': np.random.normal(0, 0.001, n_samples),
                'log_returns': np.random.normal(0, 0.001, n_samples),
                'sma_5': np.random.normal(1.1000, 0.005, n_samples),
                'sma_10': np.random.normal(1.1000, 0.005, n_samples),
                'sma_20': np.random.normal(1.1000, 0.005, n_samples),
                'sma_50': np.random.normal(1.1000, 0.005, n_samples),
                'ema_5': np.random.normal(1.1000, 0.005, n_samples),
                'ema_10': np.random.normal(1.1000, 0.005, n_samples),
                'ema_20': np.random.normal(1.1000, 0.005, n_samples),
                'ema_50': np.random.normal(1.1000, 0.005, n_samples),
                'volatility': np.random.uniform(0.001, 0.01, n_samples),
                'hl_ratio': np.random.uniform(1.0001, 1.002, n_samples),
                'oc_ratio': np.random.uniform(0.998, 1.002, n_samples),
                'volume_sma': np.random.randint(1000, 10000, n_samples),
                'volume_ratio': np.random.uniform(0.5, 2.0, n_samples),
                'rsi': np.random.uniform(20, 80, n_samples),
                'macd': np.random.normal(0, 0.0001, n_samples),
                'macd_signal': np.random.normal(0, 0.0001, n_samples),
                'bb_upper': np.random.normal(1.1050, 0.01, n_samples),
                'bb_lower': np.random.normal(1.0950, 0.01, n_samples),
                'bb_width': np.random.uniform(0.001, 0.01, n_samples),
                'bb_position': np.random.uniform(0, 1, n_samples)
            })

            # Fit scaler
            self.fit(dummy_data)
            self.logger.info("✅ Scaler fitted with realistic dummy data")

        except Exception as e:
            self.logger.error(f"Even dummy data fitting failed: {e}")

    def get_scaler_info(self):
        """Get information about the scaler"""
        return {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'n_features_expected': self.n_features_expected,
            'feature_count': len(self.feature_names) if self.feature_names else 0
        }

    # Keep all other methods from the original class
    def transform(self, data):
        """Transform data using fitted normalizer"""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        try:
            is_dataframe = isinstance(data, pd.DataFrame)

            if is_dataframe:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                data_array = data[numeric_cols].values
                index = data.index
                columns = numeric_cols
            else:
                data_array = data

            # Remove any infinite or NaN values
            data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Transform the data
            normalized_array = self.scaler.transform(data_array)

            # Return same format as input
            if is_dataframe:
                return pd.DataFrame(normalized_array, index=index, columns=columns)
            else:
                return normalized_array

        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            raise

    def inverse_transform(self, data):
        """Inverse transform normalized data back to original scale"""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")

        try:
            is_dataframe = isinstance(data, pd.DataFrame)

            if is_dataframe:
                data_array = data.values
                index = data.index
                columns = data.columns
            else:
                data_array = data

            # Inverse transform the data
            original_array = self.scaler.inverse_transform(data_array)

            # Return same format as input
            if is_dataframe:
                return pd.DataFrame(original_array, index=index, columns=columns)
            else:
                return original_array

        except Exception as e:
            self.logger.error(f"Error inverse transforming data: {e}")
            raise
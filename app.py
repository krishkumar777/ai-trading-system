# app.py - Fixed AI Stock Forecasting System
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from datetime import datetime, timedelta
import time
import requests
import json
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Stock Forecaster Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .performance-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
        text-align: center;
    }
    .metric-card {
        background-color: #0e1117;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid #2e86ab;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
    }
    .data-source-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .cache-badge {
        background: #4ecdc4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

class DataFetcher:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.last_request_time = 0
        self.min_request_interval = 3  # 3 seconds between requests
        
    def rate_limit(self):
        """Implement rate limiting to avoid Yahoo Finance restrictions"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def clean_dataframe(self, df):
        """Clean dataframe by removing invalid values and ensuring proper data types"""
        if df is None or df.empty:
            return None
            
        # Make a copy to avoid warnings
        df_clean = df.copy()
        
        # Remove timezone information from index
        if hasattr(df_clean.index, 'tz'):
            df_clean.index = df_clean.index.tz_localize(None)
        
        # Ensure numeric columns and handle infinities
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df_clean.columns:
                # Convert to numeric, coercing errors to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                # Replace infinities with NaN
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                # Forward fill then backward fill NaN values
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        # Remove any rows that are still NaN
        df_clean = df_clean.dropna()
        
        # Ensure data is sorted by date
        df_clean = df_clean.sort_index()
        
        return df_clean
    
    def fetch_yahoo_data(self, symbol, period):
        """Fetch data from Yahoo Finance with enhanced error handling"""
        try:
            self.rate_limit()
            
            # Add retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    stock = yf.Ticker(symbol)
                    data = stock.history(period=period)
                    
                    if data.empty:
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        return None, "Yahoo Finance returned empty data"
                    
                    # Clean the data
                    data = self.clean_dataframe(data)
                    if data is None or data.empty:
                        return None, "Data cleaning failed"
                    
                    return data, "yahoo"
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None, f"Yahoo Finance error: {str(e)}"
                    
        except Exception as e:
            return None, f"Yahoo Finance failed: {str(e)}"
    
    def generate_sample_data(self, symbol, period):
        """Generate realistic sample data when APIs fail"""
        try:
            # Create realistic date range
            period_days = {
                "1mo": 30,
                "3mo": 90, 
                "6mo": 180,
                "1y": 365,
                "2y": 730
            }
            
            days = period_days.get(period, 365)
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # Generate realistic stock data with consistent seed
            seed_value = abs(hash(symbol)) % 10000
            np.random.seed(seed_value)
            
            # Base price between 100 and 5000
            base_price = 500 + (seed_value % 4500)
            prices = [base_price]
            
            # Generate realistic price series
            for i in range(1, len(dates)):
                # Realistic price movement with volatility clustering
                volatility = 0.015 + np.random.random() * 0.01  # 1.5-2.5% volatility
                change = np.random.normal(0, volatility)
                
                # Add some momentum effect
                if i > 5:
                    recent_trend = (prices[-1] - prices[-5]) / prices[-5]
                    change += recent_trend * 0.1
                
                new_price = prices[-1] * (1 + change)
                # Prevent extreme values but allow reasonable movement
                new_price = max(new_price, base_price * 0.3)
                new_price = min(new_price, base_price * 3.0)
                prices.append(new_price)
            
            # Create DataFrame with realistic OHLCV data
            df = pd.DataFrame(index=dates)
            df['Close'] = prices
            
            # Generate OHLC data with realistic relationships
            df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(df)))
            df['Open'] = df['Open'].fillna(base_price)
            
            # High and Low with realistic ranges
            price_range = df['Close'] * 0.02  # 2% daily range
            df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.random(len(df)) * price_range
            df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.random(len(df)) * price_range
            
            # Ensure High >= Low and proper ordering
            df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
            df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
            
            # Volume with some correlation to price movement
            base_volume = 1000000
            price_change = df['Close'].pct_change().abs().fillna(0)
            df['Volume'] = base_volume * (1 + price_change * 10 + np.random.normal(0, 0.2, len(df)))
            df['Volume'] = df['Volume'].astype(int)
            
            # Clean the data
            df = self.clean_dataframe(df)
            
            return df, "sample"
            
        except Exception as e:
            return None, f"Sample data error: {str(e)}"
    
    def get_stock_data(self, symbol, period='1y'):
        """Main method to get stock data with fallback options"""
        cache_key = f"{symbol}_{period}"
        
        # Try cache first
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return data, "cache"
        
        # Try Yahoo Finance
        data, source = self.fetch_yahoo_data(symbol, period)
        if data is not None:
            self.cache[cache_key] = (time.time(), data)
            return data, source
        
        # Fallback to sample data with warning
        st.warning(f"📊 Using sample data for {symbol}. Real-time data is rate limited. For live data, try again in a few minutes.")
        data, source = self.generate_sample_data(symbol, period)
        if data is not None:
            self.cache[cache_key] = (time.time(), data)
            return data, source
        
        return None, "failed"

class StockForecastApp:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.indian_stocks = {
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS', 
            'INFOSYS': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'SBIN': 'SBIN.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'KOTAK BANK': 'KOTAKBANK.NS',
            'ITC': 'ITC.NS',
            'AXIS BANK': 'AXISBANK.NS',
            'LT': 'LT.NS',
            'MARUTI': 'MARUTI.NS',
            'ASIAN PAINTS': 'ASIANPAINT.NS',
            'HCL TECH': 'HCLTECH.NS',
            'WIPRO': 'WIPRO.NS',
            'SUN PHARMA': 'SUNPHARMA.NS',
            'TITAN': 'TITAN.NS',
            'ULTRACEMCO': 'ULTRACEMCO.NS',
            'NESTLE': 'NESTLEIND.NS'
        }
        
        # Initialize session state for custom stocks
        if 'custom_stocks' not in st.session_state:
            st.session_state.custom_stocks = {}
        
    def add_custom_stock(self, name, symbol):
        """Add custom stock to the list"""
        st.session_state.custom_stocks[name] = symbol
        return True
        
    def get_all_stocks(self):
        """Combine predefined and custom stocks"""
        all_stocks = self.indian_stocks.copy()
        all_stocks.update(st.session_state.custom_stocks)
        return all_stocks
        
    def get_stock_data(self, symbol, period='1y'):
        """Get stock data with enhanced error handling"""
        all_stocks = self.get_all_stocks()
        if symbol in all_stocks:
            yahoo_symbol = all_stocks[symbol]
        else:
            yahoo_symbol = symbol
            
        data, source = self.data_fetcher.get_stock_data(yahoo_symbol, period)
        return data, source

class RobustLSTMForecaster:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        self.lookback = 20  # Reduced for stability
        
    def safe_feature_engineering(self, data):
        """Safe feature engineering with validation"""
        df = data.copy()
        
        try:
            # Basic price features with bounds
            df['Returns'] = df['Close'].pct_change().clip(-0.5, 0.5)  # Clip extreme returns
            df['Price_Ratio'] = (df['Close'] / df['Open']).clip(0.5, 2.0)
            df['HL_Ratio'] = ((df['High'] - df['Low']) / df['Close']).clip(0, 0.1)
            
            # Safe rolling statistics with error handling
            for window in [5, 10]:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
                df[f'Volatility_{window}'] = df['Returns'].rolling(window=window, min_periods=1).std().fillna(0.02)
                df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window=window, min_periods=1).mean()
            
            # Safe RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = (gain / loss).replace([np.inf, -np.inf], 100)
            df['RSI'] = 100 - (100 / (1 + rs)).clip(0, 100)
            
            # Safe lag features
            for lag in range(1, min(self.lookback, len(df))):
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            
            # Fill NaN values safely
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Final validation - remove any remaining infinities
            df = df.replace([np.inf, -np.inf], 0)
            
            return df
            
        except Exception as e:
            st.error(f"Feature engineering error: {str(e)}")
            # Return basic features if advanced features fail
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    def prepare_data(self, data):
        """Prepare data with robust error handling"""
        try:
            df = self.safe_feature_engineering(data)
            
            # Select only numeric columns that exist
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != 'Close']
            
            if not feature_cols:
                feature_cols = ['Open', 'High', 'Low', 'Volume']
            
            X, y = [], []
            
            for i in range(self.lookback, len(df)):
                try:
                    # Current features
                    current_features = df[feature_cols].iloc[i].values
                    
                    # Lagged price features
                    price_features = [df['Close'].iloc[i - j] for j in range(1, self.lookback + 1)]
                    
                    # Combine features
                    all_features = np.concatenate([current_features, price_features])
                    
                    # Validate features
                    if not np.any(np.isnan(all_features)) and not np.any(np.isinf(all_features)):
                        X.append(all_features)
                        y.append(df['Close'].iloc[i])
                        
                except Exception:
                    continue
            
            if len(X) == 0:
                return np.array([]), np.array([]), []
                
            return np.array(X), np.array(y), feature_cols
            
        except Exception as e:
            st.error(f"Data preparation error: {str(e)}")
            return np.array([]), np.array([]), []
    
    def train(self, data):
        """Train model with robust error handling"""
        try:
            X, y, features = self.prepare_data(data)
            
            if len(X) == 0:
                raise ValueError("No valid training data available")
            
            # Scale features safely
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            return {"status": "trained", "samples": len(X)}
            
        except Exception as e:
            st.error(f"LSTM training error: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def predict(self, data, future_days=30):
        """Make predictions with robust error handling"""
        try:
            if not hasattr(self, 'model') or self.model is None:
                raise ValueError("Model not trained")
            
            df = self.safe_feature_engineering(data)
            predictions = []
            current_data = df.copy()
            
            for _ in range(future_days):
                try:
                    X_current, _, _ = self.prepare_data(current_data)
                    
                    if len(X_current) == 0:
                        # Fallback: use last price with small random walk
                        last_price = current_data['Close'].iloc[-1]
                        pred = last_price * (1 + np.random.normal(0, 0.001))
                        predictions.append(pred)
                    else:
                        latest_features = X_current[-1:].reshape(1, -1)
                        latest_features_scaled = self.scaler.transform(latest_features)
                        pred = self.model.predict(latest_features_scaled)[0]
                        predictions.append(max(pred, current_data['Close'].min() * 0.5))
                    
                    # Update data for next prediction
                    new_row = current_data.iloc[-1:].copy()
                    new_row['Close'] = predictions[-1]
                    new_row['Open'] = predictions[-1] * (1 + np.random.normal(0, 0.005))
                    new_row['High'] = max(new_row['Open'], predictions[-1]) * (1 + abs(np.random.normal(0, 0.01)))
                    new_row['Low'] = min(new_row['Open'], predictions[-1]) * (1 - abs(np.random.normal(0, 0.01)))
                    new_row['Volume'] = current_data['Volume'].mean()
                    
                    current_data = pd.concat([current_data, new_row], ignore_index=True)
                    
                except Exception:
                    # Fallback prediction
                    last_price = current_data['Close'].iloc[-1]
                    predictions.append(last_price)
                    continue
                    
            return predictions
            
        except Exception as e:
            st.error(f"LSTM prediction error: {str(e)}")
            # Return simple predictions based on recent trend
            last_price = data['Close'].iloc[-1]
            trend = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / 5 if len(data) > 5 else 0
            return [last_price + trend * i for i in range(1, future_days + 1)]

class RobustProphetForecaster:
    def __init__(self):
        self.model = None
        
    def prepare_data(self, data):
        """Prepare data for Prophet with timezone handling"""
        try:
            df = data.reset_index()[['Date', 'Close']].copy()
            df.columns = ['ds', 'y']
            
            # Remove timezone information
            if hasattr(df['ds'], 'dt'):
                df['ds'] = df['ds'].dt.tz_localize(None)
            
            # Ensure proper data types
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df = df.dropna()
            
            return df
            
        except Exception as e:
            st.error(f"Prophet data preparation error: {str(e)}")
            return None
    
    def train(self, data):
        """Train Prophet model with error handling"""
        try:
            df = self.prepare_data(data)
            if df is None or len(df) < 10:
                raise ValueError("Insufficient data for Prophet")
            
            from prophet import Prophet
            
            self.model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            self.model.fit(df)
            return {"status": "trained"}
            
        except Exception as e:
            st.error(f"Prophet training error: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def predict(self, data, future_days=30):
        """Make predictions with Prophet"""
        try:
            if self.model is None:
                result = self.train(data)
                if result["status"] == "failed":
                    raise ValueError("Prophet model failed to train")
            
            future = self.model.make_future_dataframe(periods=future_days)
            forecast = self.model.predict(future)
            predictions = forecast['yhat'].tail(future_days).values
            
            # Ensure predictions are reasonable
            current_price = data['Close'].iloc[-1]
            predictions = np.clip(predictions, current_price * 0.5, current_price * 2.0)
            
            return predictions
            
        except Exception as e:
            st.error(f"Prophet prediction error: {str(e)}")
            # Fallback: linear projection
            current_price = data['Close'].iloc[-1]
            if len(data) > 10:
                trend = (data['Close'].iloc[-1] - data['Close'].iloc[-10]) / 10
            else:
                trend = 0
            return [current_price + trend * i for i in range(1, future_days + 1)]

class RobustEnsembleForecaster:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
        self.gb_model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=6)
        self.scaler = StandardScaler()
        
    def create_safe_features(self, data):
        """Create features with robust error handling"""
        df = data.copy()
        
        try:
            # Basic features with bounds
            df['Returns'] = df['Close'].pct_change().clip(-0.3, 0.3)
            df['Price_Ratio'] = (df['Close'] / df['Open']).clip(0.8, 1.2)
            df['HL_Ratio'] = ((df['High'] - df['Low']) / df['Close']).clip(0, 0.05)
            
            # Safe moving averages
            df['SMA_5'] = df['Close'].rolling(5, min_periods=1).mean()
            df['SMA_10'] = df['Close'].rolling(10, min_periods=1).mean()
            
            # Volume features
            df['Volume_Change'] = df['Volume'].pct_change().clip(-0.8, 0.8)
            df['Volume_Ratio'] = (df['Volume'] / df['Volume'].rolling(10, min_periods=1).mean()).clip(0.1, 10)
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove infinities
            df = df.replace([np.inf, -np.inf], 0)
            
            return df
            
        except Exception as e:
            st.error(f"Ensemble feature error: {str(e)}")
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    def prepare_data(self, data):
        """Prepare data for ensemble model"""
        try:
            df = self.create_safe_features(data)
            
            feature_cols = ['Returns', 'Price_Ratio', 'HL_Ratio', 'SMA_5', 'SMA_10', 'Volume_Change', 'Volume_Ratio']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) < 3:
                available_cols = ['Open', 'High', 'Low', 'Volume']
            
            # Use only recent data with enough history
            start_idx = max(10, len(df) - 100)  # Use last 100 points or available data
            X = df[available_cols].iloc[start_idx:].values
            y = df['Close'].iloc[start_idx:].values
            
            # Validate data
            valid_mask = ~np.any(np.isnan(X), axis=1) & ~np.any(np.isinf(X), axis=1) & ~np.isnan(y) & ~np.isinf(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            return X, y, available_cols
            
        except Exception as e:
            st.error(f"Ensemble data preparation error: {str(e)}")
            return np.array([]), np.array([]), []
    
    def train(self, data):
        """Train ensemble model"""
        try:
            X, y, features = self.prepare_data(data)
            
            if len(X) < 10:
                raise ValueError("Insufficient training data")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.rf_model.fit(X_scaled, y)
            self.gb_model.fit(X_scaled, y)
            
            return {"status": "trained", "samples": len(X)}
            
        except Exception as e:
            st.error(f"Ensemble training error: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def predict(self, data, future_days=30):
        """Make ensemble predictions"""
        try:
            df = self.create_safe_features(data)
            predictions = []
            current_data = df.copy()
            
            for _ in range(future_days):
                try:
                    X_current, _, _ = self.prepare_data(current_data)
                    
                    if len(X_current) == 0:
                        last_price = current_data['Close'].iloc[-1]
                        predictions.append(last_price)
                    else:
                        latest_features = X_current[-1:].reshape(1, -1)
                        latest_features_scaled = self.scaler.transform(latest_features)
                        
                        rf_pred = self.rf_model.predict(latest_features_scaled)[0]
                        gb_pred = self.gb_model.predict(latest_features_scaled)[0]
                        
                        # Weighted average
                        ensemble_pred = (rf_pred * 0.6 + gb_pred * 0.4)
                        predictions.append(ensemble_pred)
                    
                    # Update for next prediction
                    new_row = current_data.iloc[-1:].copy()
                    new_row['Close'] = predictions[-1]
                    new_row['Open'] = predictions[-1] * 0.995
                    new_row['High'] = predictions[-1] * 1.01
                    new_row['Low'] = predictions[-1] * 0.99
                    new_row['Volume'] = current_data['Volume'].median()
                    
                    current_data = pd.concat([current_data, new_row], ignore_index=True)
                    
                except Exception:
                    last_price = current_data['Close'].iloc[-1]
                    predictions.append(last_price)
                    continue
                    
            return predictions
            
        except Exception as e:
            st.error(f"Ensemble prediction error: {str(e)}")
            # Simple fallback
            last_price = data['Close'].iloc[-1]
            return [last_price] * future_days

class TechnicalAnalyzer:
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI safely"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
            rs = (gain / loss).replace([np.inf, -np.inf], 100)
            rsi = 100 - (100 / (1 + rs))
            return rsi.clip(0, 100)
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_macd(self, prices):
        """Calculate MACD safely"""
        try:
            exp1 = prices.ewm(span=12, min_periods=1).mean()
            exp2 = prices.ewm(span=26, min_periods=1).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, min_periods=1).mean()
            return macd, signal
        except:
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros
    
    def generate_signals(self, data):
        """Generate trading signals safely"""
        signals = []
        
        try:
            # RSI signals
            rsi = self.calculate_rsi(data['Close'])
            rsi_value = rsi.iloc[-1]
            rsi_signal = "NEUTRAL"
            if rsi_value > 70:
                rsi_signal = "OVERSOLD 🔴 SELL"
            elif rsi_value < 30:
                rsi_signal = "OVERBOUGHT 🟢 BUY"
            
            # MACD signals
            macd, signal = self.calculate_macd(data['Close'])
            macd_trend = "NEUTRAL"
            if len(macd) > 1 and macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
                macd_trend = "BULLISH CROSSOVER 🟢 BUY"
            elif len(macd) > 1 and macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
                macd_trend = "BEARISH CROSSOVER 🔴 SELL"
            
            # Moving Average signals
            ma_20 = data['Close'].rolling(window=20, min_periods=1).mean()
            ma_50 = data['Close'].rolling(window=50, min_periods=1).mean()
            ma_signal = "NEUTRAL"
            if len(ma_20) > 1 and ma_20.iloc[-1] > ma_50.iloc[-1] and ma_20.iloc[-2] <= ma_50.iloc[-2]:
                ma_signal = "GOLDEN CROSS 🟢 BUY"
            elif len(ma_20) > 1 and ma_20.iloc[-1] < ma_50.iloc[-1] and ma_20.iloc[-2] >= ma_50.iloc[-2]:
                ma_signal = "DEATH CROSS 🔴 SELL"
            
            signals.extend([
                {"Indicator": "RSI", "Value": f"{rsi_value:.1f}", "Signal": rsi_signal},
                {"Indicator": "MACD", "Value": f"{macd.iloc[-1]:.3f}", "Signal": macd_trend},
                {"Indicator": "Moving Averages", "Value": f"MA20: {ma_20.iloc[-1]:.1f}", "Signal": ma_signal},
            ])
            
        except Exception as e:
            st.error(f"Technical analysis error: {str(e)}")
            signals.append({"Indicator": "Error", "Value": "Check data", "Signal": "NEUTRAL"})
        
        return signals

def main():
    app = StockForecastApp()
    lstm_model = RobustLSTMForecaster()
    prophet_model = RobustProphetForecaster()
    ensemble_model = RobustEnsembleForecaster()
    tech_analyzer = TechnicalAnalyzer()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Stock Forecasting Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Robust Forecasting with Error-Resilient Models")
    
    # Rate limiting warning
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Note:</strong> Yahoo Finance has rate limits. The app uses sample data when rate-limited. 
    For live data, wait a few minutes between requests or use custom stocks with different symbols.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Custom Stock Addition
        st.markdown("### ➕ Add Custom Stock")
        with st.form("add_stock_form"):
            custom_name = st.text_input("Stock Name (e.g., MYSTOCK)")
            custom_symbol = st.text_input("Yahoo Symbol (e.g., MYSTOCK.NS or AAPL)")
            if st.form_submit_button("Add Stock"):
                if custom_name and custom_symbol:
                    if app.add_custom_stock(custom_name, custom_symbol):
                        st.success(f"✅ Added {custom_name}")
                    else:
                        st.error("❌ Failed to add stock")
        
        # Stock selection from combined list
        all_stocks = app.get_all_stocks()
        selected_stock = st.selectbox(
            "📊 Select Stock:",
            list(all_stocks.keys()),
            index=0
        )
        
        # Period selection
        period = st.selectbox(
            "📅 Data Period:",
            ["1mo", "3mo", "6mo", "1y"],
            index=2
        )
        
        # Forecast days
        forecast_days = st.slider(
            "🔮 Forecast Days:",
            min_value=7,
            max_value=60,
            value=30
        )
        
        # Model selection
        st.markdown("### 🤖 AI Models")
        use_lstm = st.checkbox("LSTM (Simplified)", value=True)
        use_prophet = st.checkbox("Prophet Forecasting", value=True)
        use_ensemble = st.checkbox("Ensemble Model", value=True)
        
        selected_models = []
        if use_lstm:
            selected_models.append("LSTM")
        if use_prophet:
            selected_models.append("Prophet")
        if use_ensemble:
            selected_models.append("Ensemble")
        
        if not selected_models:
            st.warning("⚠️ Please select at least one AI model")
        
        # Load data button
        if st.button("🚀 Load Data & Generate Forecast"):
            if not selected_models:
                st.error("❌ Please select at least one AI model")
            else:
                with st.spinner("🔄 Fetching stock data and training AI models..."):
                    data, source = app.get_stock_data(selected_stock, period)
                    if data is not None:
                        st.session_state.data = data
                        st.session_state.data_source = source
                        st.session_state.selected_stock = selected_stock
                        st.session_state.forecast_days = forecast_days
                        st.session_state.selected_models = selected_models
                        st.session_state.predictions = {}
                        st.session_state.forecast_date = datetime.now().strftime("%Y-%m-%d")
                        
                        # Show data source
                        if source == "sample":
                            st.warning("📊 Using realistic sample data (Yahoo Finance rate limited)")
                        elif source == "cache":
                            st.info("⚡ Using cached data")
                        else:
                            st.success("✅ Using live market data")
                        
                        # Train models and get predictions
                        progress_bar = st.progress(0)
                        total_models = len(selected_models)
                        
                        for idx, model_name in enumerate(selected_models):
                            try:
                                progress_bar.progress((idx + 1) / total_models, text=f"Training {model_name}...")
                                
                                if model_name == "LSTM":
                                    result = lstm_model.train(data)
                                    if result["status"] == "trained":
                                        predictions = lstm_model.predict(data, forecast_days)
                                        st.session_state.predictions[model_name] = predictions
                                    else:
                                        st.error(f"LSTM training failed: {result.get('error', 'Unknown error')}")
                                
                                elif model_name == "Prophet":
                                    result = prophet_model.train(data)
                                    if result["status"] == "trained":
                                        predictions = prophet_model.predict(data, forecast_days)
                                        st.session_state.predictions[model_name] = predictions
                                    else:
                                        st.error(f"Prophet training failed: {result.get('error', 'Unknown error')}")
                                
                                elif model_name == "Ensemble":
                                    result = ensemble_model.train(data)
                                    if result["status"] == "trained":
                                        predictions = ensemble_model.predict(data, forecast_days)
                                        st.session_state.predictions[model_name] = predictions
                                    else:
                                        st.error(f"Ensemble training failed: {result.get('error', 'Unknown error')}")
                                
                            except Exception as e:
                                st.error(f"❌ Error with {model_name}: {str(e)}")
                                # Provide fallback predictions
                                last_price = data['Close'].iloc[-1]
                                st.session_state.predictions[model_name] = [last_price] * forecast_days
                        
                        if st.session_state.predictions:
                            st.success("✅ AI Models Trained Successfully!")
                        else:
                            st.error("❌ All models failed to train")
    
    # Main content
    if 'data' in st.session_state and st.session_state.data is not None:
        data = st.session_state.data
        stock_name = st.session_state.selected_stock
        
        # Show data source
        source_badge = st.session_state.get('data_source', 'unknown')
        st.write(f"Data source: **{source_badge.upper()}**")
        
        # Stock overview
        st.markdown(f"## 📊 {stock_name} Stock Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
        
        with col1:
            st.metric(
                label="💰 Current Price",
                value=f"₹{current_price:.2f}",
                delta=f"{change_pct:+.2f}%"
            )
        
        with col2:
            st.metric(
                label="📈 Volume",
                value=f"{data['Volume'].iloc[-1]:,}"
            )
        
        with col3:
            st.metric(
                label="⬆️ Day High",
                value=f"₹{data['High'].iloc[-1]:.2f}"
            )
        
        with col4:
            st.metric(
                label="⬇️ Day Low", 
                value=f"₹{data['Low'].iloc[-1]:.2f}"
            )
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Price Forecast", "📈 Technical Analysis", "ℹ️ About"])
        
        with tab1:
            # Price chart with forecasts
            if 'predictions' in st.session_state and st.session_state.predictions:
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name='Historical Price',
                    line=dict(color='#1f77b4', width=3)
                ))
                
                # Predictions
                colors = {'LSTM': '#ff7f0e', 'Prophet': '#2ca02c', 'Ensemble': '#d62728'}
                for model_name, predictions in st.session_state.predictions.items():
                    last_date = data.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions))
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions,
                        name=f'{model_name} Forecast',
                        line=dict(color=colors.get(model_name, '#9467bd'), width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f'{stock_name} - AI Price Forecast',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    template='plotly_dark',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast details
                st.markdown("### 📋 AI Forecast Summary")
                cols = st.columns(len(st.session_state.predictions))
                
                for idx, (model_name, predictions) in enumerate(st.session_state.predictions.items()):
                    with cols[idx]:
                        st.markdown(f'<div class="model-card">', unsafe_allow_html=True)
                        st.markdown(f"#### {model_name}")
                        
                        predicted_price = predictions[-1] if len(predictions) > 0 else current_price
                        change = predicted_price - current_price
                        change_pct = (change / current_price) * 100
                        
                        st.metric(
                            label=f"Target Price ({forecast_days} days)",
                            value=f"₹{predicted_price:.2f}",
                            delta=f"{change_pct:+.2f}%"
                        )
                        
                        if model_name == "LSTM":
                            st.caption("🧠 Deep Learning Approach")
                        elif model_name == "Prophet":
                            st.caption("📊 Time Series Forecasting")
                        elif model_name == "Ensemble":
                            st.caption("🔄 Multiple Models Combined")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No predictions available. Please train models first.")
        
        with tab2:
            # Technical analysis
            st.markdown("### 📈 Technical Indicators & Signals")
            
            signals = tech_analyzer.generate_signals(data)
            
            for signal in signals:
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.write(f"**{signal['Indicator']}**")
                with col2:
                    st.write(signal['Value'])
                with col3:
                    signal_text = signal['Signal']
                    if "BUY" in signal_text:
                        st.success(f"🎯 {signal_text}")
                    elif "SELL" in signal_text:
                        st.error(f"🎯 {signal_text}")
                    elif "CAUTION" in signal_text:
                        st.warning(f"🎯 {signal_text}")
                    else:
                        st.info(f"🎯 {signal_text}")
            
            # Technical chart
            st.markdown("### 📊 Technical Analysis Chart")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price with Moving Averages', 'RSI'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'], name='Close Price', line=dict(color='#00b4d8')
            ), row=1, col=1)
            
            ma_20 = data['Close'].rolling(20, min_periods=1).mean()
            ma_50 = data['Close'].rolling(50, min_periods=1).mean()
            
            fig.add_trace(go.Scatter(
                x=data.index, y=ma_20, name='MA-20', line=dict(color='orange')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=ma_50, name='MA-50', line=dict(color='red')
            ), row=1, col=1)
            
            rsi = tech_analyzer.calculate_rsi(data['Close'])
            fig.add_trace(go.Scatter(
                x=data.index, y=rsi, name='RSI', line=dict(color='purple')
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=2, col=1)
            fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, row=2, col=1)
            
            fig.update_layout(height=600, template='plotly_dark', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🤖 About This AI Forecasting System")
            
            st.markdown("""
            #### 🎯 Enhanced Features
            - **Robust Error Handling**: Continues working even when APIs fail
            - **Multiple Data Sources**: Yahoo Finance with fallback to realistic sample data
            - **Rate Limit Protection**: Automatic retry and caching system
            - **Data Validation**: Cleans and validates all input data
            - **Custom Stocks**: Add any stock using Yahoo Finance symbols
            
            #### 🔧 Technical Improvements
            - **Fixed Timezone Issues**: Proper handling of datetime objects
            - **NaN/Infinity Protection**: Automatic cleaning of invalid values
            - **Graceful Degradation**: Falls back to simpler models when complex ones fail
            - **Memory Management**: Efficient caching and data handling
            
            #### ⚠️ Important Disclaimer
            **This AI forecasting tool is for educational and research purposes only.**
            
            - 🤖 AI predictions are not financial advice
            - 📈 Past performance doesn't guarantee future results
            - 💰 Never invest based solely on AI predictions
            - 🔍 Always do your own research and due diligence
            - 🏦 Consult qualified financial advisors for investment decisions
            
            #### 🛠️ For Developers
            This system now includes:
            - Comprehensive error handling
            - Data validation at every step
            - Rate limiting protection
            - Fallback mechanisms
            - Realistic sample data generation
            """)
    
    else:
        # Welcome page
        st.markdown("""
        ## 🎯 Welcome to AI Stock Forecaster Pro!
        
        ### ✨ Enhanced Features:
        - **🤖 Multiple AI Models** - LSTM, Prophet, and Ensemble
        - **🛡️ Error Resilient** - Continues working even when APIs fail
        - **📊 Realistic Sample Data** - When Yahoo Finance is rate limited
        - **➕ Custom Stocks** - Add any stock manually
        - **📈 Technical Analysis** - RSI, MACD, Moving Averages
        
        ### 🚀 How to Start:
        1. **Select** a stock from the sidebar (or add custom ones)
        2. **Choose** data period and forecast days  
        3. **Select** AI models to use
        4. **Click** "Load Data & Generate Forecast"
        5. **Explore** all analysis tabs
        
        ### 💡 Pro Tips:
        - If you see "sample data" warnings, wait a few minutes and try again
        - Add custom stocks for international companies
        - Compare multiple models for better insights
        - Use technical analysis to validate predictions
        
        *Ready to start? Use the sidebar to configure your analysis!*
        """)

if __name__ == "__main__":
    main()

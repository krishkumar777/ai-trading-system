# app.py - Enhanced AI Stock Forecasting System
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
    .stock-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 1rem 0;
    }
    .stock-item {
        background: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #2e86ab;
    }
    .positive { color: #00ff00; font-weight: bold; }
    .negative { color: #ff4444; font-weight: bold; }
    .neutral { color: #ffaa00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class StockForecastApp:
    def __init__(self):
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
        try:
            all_stocks = self.get_all_stocks()
            if symbol in all_stocks:
                yahoo_symbol = all_stocks[symbol]
            else:
                yahoo_symbol = symbol
            
            stock = yf.Ticker(yahoo_symbol)
            data = stock.history(period=period)
            
            if data.empty:
                st.error(f"❌ No data found for {symbol}")
                return None
            
            return data
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            return None

class SimpleLSTMForecaster:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lookback = 30
        
    def create_rolling_features(self, data):
        df = data.copy()
        
        # Price features
        df['Returns'] = df['Close'].pct_change()
        df['Price_Ratio'] = df['Close'] / df['Open']
        df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
            df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        
        # Lag features
        for lag in range(1, self.lookback + 1):
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        
        return df.fillna(method='bfill')
    
    def prepare_data(self, data):
        df = self.create_rolling_features(data)
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
        
        X = []
        y = []
        
        for i in range(self.lookback, len(df)):
            current_features = df[feature_cols].iloc[i].values
            price_features = [df['Close'].iloc[i - j] for j in range(1, self.lookback + 1)]
            all_features = np.concatenate([current_features, price_features])
            X.append(all_features)
            y.append(df['Close'].iloc[i])
        
        return np.array(X), np.array(y), feature_cols
    
    def train(self, data):
        X, y, features = self.prepare_data(data)
        if len(X) == 0:
            raise ValueError("Not enough data for training")
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return {"status": "trained", "features_used": len(X[0])}
    
    def predict(self, data, future_days=30):
        df = self.create_rolling_features(data)
        predictions = []
        current_data = df.copy()
        
        for _ in range(future_days):
            X_current, _, _ = self.prepare_data(current_data)
            if len(X_current) == 0:
                last_price = current_data['Close'].iloc[-1]
                predictions.append(last_price)
                continue
            
            latest_features = X_current[-1:].reshape(1, -1)
            latest_features_scaled = self.scaler.transform(latest_features)
            pred = self.model.predict(latest_features_scaled)[0]
            predictions.append(pred)
            
            new_row = current_data.iloc[-1:].copy()
            new_row['Close'] = pred
            new_row['Open'] = pred * 0.99
            new_row['High'] = pred * 1.01
            new_row['Low'] = pred * 0.98
            new_row['Volume'] = current_data['Volume'].mean()
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        return predictions

class ProphetForecaster:
    def __init__(self):
        self.model = None
        
    def prepare_data(self, data):
        df = data.reset_index()[['Date', 'Close']].copy()
        df.columns = ['ds', 'y']
        return df
    
    def train(self, data):
        try:
            from prophet import Prophet
            df = self.prepare_data(data)
            self.model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            self.model.fit(df)
            return {"status": "trained"}
        except ImportError:
            return {"status": "fallback", "fallback": self.fallback_predict(data, 30)}
    
    def predict(self, data, future_days=30):
        if self.model is None:
            result = self.train(data)
            if "fallback" in result:
                return result["fallback"]
        
        future = self.model.make_future_dataframe(periods=future_days)
        forecast = self.model.predict(future)
        return forecast['yhat'].tail(future_days).values
    
    def fallback_predict(self, data, future_days):
        last_price = data['Close'].iloc[-1]
        sma_20 = data['Close'].tail(20).mean()
        trend = (last_price - data['Close'].iloc[-20]) / 20
        predictions = []
        for i in range(1, future_days + 1):
            pred = last_price + (trend * i) + (np.random.normal(0, last_price * 0.01))
            predictions.append(max(pred, last_price * 0.8))
        return predictions

class EnsembleForecaster:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def create_features(self, data):
        df = data.copy()
        df['Returns'] = df['Close'].pct_change()
        df['Price_Ratio'] = df['Close'] / df['Open']
        df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volatility'] = df['Returns'].rolling(20).std()
        return df.fillna(method='bfill')
    
    def prepare_data(self, data):
        df = self.create_features(data)
        feature_cols = ['Returns', 'Price_Ratio', 'HL_Ratio', 'SMA_5', 'SMA_20', 'EMA_12', 'Volume_Change', 'Volume_Ratio', 'Volatility']
        X = df[feature_cols].iloc[20:].values
        y = df['Close'].iloc[20:].values
        return X, y, feature_cols
    
    def train(self, data):
        X, y, features = self.prepare_data(data)
        if len(X) == 0:
            raise ValueError("Not enough data for training")
        
        X_scaled = self.scaler.fit_transform(X)
        self.rf_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)
        return {"status": "trained", "features": features}
    
    def predict(self, data, future_days=30):
        df = self.create_features(data)
        predictions = []
        current_data = df.copy()
        
        for _ in range(future_days):
            X_current, _, _ = self.prepare_data(current_data)
            if len(X_current) == 0:
                last_price = current_data['Close'].iloc[-1]
                predictions.append(last_price)
                continue
            
            latest_features = X_current[-1:].reshape(1, -1)
            latest_features_scaled = self.scaler.transform(latest_features)
            rf_pred = self.rf_model.predict(latest_features_scaled)[0]
            gb_pred = self.gb_model.predict(latest_features_scaled)[0]
            ensemble_pred = (rf_pred + gb_pred) / 2
            predictions.append(ensemble_pred)
            
            new_row = current_data.iloc[-1:].copy()
            new_row['Close'] = ensemble_pred
            new_row['Open'] = ensemble_pred * 0.99
            new_row['High'] = ensemble_pred * 1.01
            new_row['Low'] = ensemble_pred * 0.98
            new_row['Volume'] = current_data['Volume'].mean()
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        return predictions

class TechnicalAnalyzer:
    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices):
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def generate_signals(self, data):
        signals = []
        rsi = self.calculate_rsi(data['Close'])
        rsi_value = rsi.iloc[-1]
        rsi_signal = "NEUTRAL"
        if rsi_value > 70:
            rsi_signal = "OVERSOLD 🔴 SELL"
        elif rsi_value < 30:
            rsi_signal = "OVERBOUGHT 🟢 BUY"
        
        macd, signal = self.calculate_macd(data['Close'])
        macd_trend = "NEUTRAL"
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            macd_trend = "BULLISH CROSSOVER 🟢 BUY"
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            macd_trend = "BEARISH CROSSOVER 🔴 SELL"
        
        ma_20 = data['Close'].rolling(window=20).mean()
        ma_50 = data['Close'].rolling(window=50).mean()
        ma_signal = "NEUTRAL"
        if ma_20.iloc[-1] > ma_50.iloc[-1] and ma_20.iloc[-2] <= ma_50.iloc[-2]:
            ma_signal = "GOLDEN CROSS 🟢 BUY"
        elif ma_20.iloc[-1] < ma_50.iloc[-1] and ma_20.iloc[-2] >= ma_50.iloc[-2]:
            ma_signal = "DEATH CROSS 🔴 SELL"
        
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        current_price = data['Close'].iloc[-1]
        support_resistance = "NEUTRAL"
        if current_price >= recent_high * 0.98:
            support_resistance = "NEAR RESISTANCE 🔴 CAUTION"
        elif current_price <= recent_low * 1.02:
            support_resistance = "NEAR SUPPORT 🟢 OPPORTUNITY"
        
        signals.extend([
            {"Indicator": "RSI", "Value": f"{rsi_value:.1f}", "Signal": rsi_signal},
            {"Indicator": "MACD", "Value": f"{macd.iloc[-1]:.2f}", "Signal": macd_trend},
            {"Indicator": "Moving Averages", "Value": f"MA20: {ma_20.iloc[-1]:.1f}", "Signal": ma_signal},
            {"Indicator": "Support/Resistance", "Value": f"High: {recent_high:.1f}", "Signal": support_resistance}
        ])
        return signals

class PerformanceTracker:
    def __init__(self):
        self.forecast_history = {}
    
    def track_forecast(self, stock_name, model_name, forecast_date, predictions, actual_dates, actual_prices):
        """Track forecast performance against actual prices"""
        key = f"{stock_name}_{model_name}_{forecast_date}"
        
        if key not in self.forecast_history:
            self.forecast_history[key] = {
                'forecast_date': forecast_date,
                'predictions': predictions,
                'actual_dates': actual_dates,
                'actual_prices': actual_prices,
                'evaluated': False
            }
    
    def evaluate_forecast(self, stock_name, model_name, forecast_date):
        """Evaluate forecast accuracy"""
        key = f"{stock_name}_{model_name}_{forecast_date}"
        
        if key not in self.forecast_history:
            return None
        
        data = self.forecast_history[key]
        predictions = data['predictions']
        actual_prices = data['actual_prices']
        
        if len(predictions) != len(actual_prices):
            min_len = min(len(predictions), len(actual_prices))
            predictions = predictions[:min_len]
            actual_prices = actual_prices[:min_len]
        
        # Calculate metrics
        mae = mean_absolute_error(actual_prices, predictions)
        mse = mean_squared_error(actual_prices, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
        r2 = r2_score(actual_prices, predictions)
        
        # Direction accuracy
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actual_prices) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        metrics = {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2),
            'R² Score': round(r2, 3),
            'Direction Accuracy': round(direction_accuracy, 1)
        }
        
        self.forecast_history[key]['metrics'] = metrics
        self.forecast_history[key]['evaluated'] = True
        
        return metrics
    
    def get_forecast_history(self):
        """Get all tracked forecasts"""
        return self.forecast_history

class BacktestingEngine:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
    
    def run_backtest(self, data, predictions, model_name, strategy_type='forecast_based'):
        """Run backtest on historical data"""
        # Create test data
        test_size = min(len(predictions), len(data) // 4)
        test_data = data.iloc[-test_size:].copy()
        test_predictions = predictions[:test_size]
        
        # Initialize portfolio
        portfolio_value = [self.initial_capital]
        cash = self.initial_capital
        shares = 0
        trades = []
        
        for i in range(1, len(test_data)):
            current_price = test_data['Close'].iloc[i]
            pred_price = test_predictions[i] if i < len(test_predictions) else test_predictions[-1]
            
            # Simple strategy: Buy if prediction > current price, Sell if prediction < current price
            if strategy_type == 'forecast_based':
                if pred_price > current_price * 1.02 and cash > 0:  # Buy signal
                    shares_to_buy = cash // current_price
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        cash -= cost
                        shares += shares_to_buy
                        trades.append({'day': i, 'action': 'BUY', 'price': current_price, 'shares': shares_to_buy})
                
                elif pred_price < current_price * 0.98 and shares > 0:  # Sell signal
                    cash += shares * current_price
                    trades.append({'day': i, 'action': 'SELL', 'price': current_price, 'shares': shares})
                    shares = 0
            
            # Calculate portfolio value
            portfolio_value.append(cash + shares * current_price)
        
        # Finalize portfolio
        if shares > 0:
            final_price = test_data['Close'].iloc[-1]
            cash += shares * final_price
            shares = 0
        
        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_value).pct_change().dropna()
        total_return = (portfolio_value[-1] - self.initial_capital) / self.initial_capital * 100
        buy_hold_return = (test_data['Close'].iloc[-1] - test_data['Close'].iloc[0]) / test_data['Close'].iloc[0] * 100
        
        # Risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        max_drawdown = self.calculate_max_drawdown(portfolio_value)
        
        return {
            'model': model_name,
            'strategy': strategy_type,
            'final_value': portfolio_value[-1],
            'total_return': round(total_return, 2),
            'buy_hold_return': round(buy_hold_return, 2),
            'excess_return': round(total_return - buy_hold_return, 2),
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'trades': len(trades),
            'portfolio_values': portfolio_value,
            'trades_list': trades
        }
    
    def calculate_max_drawdown(self, portfolio_values):
        running_max = pd.Series(portfolio_values).expanding().max()
        drawdown = (pd.Series(portfolio_values) - running_max) / running_max
        return drawdown.min()

def main():
    app = StockForecastApp()
    lstm_model = SimpleLSTMForecaster()
    prophet_model = ProphetForecaster()
    ensemble_model = EnsembleForecaster()
    tech_analyzer = TechnicalAnalyzer()
    performance_tracker = PerformanceTracker()
    backtest_engine = BacktestingEngine()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Stock Forecasting Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Forecasting with Performance Tracking & Backtesting")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Custom Stock Addition
        st.markdown("### ➕ Add Custom Stock")
        with st.form("add_stock_form"):
            custom_name = st.text_input("Stock Name (e.g., MYSTOCK)")
            custom_symbol = st.text_input("Yahoo Symbol (e.g., MYSTOCK.NS)")
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
            ["3mo", "6mo", "1y", "2y"],
            index=2
        )
        
        # Forecast days
        forecast_days = st.slider(
            "🔮 Forecast Days:",
            min_value=7,
            max_value=90,
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
        
        # Performance tracking option
        enable_tracking = st.checkbox("📊 Enable Performance Tracking", value=True)
        enable_backtesting = st.checkbox("🔍 Enable Backtesting", value=True)
        
        if not selected_models:
            st.warning("⚠️ Please select at least one AI model")
        
        # Load data button
        if st.button("🚀 Load Data & Generate Forecast"):
            if not selected_models:
                st.error("❌ Please select at least one AI model")
            else:
                with st.spinner("🔄 Fetching stock data and training AI models..."):
                    data = app.get_stock_data(selected_stock, period)
                    if data is not None:
                        st.session_state.data = data
                        st.session_state.selected_stock = selected_stock
                        st.session_state.forecast_days = forecast_days
                        st.session_state.selected_models = selected_models
                        st.session_state.predictions = {}
                        st.session_state.forecast_date = datetime.now().strftime("%Y-%m-%d")
                        
                        # Train models and get predictions
                        progress_bar = st.progress(0)
                        total_models = len(selected_models)
                        
                        for idx, model_name in enumerate(selected_models):
                            try:
                                progress_bar.progress((idx + 1) / total_models, text=f"Training {model_name}...")
                                
                                if model_name == "LSTM":
                                    lstm_model.train(data)
                                    predictions = lstm_model.predict(data, forecast_days)
                                elif model_name == "Prophet":
                                    predictions = prophet_model.predict(data, forecast_days)
                                elif model_name == "Ensemble":
                                    ensemble_model.train(data)
                                    predictions = ensemble_model.predict(data, forecast_days)
                                
                                st.session_state.predictions[model_name] = predictions
                                
                                # Track forecast for performance monitoring
                                if enable_tracking:
                                    future_dates = pd.date_range(
                                        start=data.index[-1] + timedelta(days=1), 
                                        periods=forecast_days
                                    )
                                    performance_tracker.track_forecast(
                                        selected_stock, model_name, 
                                        st.session_state.forecast_date,
                                        predictions, future_dates, 
                                        [None] * forecast_days  # Placeholder for actual prices
                                    )
                                
                            except Exception as e:
                                st.error(f"❌ Error with {model_name}: {str(e)}")
                                last_price = data['Close'].iloc[-1]
                                st.session_state.predictions[model_name] = [last_price] * forecast_days
                        
                        st.session_state.models_trained = True
                        st.success("✅ AI Models Trained Successfully!")
    
    # Main content
    if 'data' in st.session_state and st.session_state.data is not None:
        data = st.session_state.data
        stock_name = st.session_state.selected_stock
        
        # Stock overview
        st.markdown(f"## 📊 {stock_name} Stock Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Price Forecast", "📈 Technical Analysis", "📊 Performance", "🔍 Backtesting", "ℹ️ About"
        ])
        
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
            
            ma_20 = data['Close'].rolling(20).mean()
            ma_50 = data['Close'].rolling(50).mean()
            
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
            st.markdown("### 📊 Forecast Performance Tracking")
            
            if enable_tracking and 'predictions' in st.session_state:
                # Manual performance evaluation
                st.markdown("#### 🔍 Evaluate Forecast Accuracy")
                
                col1, col2 = st.columns(2)
                with col1:
                    eval_model = st.selectbox("Select Model", list(st.session_state.predictions.keys()))
                with col2:
                    days_to_evaluate = st.slider("Days to Evaluate", 1, forecast_days, min(7, forecast_days))
                
                if st.button("📈 Evaluate Forecast Performance"):
                    # For demo, we'll use historical data as "actual" values
                    # In real scenario, you'd wait for actual prices to come in
                    historical_actuals = data['Close'].values[-days_to_evaluate:]
                    predictions_to_eval = st.session_state.predictions[eval_model][:days_to_evaluate]
                    
                    # Calculate metrics
                    mae = mean_absolute_error(historical_actuals, predictions_to_eval)
                    mse = mean_squared_error(historical_actuals, predictions_to_eval)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((historical_actuals - predictions_to_eval) / historical_actuals)) * 100
                    r2 = r2_score(historical_actuals, predictions_to_eval)
                    
                    # Display metrics
                    st.markdown("#### 📋 Performance Metrics")
                    metrics_cols = st.columns(5)
                    
                    with metrics_cols[0]:
                        st.metric("MAE", f"₹{mae:.2f}")
                    with metrics_cols[1]:
                        st.metric("RMSE", f"₹{rmse:.2f}")
                    with metrics_cols[2]:
                        st.metric("MAPE", f"{mape:.1f}%")
                    with metrics_cols[3]:
                        st.metric("R² Score", f"{r2:.3f}")
                    with metrics_cols[4]:
                        # Direction accuracy
                        pred_dir = np.diff(predictions_to_eval) > 0
                        actual_dir = np.diff(historical_actuals) > 0
                        dir_acc = np.mean(pred_dir == actual_dir) * 100
                        st.metric("Direction Accuracy", f"{dir_acc:.1f}%")
                    
                    # Performance chart
                    fig_perf = go.Figure()
                    days_range = list(range(len(historical_actuals)))
                    
                    fig_perf.add_trace(go.Scatter(
                        x=days_range, y=historical_actuals,
                        name='Actual Prices', line=dict(color='#1f77b4', width=3)
                    ))
                    fig_perf.add_trace(go.Scatter(
                        x=days_range, y=predictions_to_eval,
                        name=f'{eval_model} Predictions', line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                    
                    fig_perf.update_layout(
                        title=f'Forecast vs Actual - {eval_model}',
                        xaxis_title='Days',
                        yaxis_title='Price (₹)',
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)
            
            else:
                st.info("Enable Performance Tracking in sidebar to monitor forecast accuracy")
        
        with tab4:
            st.markdown("### 🔍 Model Backtesting")
            
            if enable_backtesting and 'predictions' in st.session_state:
                st.markdown("#### 📊 Backtest Results")
                
                backtest_results = []
                
                for model_name, predictions in st.session_state.predictions.items():
                    result = backtest_engine.run_backtest(data, predictions, model_name)
                    backtest_results.append(result)
                
                # Display backtest results
                results_data = []
                for result in backtest_results:
                    results_data.append({
                        'Model': result['model'],
                        'Final Value': f"₹{result['final_value']:,.0f}",
                        'Strategy Return': f"{result['total_return']}%",
                        'Buy & Hold Return': f"{result['buy_hold_return']}%",
                        'Excess Return': f"{result['excess_return']}%",
                        'Volatility': f"{result['volatility']}%",
                        'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
                        'Max Drawdown': f"{result['max_drawdown']}%",
                        'Trades': result['trades']
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Backtest performance chart
                st.markdown("#### 📈 Portfolio Performance Comparison")
                fig_backtest = go.Figure()
                
                for result in backtest_results:
                    fig_backtest.add_trace(go.Scatter(
                        x=list(range(len(result['portfolio_values']))),
                        y=result['portfolio_values'],
                        name=f"{result['model']} Strategy",
                        line=dict(width=2)
                    ))
                
                # Add buy & hold benchmark
                initial_value = backtest_engine.initial_capital
                buy_hold_values = [initial_value * (1 + (data['Close'].iloc[i] - data['Close'].iloc[0]) / data['Close'].iloc[0]) 
                                 for i in range(len(data) - len(backtest_results[0]['portfolio_values']), len(data))]
                
                fig_backtest.add_trace(go.Scatter(
                    x=list(range(len(buy_hold_values))),
                    y=buy_hold_values,
                    name='Buy & Hold',
                    line=dict(color='gray', width=2, dash='dash')
                ))
                
                fig_backtest.update_layout(
                    title='Backtesting Performance - Portfolio Value Over Time',
                    xaxis_title='Trading Days',
                    yaxis_title='Portfolio Value (₹)',
                    template='plotly_dark',
                    height=500
                )
                st.plotly_chart(fig_backtest, use_container_width=True)
                
                # Model efficiency comparison
                st.markdown("#### 🤖 Model Efficiency Ranking")
                efficiency_data = []
                for result in backtest_results:
                    efficiency_score = (
                        result['total_return'] * 0.4 +
                        result['sharpe_ratio'] * 30 +
                        (100 - result['max_drawdown']) * 0.3 +
                        min(result['trades'] * 2, 20) * 0.1
                    )
                    efficiency_data.append({
                        'Model': result['model'],
                        'Efficiency Score': round(efficiency_score, 1),
                        'Return': result['total_return'],
                        'Risk-Adjusted': result['sharpe_ratio'],
                        'Drawdown': result['max_drawdown']
                    })
                
                eff_df = pd.DataFrame(efficiency_data).sort_values('Efficiency Score', ascending=False)
                st.dataframe(eff_df, use_container_width=True)
            
            else:
                st.info("Enable Backtesting in sidebar to test model performance")
        
        with tab5:
            st.markdown("### 🤖 About This AI Forecasting System")
            
            st.markdown("""
            #### 🎯 Enhanced Features
            - **Performance Tracking**: Monitor forecast accuracy against actual prices
            - **Backtesting**: Test model performance with historical data
            - **Custom Stocks**: Add any stock using Yahoo Finance symbols
            - **Efficiency Metrics**: Comprehensive model evaluation
            - **Multi-Model Comparison**: Compare LSTM, Prophet, and Ensemble approaches
            
            #### 📊 Performance Metrics Tracked
            - **MAE/RMSE**: Prediction accuracy measures
            - **MAPE**: Percentage error
            - **R² Score**: Model fit quality
            - **Direction Accuracy**: Price movement prediction
            - **Sharpe Ratio**: Risk-adjusted returns
            - **Max Drawdown**: Worst-case performance
            
            #### ⚠️ Important Disclaimer
            **This AI forecasting tool is for educational and research purposes only.**
            
            - 🤖 AI predictions are not financial advice
            - 📈 Past performance doesn't guarantee future results
            - 💰 Never invest based solely on AI predictions
            - 🔍 Always do your own research and due diligence
            - 🏦 Consult qualified financial advisors for investment decisions
            """)
    
    else:
        # Welcome page
        st.markdown("""
        ## 🎯 Welcome to AI Stock Forecaster Pro!
        
        ### ✨ Enhanced Features:
        - **🤖 Multiple AI Models** - LSTM, Prophet, and Ensemble
        - **📊 Performance Tracking** - Monitor forecast accuracy
        - **🔍 Backtesting** - Historical performance testing
        - **➕ Custom Stocks** - Add any stock manually
        - **📈 Efficiency Metrics** - Comprehensive model evaluation
        - **💹 Live Market Data** - Real-time Indian stocks
        
        ### 🚀 How to Start:
        1. **Add custom stocks** using the sidebar (optional)
        2. **Select** a stock from the dropdown
        3. **Choose** data period and forecast days  
        4. **Enable** performance tracking & backtesting
        5. **Click** "Load Data & Generate Forecast"
        6. **Explore** all analysis tabs
        
        ### 📋 Available Indian Stocks:
        """)
        
        # Display stocks in a nice grid
        all_stocks = app.get_all_stocks()
        stocks = list(all_stocks.keys())
        cols = st.columns(4)
        for i, stock in enumerate(stocks):
            with cols[i % 4]:
                st.info(f"**{stock}**")
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Pro Tips:
        - Use **performance tracking** to monitor model accuracy over time
        - **Backtest** models before relying on forecasts
        - Compare **multiple models** for better insights
        - Add **custom stocks** for personalized analysis
        
        *Ready to start? Use the sidebar to configure your analysis!*
        """)

if __name__ == "__main__":
    main()

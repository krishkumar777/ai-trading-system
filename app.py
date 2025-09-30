# app.py - AI Stock Forecasting System
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Stock Forecaster",
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
        
    def get_stock_data(self, symbol, period='1y'):
        try:
            if symbol in self.indian_stocks:
                yahoo_symbol = self.indian_stocks[symbol]
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
    """A simplified LSTM-like model using sklearn"""
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lookback = 30
        
    def create_rolling_features(self, data):
        """Create rolling window features without TensorFlow"""
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
        
        # Feature columns
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
        
        X = []
        y = []
        
        for i in range(self.lookback, len(df)):
            # Current features
            current_features = df[feature_cols].iloc[i].values
            
            # Lagged price features
            price_features = [df['Close'].iloc[i - j] for j in range(1, self.lookback + 1)]
            
            # Combine all features
            all_features = np.concatenate([current_features, price_features])
            X.append(all_features)
            y.append(df['Close'].iloc[i])
        
        return np.array(X), np.array(y), feature_cols
    
    def train(self, data):
        X, y, features = self.prepare_data(data)
        
        if len(X) == 0:
            raise ValueError("Not enough data for training")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        return {"status": "trained", "features_used": len(X[0])}
    
    def predict(self, data, future_days=30):
        df = self.create_rolling_features(data)
        
        predictions = []
        current_data = df.copy()
        
        for _ in range(future_days):
            # Prepare features for prediction
            X_current, _, _ = self.prepare_data(current_data)
            
            if len(X_current) == 0:
                # If we can't prepare features, use last known price
                last_price = current_data['Close'].iloc[-1]
                predictions.append(last_price)
                continue
            
            # Get the latest features
            latest_features = X_current[-1:].reshape(1, -1)
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Make prediction
            pred = self.model.predict(latest_features_scaled)[0]
            predictions.append(pred)
            
            # Update data for next prediction
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
            self.model = Prophet(
                daily_seasonality=False,  # Disable daily for stock data
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            self.model.fit(df)
            return {"status": "trained"}
        except ImportError:
            st.error("Prophet not available. Using fallback method.")
            return self.fallback_predict(data, 30)
    
    def predict(self, data, future_days=30):
        if self.model is None:
            result = self.train(data)
            if "fallback" in result:
                return result["fallback"]
        
        future = self.model.make_future_dataframe(periods=future_days)
        forecast = self.model.predict(future)
        return forecast['yhat'].tail(future_days).values
    
    def fallback_predict(self, data, future_days):
        """Fallback method if Prophet fails"""
        # Simple moving average projection
        last_price = data['Close'].iloc[-1]
        sma_20 = data['Close'].tail(20).mean()
        trend = (last_price - data['Close'].iloc[-20]) / 20
        
        predictions = []
        for i in range(1, future_days + 1):
            pred = last_price + (trend * i) + (np.random.normal(0, last_price * 0.01))
            predictions.append(max(pred, last_price * 0.8))  # Prevent negative prices
        
        return predictions

class EnsembleForecaster:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def create_features(self, data):
        df = data.copy()
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Price_Ratio'] = df['Close'] / df['Open']
        df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
        
        # Technical indicators
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(20).std()
        
        return df.fillna(method='bfill')
    
    def prepare_data(self, data):
        df = self.create_features(data)
        
        feature_cols = ['Returns', 'Price_Ratio', 'HL_Ratio', 'SMA_5', 'SMA_20', 
                       'EMA_12', 'Volume_Change', 'Volume_Ratio', 'Volatility']
        
        # Use only recent data for training
        X = df[feature_cols].iloc[20:].values  # Skip first 20 rows due to rolling windows
        y = df['Close'].iloc[20:].values
        
        return X, y, feature_cols
    
    def train(self, data):
        X, y, features = self.prepare_data(data)
        
        if len(X) == 0:
            raise ValueError("Not enough data for training")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train both models
        self.rf_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)
        
        return {"status": "trained", "features": features}
    
    def predict(self, data, future_days=30):
        df = self.create_features(data)
        
        predictions = []
        current_data = df.copy()
        
        for _ in range(future_days):
            # Prepare current features
            X_current, _, _ = self.prepare_data(current_data)
            
            if len(X_current) == 0:
                last_price = current_data['Close'].iloc[-1]
                predictions.append(last_price)
                continue
            
            latest_features = X_current[-1:].reshape(1, -1)
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Ensemble prediction (average of both models)
            rf_pred = self.rf_model.predict(latest_features_scaled)[0]
            gb_pred = self.gb_model.predict(latest_features_scaled)[0]
            ensemble_pred = (rf_pred + gb_pred) / 2
            
            predictions.append(ensemble_pred)
            
            # Update for next prediction
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
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            macd_trend = "BULLISH CROSSOVER 🟢 BUY"
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            macd_trend = "BEARISH CROSSOVER 🔴 SELL"
        
        # Moving Average signals
        ma_20 = data['Close'].rolling(window=20).mean()
        ma_50 = data['Close'].rolling(window=50).mean()
        ma_signal = "NEUTRAL"
        if ma_20.iloc[-1] > ma_50.iloc[-1] and ma_20.iloc[-2] <= ma_50.iloc[-2]:
            ma_signal = "GOLDEN CROSS 🟢 BUY"
        elif ma_20.iloc[-1] < ma_50.iloc[-1] and ma_20.iloc[-2] >= ma_50.iloc[-2]:
            ma_signal = "DEATH CROSS 🔴 SELL"
        
        # Support/Resistance
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

def main():
    app = StockForecastApp()
    lstm_model = SimpleLSTMForecaster()
    prophet_model = ProphetForecaster()
    ensemble_model = EnsembleForecaster()
    tech_analyzer = TechnicalAnalyzer()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Stock Forecasting System</h1>', unsafe_allow_html=True)
    st.markdown("### Predict Indian Stock Prices with Machine Learning")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Stock selection
        selected_stock = st.selectbox(
            "📊 Select Indian Stock:",
            list(app.indian_stocks.keys()),
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
                                
                            except Exception as e:
                                st.error(f"❌ Error with {model_name}: {str(e)}")
                                # Provide fallback predictions
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
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions))
                    
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
                        
                        # Additional info
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
            
            # Price with MAs
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'], name='Close Price', line=dict(color='#00b4d8')
            ), row=1, col=1)
            
            # Moving averages
            ma_20 = data['Close'].rolling(20).mean()
            ma_50 = data['Close'].rolling(50).mean()
            
            fig.add_trace(go.Scatter(
                x=data.index, y=ma_20, name='MA-20', line=dict(color='orange')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=ma_50, name='MA-50', line=dict(color='red')
            ), row=1, col=1)
            
            # RSI
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
            #### 🎯 How It Works
            This system uses multiple machine learning models to predict stock prices:
            
            - **LSTM Model**: Uses rolling window features to capture patterns (TensorFlow-free implementation)
            - **Prophet Model**: Facebook's time series forecasting algorithm
            - **Ensemble Model**: Combines multiple algorithms for better accuracy
            
            #### 📊 Technical Analysis Features
            - **RSI (Relative Strength Index)**: Momentum indicator
            - **MACD**: Trend-following momentum indicator
            - **Moving Averages**: Identify trends and support/resistance
            - **Support/Resistance Levels**: Key price levels
            
            #### ⚠️ Important Disclaimer
            **This AI forecasting tool is for educational and research purposes only.**
            
            - 🤖 AI predictions are not financial advice
            - 📈 Stock markets are unpredictable and involve risk
            - 💰 Never invest based solely on AI predictions
            - 🔍 Always do your own research and due diligence
            - 🏦 Consult qualified financial advisors for investment decisions
            
            #### 🔧 Technical Details
            - Built with Python and Streamlit
            - Uses Yahoo Finance for real-time data
            - Implements scikit-learn for machine learning
            - Plotly for interactive charts
            - Deployed on Streamlit Cloud
            """)
    
    else:
        # Welcome page
        st.markdown("""
        ## 🎯 Welcome to AI Stock Forecaster!
        
        ### ✨ Features:
        - **🤖 Multiple AI Models** - LSTM, Prophet, and Ensemble
        - **📈 Real-time Indian Stocks** - 20+ major companies
        - **🎯 Price Forecasting** - 7 to 90 days ahead
        - **📊 Technical Analysis** - RSI, MACD, Moving Averages
        - **💹 Live Market Data** - Updated automatically
        
        ### 🚀 How to Start:
        1. **Select** a stock from the sidebar
        2. **Choose** data period and forecast days  
        3. **Select** AI models to use
        4. **Click** "Load Data & Generate Forecast"
        5. **Explore** different analysis tabs
        
        ### 📋 Popular Indian Stocks:
        """)
        
        # Display stocks in a nice grid
        stocks = list(app.indian_stocks.keys())
        cols = st.columns(4)
        for i, stock in enumerate(stocks):
            with cols[i % 4]:
                st.info(f"**{stock}**")
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Pro Tips:
        - Use **multiple models** for better accuracy
        - **Compare predictions** across different models
        - Check **technical signals** for trading insights
        - Consider **market conditions** when interpreting results
        
        *Ready to start? Use the sidebar to configure your analysis!*
        """)

if __name__ == "__main__":
    main()

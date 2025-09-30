# stock_forecast_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
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
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin: 1rem 0;
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
    .positive {
        color: #00ff00;
        font-weight: bold;
    }
    .negative {
        color: #ff4444;
        font-weight: bold;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        font-weight: bold;
    }
    .stSelectbox, .stSlider, .stCheckbox {
        margin: 0.5rem 0;
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
            'HCL TECH': 'HCLTECH.NS'
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

class LSTMForecaster:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.lookback = 60
        
    def create_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def add_technical_indicators(self, data):
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        
        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        
        return data.fillna(method='bfill')
    
    def prepare_data(self, data):
        data = self.add_technical_indicators(data.copy())
        feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
        available_columns = [col for col in feature_columns if col in data.columns]
        
        scaled_data = self.scaler.fit_transform(data[available_columns])
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y), available_columns
    
    def train(self, data):
        X, y, features = self.prepare_data(data)
        if len(X) == 0:
            raise ValueError("Not enough data for training")
            
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.model = self.create_model((X.shape[1], X.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=0,
            shuffle=False
        )
        return history
    
    def predict(self, data, future_days=30):
        if self.model is None:
            self.train(data)
            
        X, _, features = self.prepare_data(data)
        last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(future_days):
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            new_row = current_sequence[0, -1].copy()
            new_row[0] = next_pred
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = new_row
        
        dummy_array = np.zeros((len(predictions), len(features)))
        dummy_array[:, 0] = predictions
        predictions = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        return predictions

class ProphetForecaster:
    def __init__(self):
        self.model = None
        
    def prepare_data(self, data):
        df = data.reset_index()[['Date', 'Close']].copy()
        df.columns = ['ds', 'y']
        return df
    
    def train(self, data):
        df = self.prepare_data(data)
        self.model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        self.model.fit(df)
    
    def predict(self, data, future_days=30):
        if self.model is None:
            self.train(data)
            
        future = self.model.make_future_dataframe(periods=future_days)
        forecast = self.model.predict(future)
        return forecast['yhat'].tail(future_days).values

class EnsembleForecaster:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.lookback = 30
        
    def add_technical_indicators(self, data):
        # Price features
        data['Price_Ratio'] = data['Close'] / data['Open']
        data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
        data['Price_Change'] = data['Close'].pct_change()
        
        # Moving averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data.fillna(method='bfill')
    
    def prepare_features(self, data):
        data = self.add_technical_indicators(data.copy())
        feature_columns = ['Open', 'High', 'Low', 'Volume', 'Price_Ratio', 'HL_Ratio', 'Price_Change', 'SMA_5', 'SMA_20', 'RSI']
        available_features = [col for col in feature_columns if col in data.columns]
        return data[available_features], available_features
    
    def train(self, data):
        features, feature_names = self.prepare_features(data)
        target = data['Close'].values[self.lookback:]
        X = features.values[self.lookback:]
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = target[:split_idx], target[split_idx:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.rf_model.fit(X_train_scaled, y_train)
        self.gb_model.fit(X_train_scaled, y_train)
        
        return {"status": "trained"}
    
    def predict(self, data, future_days=30):
        features, _ = self.prepare_features(data)
        predictions = []
        current_data = data.copy()
        
        for _ in range(future_days):
            current_features, _ = self.prepare_features(current_data)
            latest_features = current_features.iloc[-1:].values
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
        
        # RSI signals
        rsi = self.calculate_rsi(data['Close'])
        rsi_signal = "NEUTRAL"
        if rsi.iloc[-1] > 70:
            rsi_signal = "OVERSOLD (SELL)"
        elif rsi.iloc[-1] < 30:
            rsi_signal = "OVERBOUGHT (BUY)"
        
        # MACD signals
        macd, signal = self.calculate_macd(data['Close'])
        macd_trend = "NEUTRAL"
        if macd.iloc[-1] > signal.iloc[-1]:
            macd_trend = "BULLISH (BUY)"
        else:
            macd_trend = "BEARISH (SELL)"
        
        # Moving Average
        ma_20 = data['Close'].rolling(window=20).mean()
        ma_50 = data['Close'].rolling(window=50).mean()
        ma_signal = "NEUTRAL"
        if ma_20.iloc[-1] > ma_50.iloc[-1]:
            ma_signal = "BULLISH (BUY)"
        else:
            ma_signal = "BEARISH (SELL)"
        
        signals.extend([
            {"Indicator": "RSI", "Value": f"{rsi.iloc[-1]:.2f}", "Signal": rsi_signal},
            {"Indicator": "MACD", "Value": f"{macd.iloc[-1]:.2f}", "Signal": macd_trend},
            {"Indicator": "Moving Averages", "Value": f"MA20: {ma_20.iloc[-1]:.2f}", "Signal": ma_signal}
        ])
        
        return signals

def main():
    app = StockForecastApp()
    lstm_model = LSTMForecaster()
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
        use_lstm = st.checkbox("LSTM Neural Network", value=True)
        use_prophet = st.checkbox("Facebook Prophet", value=True)
        use_ensemble = st.checkbox("Ensemble Model", value=True)
        
        selected_models = []
        if use_lstm:
            selected_models.append("LSTM")
        if use_prophet:
            selected_models.append("Prophet")
        if use_ensemble:
            selected_models.append("Ensemble")
        
        # Load data button
        if st.button("🚀 Load Data & Generate Forecast", use_container_width=True):
            with st.spinner("🔄 Fetching stock data and training AI models..."):
                data = app.get_stock_data(selected_stock, period)
                if data is not None:
                    st.session_state.data = data
                    st.session_state.selected_stock = selected_stock
                    st.session_state.forecast_days = forecast_days
                    st.session_state.selected_models = selected_models
                    st.session_state.predictions = {}
                    
                    # Train models and get predictions
                    for model_name in selected_models:
                        try:
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
                            st.error(f"Error with {model_name}: {str(e)}")
                    
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
        tab1, tab2, tab3 = st.tabs(["🎯 Price Forecast", "📈 Technical Analysis", "🤖 Model Details"])
        
        with tab1:
            # Price chart with forecasts
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
                    
                    predicted_price = predictions[-1]
                    change = predicted_price - current_price
                    change_pct = (change / current_price) * 100
                    
                    st.metric(
                        label=f"Target Price ({forecast_days} days)",
                        value=f"₹{predicted_price:.2f}",
                        delta=f"{change_pct:+.2f}%"
                    )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            # Technical analysis
            st.markdown("### 📈 Technical Indicators")
            
            signals = tech_analyzer.generate_signals(data)
            
            for signal in signals:
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.write(f"**{signal['Indicator']}**")
                with col2:
                    st.write(signal['Value'])
                with col3:
                    if "BUY" in signal['Signal']:
                        st.success(f"🎯 {signal['Signal']}")
                    elif "SELL" in signal['Signal']:
                        st.error(f"🎯 {signal['Signal']}")
                    else:
                        st.info(f"🎯 {signal['Signal']}")
            
            # Technical chart
            st.markdown("### 📊 Technical Chart")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price with Indicators', 'RSI'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Price
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'], name='Close Price', line=dict(color='#00b4d8')
            ), row=1, col=1)
            
            # RSI
            rsi = tech_analyzer.calculate_rsi(data['Close'])
            fig.add_trace(go.Scatter(
                x=data.index, y=rsi, name='RSI', line=dict(color='purple')
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(height=600, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🤖 AI Model Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                #### 🧠 LSTM Neural Network
                - **Type**: Deep Learning
                - **Best for**: Complex patterns
                - **Strengths**: Learns long-term dependencies
                - **Training**: Requires more data
                """)
            
            with col2:
                st.markdown("""
                #### 📊 Facebook Prophet
                - **Type**: Time Series Forecasting
                - **Best for**: Seasonal patterns
                - **Strengths**: Handles holidays
                - **Training**: Fast and robust
                """)
            
            with col3:
                st.markdown("""
                #### 🔄 Ensemble Model
                - **Type**: Machine Learning Ensemble
                - **Best for**: Overall accuracy
                - **Strengths**: Combines multiple models
                - **Training**: Balanced approach
                """)
            
            st.markdown("---")
            st.markdown("""
            ### ⚠️ Important Disclaimer
            **This AI forecasting tool is for educational and research purposes only.**
            
            - 🤖 AI predictions are not financial advice
            - 📈 Stock markets are unpredictable
            - 💰 Never invest based solely on AI predictions
            - 🔍 Always do your own research
            - 🏦 Consult financial advisors for investment decisions
            """)
    
    else:
        # Welcome page
        st.markdown("""
        ## 🎯 Welcome to AI Stock Forecaster!
        
        ### ✨ Features:
        - **🤖 Multiple AI Models** - LSTM, Prophet, and Ensemble
        - **📈 Real-time Indian Stocks** - Major companies included
        - **🎯 Price Forecasting** - 7 to 90 days ahead
        - **📊 Technical Analysis** - RSI, MACD, Moving Averages
        - **💹 Live Market Data** - Updated automatically
        
        ### 🚀 How to Start:
        1. **Select** a stock from the sidebar
        2. **Choose** data period and forecast days  
        3. **Select** AI models to use
        4. **Click** "Load Data & Generate Forecast"
        5. **Explore** different analysis tabs
        
        ### 📊 Supported Stocks:
        """)
        
        # Display stocks in a nice grid
        cols = st.columns(4)
        stocks = list(app.indian_stocks.keys())
        for i, stock in enumerate(stocks):
            with cols[i % 4]:
                st.write(f"✅ {stock}")
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Pro Tips:
        - Use **multiple models** for better accuracy
        - **Compare predictions** across different models
        - Check **technical signals** for trading insights
        - **Monitor regularly** for updated forecasts
        
        *Ready to start? Use the sidebar to configure your analysis!*
        """)

if __name__ == "__main__":
    main()

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from dateutil.parser import parse
import ccxt

# Fetch and preprocess data
def fetch_and_process_data(ticker, start_date, end_date):
    # Parse dates to handle multiple formats
    start_date = parse(start_date).strftime('%Y-%m-%d')
    end_date = parse(end_date).strftime('%Y-%m-%d')
    
    if ticker in ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX', 'SHIB', 'RNDR','STRK','MANTA','METIS',"ARKM","APT","SUI","HBAR",'DOT','MATIC','CAKE','MANA','MINA','SOL','LINK','LTC','1INCH','NEAR']:
        exchange = ccxt.phemex()
        ohlcv = exchange.fetch_ohlcv(ticker + '/USDT', timeframe='1d', since=exchange.parse8601(start_date + 'T00:00:00Z'))
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    else:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
    
    
    # Add indicators
    macd = MACD(close=df['Close'], window_slow=90, window_fast=30, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA90'] = df['Close'].rolling(window=90).mean()
    
    ema30 = EMAIndicator(close=df['Close'], window=30)
    ema90 = EMAIndicator(close=df['Close'], window=90)
    df['EMA30'] = ema30.ema_indicator()
    df['EMA90'] = ema90.ema_indicator()
    
    rsi_10 = RSIIndicator(close=df['Close'], window=10)
    df['RSI_10'] = rsi_10.rsi()
    
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    stoch = StochasticOscillator(close=df['Close'], high=df['High'], low=df['Low'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr.average_true_range()
    
    # Calculate RSI Derivative
    df['RSI_Derivative'] = df['RSI_10'].diff()

    # Calculate Fibonacci Retracement Levels
    df['Fibonacci'] = np.nan
    for i in range(len(df)):
        max_price = df['High'].iloc[:i + 1].max()
        min_price = df['Low'].iloc[:i + 1].min()
        diff = max_price - min_price
        df.loc[df.index[i], 'Fib_23.6'] = max_price - (0.236 * diff)
        df.loc[df.index[i], 'Fib_38.2'] = max_price - (0.382 * diff)
        df.loc[df.index[i], 'Fib_50.0'] = max_price - (0.5 * diff)
        df.loc[df.index[i], 'Fib_61.8'] = max_price - (0.618 * diff)
        df.loc[df.index[i], 'Fib_100'] = min_price
    
    # Determine nearest Fibonacci level and lower limit of RSI derivative
    df['RSI_Derivative_Lower_Limit'] = np.nan
    for i in range(len(df)):
        if not np.isnan(df['RSI_Derivative'].iloc[i]):
            current_price = df['Close'].iloc[i]
            levels = [df.loc[df.index[i], 'Fib_23.6'], df.loc[df.index[i], 'Fib_38.2'], 
                      df.loc[df.index[i], 'Fib_50.0'], df.loc[df.index[i], 'Fib_61.8']]
            nearest_retracement = min(levels, key=lambda x: abs(x - current_price))
            if df['RSI_Derivative'].iloc[i] < nearest_retracement:
                df.loc[df.index[i], 'RSI_Derivative_Lower_Limit'] = df['RSI_Derivative'].iloc[i]
    
    return df

def calculate_predictive_entries_and_tps(df, position_size):
    # Calculate predictive entry and TP points
    last_close = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    # Long position
    long_entry = last_close - 0.5 * atr
    long_tp = last_close + 2 * atr
    long_profit = (long_tp - long_entry) / long_entry * position_size
    
    # Short position
    short_entry = last_close + 0.5 * atr
    short_tp = last_close - 2 * atr
    short_profit = (short_entry - short_tp) / short_entry * position_size
    
    return long_entry, long_tp, long_profit, short_entry, short_tp, short_profit
def calculate_fibonacci_triangles(df):
    # Find peaks and valleys
    peaks = df['High'][(df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High'])]
    valleys = df['Low'][(df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low'])]
    
    # Calculate Fibonacci levels for each peak and valley
    triangles = []
    for i, (idx, value) in enumerate(peaks.items()):
        if i > 0:
            prev_valley = valleys[valleys.index < idx].iloc[-1]
            fib_levels = [value - (value - prev_valley) * level for level in [0.236, 0.382, 0.5, 0.618]]
            triangles.append((idx, value, fib_levels))
    
    for i, (idx, value) in enumerate(valleys.items()):
        if i > 0:
            prev_peak = peaks[peaks.index < idx].iloc[-1]
            fib_levels = [value + (prev_peak - value) * level for level in [0.236, 0.382, 0.5, 0.618]]
            triangles.append((idx, value, fib_levels))
    
    return triangles

# Initialize app and layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
server = app.server

# Company and crypto options
company_options = [
    {'label': 'Apple', 'value': 'AAPL'},
    {'label': 'Amazon', 'value': 'AMZN'},
    {'label': 'Google', 'value': 'GOOGL'},
    {'label': 'Microsoft', 'value': 'MSFT'},
    {'label': 'Tesla', 'value': 'TSLA'},
    {'label': 'Nano Dimension', 'value': 'NNDM'},
    {'label': 'Ideanomics', 'value': 'IDEX'},
    {'label': 'Senseonics', 'value': 'SENS'},
    {'label': 'Wrap Technologies', 'value': 'WRAP'},
    {'label': 'Vivani Medical', 'value': 'VANI'},
    {'label': 'BIOLASE', 'value': 'BIOL'},
    {'label': 'Polestar Automotive', 'value': 'PSNY'},
    {'label': 'Bionano Genomics', 'value': 'BNGO'},
    {'label': 'Altamira Therapeutics', 'value': 'CYTO'},
    {'label': 'AgEagle Aerial Systems', 'value': 'UAVS'},
    {'label': '22nd Century Group', 'value': 'XXII'},
    {'label': 'Atlantica Sustainable', 'value': 'AY'},
    {'label': 'Energy Recovery', 'value': 'ERII'},
    {'label': 'Pulmatrix', 'value': 'PULM'},
    {'label': 'Nektar Therapeutics', 'value': 'NKTR'},
    {'label': 'FSD Pharma', 'value': 'HUGE'},
    {'label': 'LogicBio Therapeutics', 'value': 'LOGC'},
    {'label': 'Evolus', 'value': 'EOLS'},
    {'label': 'Golden Ocean Group', 'value': 'GOGL'},
    {'label': 'Tonix Pharmaceuticals', 'value': 'TNXP'},
    {'label': 'Sunnova Energy', 'value': 'NOVA'},
    {'label': 'SunPower', 'value': 'SPWR'},
    {'label': 'Herbalife', 'value': 'HLF'},
    {'label': 'Globalstar', 'value': 'GSAT'},
    {'label': 'Bitcoin', 'value': 'BTC'},
    {'label': 'Ethereum', 'value': 'ETH'},
    {'label': 'Solana', 'value': 'SOL'},
    {'label': 'Dogecoin', 'value': 'DOGE'},
    {'label': 'Avalanche', 'value': 'AVAX'},
    {'label': 'Shiba Inu', 'value': 'SHIB'},
    {'label': 'Render Token', 'value': 'RNDR'},
    {'label': 'Stark', 'value': 'STRK'},
    {'label': 'Mana', 'value': 'MANA'},
    {'label': 'Manta', 'value': 'MANTA'},
    {'label': 'Metis', 'value': 'METIS'},
    {'label': "Arkam", 'value': 'ARKM'},
    {'label': 'Mina', 'value': 'MINA'},
    {'label': "Aptos", 'value': "APT"},
    {'label': "Sui", 'value': "SUI"},
    {'label': "HBAR", 'value': "HBAR"},
    {'label': 'PolkaDot', 'value': 'DOT'},
    {'label': 'Polygon', 'value': 'MATIC'},
    {'label': 'Pancake Swap', 'value': 'CAKE'},
    {'label': 'CHAINLINK', 'value': 'LINK'},
    {'label': 'Lightcoin', 'value': 'LTC'},
    {'label': '1INCH', 'value': '1INCH'},
    {'label': 'Near Protocol', 'value': 'NEAR'},
    {'label': 'Solana', 'value': 'SOL'},

]
# Define interval options
interval_options = [
    {'label': '1 Minute', 'value': '1m'},
    {'label': '15 Minutes', 'value': '15m'},
    {'label': '30 Minutes', 'value': '30m'},
    {'label': '1 Hour', 'value': '1h'},
    {'label': '2 Hours', 'value': '2h'},
    {'label': '1 Day', 'value': '1d'}
]



# Update the layout to include input for theoretical position size
app.layout = html.Div([
    html.H1('Pantheum Analysis Dashboard', style={'text-align': 'center', 'color': 'white'}),
    
    html.Div([
        dbc.Checklist(
            id='indicator-toggles',
            options=[
                {'label': 'MACD', 'value': 'MACD'},
                {'label': 'Moving Averages (30, 90)', 'value': 'MAs'},
                {'label': 'Exponential MAs (30, 90)', 'value': 'EMAs'},
                {'label': 'RSI (10)', 'value': 'RSI_10'},
                {'label': 'Bollinger Bands', 'value': 'BB'},
                {'label': 'Stochastic Oscillator', 'value': 'Stoch'},
                {'label': 'ATR', 'value': 'ATR'},
                {'label': 'RSI Derivative', 'value': 'RSI_Derivative'},
                {'label': 'Fibonacci Retracement', 'value': 'Fib'},
                {'label': 'RSI Derivative Lower Limit', 'value': 'RSI_Derivative_Lower_Limit'},
                {'label': 'Volume', 'value': 'Volume'},
                {'label': 'Fibonacci Triangles', 'value': 'Fib_Triangles'},
            ],
            value=['MACD', 'MAs', 'Volume', 'Fib_Triangles'],
            inline=True,
            switch=True,
            style={'color': 'white'}
        ),
        dcc.Dropdown(
            id='company-select',
            options=company_options,
            value='AAPL',
            style={'width': '50%', 'margin': '20px auto'}
        ),
        dcc.Dropdown(
            id='interval-select',
            options=interval_options,
            value='1d',
            style={'width': '50%', 'margin': '20px auto'}
        )
    ], style={'text-align': 'center'}),
    
        dcc.Input(
            id='position-size-input',
            type='number',
            placeholder='Enter theoretical position size',
            value=1000,
            style={'width': '50%', 'margin': '20px auto'}
        ),
        
    html.Div(id='predictive-entry-tp', style={'text-align': 'center', 'margin': '20px', 'color': 'white'}),
    dcc.Graph(id='price-graph'),
    
    html.Div(id='recent-high-ticker', style={'text-align': 'center', 'margin': '20px', 'color': 'white'}),
    html.Div(id='recent-low-ticker', style={'text-align': 'center', 'margin': '20px', 'color': 'white'}),
    html.Div(id='percentage-change', style={'text-align': 'center', 'margin': '20px', 'color': 'white'}),
])

# Update graph callback
@app.callback(
    [Output('price-graph', 'figure'),
     Output('recent-high-ticker', 'children'),
     Output('recent-low-ticker', 'children'),
     Output('percentage-change', 'children'),
     Output('predictive-entry-tp', 'children')],
    [Input('company-select', 'value'),
     Input('indicator-toggles', 'value'),
     Input('interval-select', 'value'),
     Input('position-size-input', 'value')]
)
def update_graph(company, selected_indicators, interval, position):
    # Adjust the date range based on the selected interval
    end_date = '2024-08-07'
    if interval == '1m':
        start_date = '2024-08-06'
    elif interval == '15m' or interval == '30m':
        start_date = '2024-08-01'
    elif interval == '1h' or interval == '2h':
        start_date = '2024-07-01'
    else:  # 1d
        start_date = '2024-01-01'

    df = fetch_and_process_data(company, start_date, end_date)
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
    
    if 'MACD' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD Signal'))
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist'))
    
    if 'MAs' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA30'], mode='lines', name='MA30'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA90'], mode='lines', name='MA90'))
    
    if 'EMAs' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA30'], mode='lines', name='EMA30'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA90'], mode='lines', name='EMA90'))
    
    if 'RSI_10' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_10'], mode='lines', name='RSI_10'))
    
    if 'BB' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], mode='lines', name='BB High'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], mode='lines', name='BB Low'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], mode='lines', name='BB Mid'))
    
    if 'Stoch' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], mode='lines', name='Stoch K'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], mode='lines', name='Stoch D'))
    
    if 'ATR' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], mode='lines', name='ATR'))
    
    if 'RSI_Derivative' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_Derivative'], mode='lines', name='RSI Derivative'))
    
    if 'Fib' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['Fib_23.6'], mode='lines', name='Fib 23.6'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Fib_38.2'], mode='lines', name='Fib 38.2'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Fib_50.0'], mode='lines', name='Fib 50.0'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Fib_61.8'], mode='lines', name='Fib 61.8'))
    
    if 'RSI_Derivative_Lower_Limit' in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_Derivative_Lower_Limit'], mode='lines', name='RSI Derivative Lower Limit'))

    if 'Volume' in selected_indicators:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='blue', opacity=0.4, yaxis='y2'))

    if 'Fib_Triangles' in selected_indicators:
        triangles = calculate_fibonacci_triangles(df)
        for idx, value, fib_levels in triangles:
            for level in fib_levels:
                fig.add_trace(go.Scatter(x=[idx, idx], y=[value, level], mode='lines', line=dict(color='rgba(255,165,0,0.5)'), showlegend=False))

        # Calculate predictive entries and TPs
    position_size = .005
    long_entry, long_tp, long_profit, short_entry, short_tp, short_profit = calculate_predictive_entries_and_tps(df, position_size)

    # Add predictive entry and TP lines to the graph
    fig.add_hline(y=long_entry, line_dash="dash", line_color="green", annotation_text="Long Entry")
    fig.add_hline(y=long_tp, line_dash="dash", line_color="green", annotation_text="Long TP")
    fig.add_hline(y=short_entry, line_dash="dash", line_color="red", annotation_text="Short Entry")
    fig.add_hline(y=short_tp, line_dash="dash", line_color="red", annotation_text="Short TP")

    fig.update_layout(
        title=f'{company} Price Analysis',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, max(df['Volume']) * 5]
        )
    )

    recent_high = df['Close'].max()
    recent_low = df['Close'].min()
    percentage_change = ((recent_high - recent_low) / recent_low) * 100
    
    predictive_entry_tp_text = (
        f"Long Entry: {long_entry:.2f}, Long TP: {long_tp:.2f}, Potential Profit: ${long_profit:.2f}<br>"
        f"Short Entry: {short_entry:.2f}, Short TP: {short_tp:.2f}, Potential Profit: ${short_profit:.2f}"
    )

    return fig, f"Recent High: {recent_high}", f"Recent Low: {recent_low}", f"Percentage Change: {percentage_change:.2f}%", predictive_entry_tp_text



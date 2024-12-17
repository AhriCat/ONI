import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ccxt
import time

class TradingBotRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TradingBotRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

class TradingBot:
    def __init__(self, api_key, secret, hidden_size=32, learning_rate=0.001):
        # Exchange setup
        self.exchange = ccxt.phemex({
            'apiKey': api_key,
            'secret': secret,
        })
        self.hidden_size = hidden_size
        self.model = TradingBotRNN(input_size=5, hidden_size=hidden_size, output_size=1)  # 5 input features
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_market_data(self, symbol, limit=100):
        # Fetch OHLCV data
        bars = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=limit)
        data = np.array(bars)
        return data

    def select_coin(self, coin):
        if coin == 'BTC':
            return 'uBTCUSD'
        else:
            return f'{coin}USD'

    def train_model(self, symbol, epochs=10):
        for epoch in range(epochs):
            data = self.get_market_data(symbol)
            x_train = torch.tensor(data[:, 1:6], dtype=torch.float32).unsqueeze(0)  # Ohlcv features (open, high, low, close, volume)
            y_train = torch.tensor(data[:, 4], dtype=torch.float32).unsqueeze(0)  # Close price as target

            hidden = self.model.init_hidden(x_train.size(0))
            self.optimizer.zero_grad()
            output, _ = self.model(x_train, hidden)
            loss = self.criterion(output, y_train[:, -1].unsqueeze(1))
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def make_trade_decision(self, symbol):
        data = self.get_market_data(symbol, limit=20)  # Last 20 data points
        x_input = torch.tensor(data[:, 1:6], dtype=torch.float32).unsqueeze(0)
        hidden = self.model.init_hidden(x_input.size(0))
        output, _ = self.model(x_input, hidden)
        decision = output.item()

        if decision > 0:
            self.place_order(symbol, 'buy', 1)
        else:
            self.place_order(symbol, 'sell', 1)

    def place_order(self, symbol, side, amount):
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            print(f"Order placed: {side} {amount} {symbol}")
        except Exception as e:
            print(f"Error placing order: {e}")

    def run(self, coin, interval=60):
        symbol = self.select_coin(coin)
        while True:
            self.make_trade_decision(symbol)
            time.sleep(interval)


trader = TradingBot

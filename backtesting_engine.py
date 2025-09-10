import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trading_tools import (
    initial_capital,
    TREND_FILTER_THRESHOLD,
    LOOK_FORWARD,
    TRESHOLD_PCT,
    STOP_LOSS_ATR,
    RISK_TO_REWARD,
    CONFIDENCE_THRESHOLD
)




class StrategyBacktester:
    """
    Класс для проведения бэктеста с точным расчетом комиссий. (Версия 2.3)
    """

    def __init__(self, model, price_data, features_data, initial_capital=initial_capital, fee=0.001):
        self.model = model
        self.price_data = price_data.loc[features_data.index].copy()
        self.features_data = features_data
        self.initial_capital = initial_capital
        self.fee = fee
        self.trades = []

    def run(self, confidence_threshold=CONFIDENCE_THRESHOLD, risk_to_reward=RISK_TO_REWARD, stop_loss_atr_multiplier=STOP_LOSS_ATR,
            trend_filter_threshold=TREND_FILTER_THRESHOLD):
        print(f"\n--- Запуск бэктеста (v2.3) с параметрами ---")
        print(
            f"Порог уверенности: {confidence_threshold * 100:.0f}% | R:R: 1:{risk_to_reward} | SL: {stop_loss_atr_multiplier}*ATR")
        print(f"Структурный фильтр тренда: Активен (порог = {trend_filter_threshold})")

        probabilities = self.model.predict_proba(self.features_data)

        equity = self.initial_capital
        equity_curve = pd.Series(index=self.features_data.index, dtype=float)

        position = 0
        trade_details = {}

        for i in range(len(self.features_data)):
            current_time = self.features_data.index[i]
            current_candle = self.price_data.iloc[i]

            # --- 1. ПРОВЕРКА ВЫХОДА ---
            if position != 0:
                exit_price = 0
                if position == 1:  # Long
                    if current_candle['low'] <= trade_details['sl']:
                        exit_price = trade_details['sl']
                    elif current_candle['high'] >= trade_details['tp']:
                        exit_price = trade_details['tp']
                elif position == -1:  # Short
                    if current_candle['high'] >= trade_details['sl']:
                        exit_price = trade_details['sl']
                    elif current_candle['low'] <= trade_details['tp']:
                        exit_price = trade_details['tp']

                if exit_price > 0:
                    # --- ИЗМЕНЕНИЕ ЗДЕСЬ: БОЛЕЕ ТОЧНЫЙ РАСЧЕТ PNL ---
                    gross_ratio = (exit_price / trade_details['entry_price']) if position == 1 else (
                                trade_details['entry_price'] / exit_price)
                    net_ratio = gross_ratio * (1 - self.fee) * (1 - self.fee)  # Применяем комиссию на вход и выход
                    equity *= net_ratio

                    trade_details.update(
                        {'exit_time': current_time, 'exit_price': exit_price, 'pnl_pct': net_ratio - 1})
                    self.trades.append(trade_details)
                    position, trade_details = 0, {}

            # --- 2. ПРОВЕРКА ВХОДА ---
            if position == 0:
                prob_down, prob_up = probabilities[i]
                atr_value = current_candle.get('atr')

                if pd.notna(atr_value) and atr_value > 0:
                    entry_signal = 0

                    # ---------------< ЛОГИКА ФИЛЬТРА >----------------
                    current_trend_strength = self.features_data['trend_strength_4h'].iloc[i]
                    # Сигнал на ЛОНГ, только если тренд бычий
                    if prob_up >= confidence_threshold and current_trend_strength > trend_filter_threshold:
                        entry_signal = 1
                    # Сигнал на ШОРТ, только если тренд медвежий
                    elif prob_down >= confidence_threshold and current_trend_strength < -trend_filter_threshold:
                        entry_signal = -1
                    # ---------------</ ЛОГИКА ФИЛЬТРА >----------------

                    if entry_signal != 0:
                        position = entry_signal
                        entry_price = current_candle['close']
                        risk_amount = atr_value * stop_loss_atr_multiplier
                        sl = entry_price - risk_amount if position == 1 else entry_price + risk_amount
                        tp = entry_price + (risk_amount * risk_to_reward) if position == 1 else entry_price - (
                                    risk_amount * risk_to_reward)
                        trade_details = {'entry_time': current_time, 'direction': 'LONG' if position == 1 else 'SHORT',
                                         'entry_price': entry_price, 'sl': sl, 'tp': tp}

            equity_curve.iloc[i] = equity

        self.equity_curve = equity_curve.ffill().bfill()
        return self.calculate_metrics()

    def calculate_metrics(self):
        # ... (этот метод остается без изменений) ...
        if not self.trades: return {}
        trades_df = pd.DataFrame(self.trades)
        total_trades = len(trades_df)
        wins = trades_df[trades_df['pnl_pct'] > 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        total_return = (self.equity_curve.iloc[-1] / self.initial_capital) - 1
        total_gain = wins['pnl_pct'].apply(lambda p: max(0, p)).sum()
        total_loss = abs(trades_df['pnl_pct'].apply(lambda p: min(0, p)).sum())
        profit_factor = total_gain / total_loss if total_loss > 0 else np.inf
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        return {
            "Итоговый капитал": f"${self.equity_curve.iloc[-1]:,.2f}",
            "Общая прибыль (%)": f"{total_return * 100:.2f}%",
            "Всего сделок": total_trades,
            "Win Rate (%)": f"{win_rate * 100:.2f}%",
            "Профит-фактор": f"{profit_factor:.2f}",
            "Макс. просадка (%)": f"{max_drawdown * 100:.2f}%"
        }

    def plot_equity_curve(self):
        # ... (этот метод остается без изменений) ...
        if not hasattr(self, 'equity_curve') or self.equity_curve.empty: return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.equity_curve.index, y=self.equity_curve.values, mode='lines', name='Equity'))
        fig.update_layout(title='Кривая Капитала Стратегии', xaxis_title='Дата', yaxis_title='Капитал ($)',
                          template='plotly_dark')
        fig.show()
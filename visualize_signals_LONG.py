# ==============================================================================
# visualize_signals_LONG.py
# ------------------------------------------------------------------------------
# ЗАДАЧА: Загрузить обученную LONG модель-"разведчика", найти последние 15
# сигналов на тестовых данных и отобразить их на интерактивном графике.
# ==============================================================================

import  json
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
import plotly


# Убедитесь, что все эти классы и переменные доступны для импорта
from trading_tools import (
    DataHandler,
    LiquidityMLModel,
    FeatureEngine_LONG,
    TICKER,
    START_DATE,
    safe_ticker,
    DOWNLOAD_DATA,
    fetch_fear_and_greed_index,
    calculate_poc_from_ltf,
    prepare_master_dataframe
)

# --- 1. НАСТРОЙКИ ---
MODEL_TYPE = 'LONG'
N_SIGNALS_TO_SHOW = 20


# --- ✅ НАЧАЛО: ЗАГРУЗКА ПАРАМЕТРОВ ---
params_filename = f"params_{MODEL_TYPE}_{safe_ticker}.json"
try:
    with open(params_filename, 'r') as f:
        saved_params = json.load(f)
    best_prediction_threshold = saved_params['prediction_threshold']
    print(f"✅ Загружен порог из файла: {best_prediction_threshold:.2f}")
except FileNotFoundError:
    print(f"❌ Файл {params_filename} не найден. Используется порог по умолчанию 0.5")
    best_prediction_threshold = 0.5
# --- ✅ КОНЕЦ БЛОКА ---


# --- НОВАЯ ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ГРАФИКА ---
def create_and_save_plot(chart_data_5m, last_signals, model_type, ticker, safe_ticker):
    """Создает и сохраняет интерактивный график Plotly."""
    print("\n--- ЭТАП 4: СОЗДАНИЕ 5M ГРАФИКА ---")
    fig = go.Figure()

    # 1. Добавляем 5-минутные свечи (без изменений)
    fig.add_trace(go.Candlestick(
        x=chart_data_5m.index,
        open=chart_data_5m['open'],
        high=chart_data_5m['high'],
        low=chart_data_5m['low'],
        close=chart_data_5m['close'],
        name=f'Цена {safe_ticker} (5m)',
        increasing_line_color='rgba(76, 175, 80, 0.8)',
        decreasing_line_color='rgba(242, 54, 69, 0.8)',
        increasing_fillcolor='rgba(76, 175, 80, 0.4)',
        decreasing_fillcolor='rgba(242, 54, 69, 0.4)',
        increasing_line_width=0.5,
        decreasing_line_width=0.5
    ))

    # --- ✅ НАЧАЛО ИЗМЕНЕНИЙ: НОВАЯ ЛОГИКА ОТРИСОВКИ СИГНАЛОВ ---

    # Готовим данные для маркеров
    marker_x = []
    marker_y = []
    marker_text = []

    # 2. Вместо красных областей, будем добавлять маркеры
    for signal_time, signal_row in last_signals.iterrows():
        # Координата X: Точное время сигнальной 1H свечи
        marker_x.append(signal_time)

        # Координата Y: Чуть выше максимума этой 1H свечи
        # Мы берем 'atr' из last_signals, чтобы отступ был адаптивным
        y_position = signal_row['low'] - signal_row['atr'] * 0.2
        marker_y.append(y_position)

        # Текст для всплывающей подсказки
        marker_text.append(f"Сигнал в {signal_time.hour}:00")

    # 3. Добавляем ОДИН слой со всеми маркерами
    fig.add_trace(go.Scatter(
        x=marker_x,
        y=marker_y,
        mode='markers',
        marker_symbol='triangle-up',
        marker_color='crimson',
        marker_size=10,
        hoverinfo='text',
        text=marker_text,
        name='Сигналы (1H)'
    ))
    # --- ✅ КОНЕЦ ИЗМЕНЕНИЙ ---

    # 4. Настраиваем внешний вид
    fig.update_layout(
        title=f'Визуализация 1H сигналов ({model_type}) на 5M графике для {ticker}',
        xaxis_title='Дата',
        yaxis_title='Цена',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        showlegend=False
    )

    # 5. Сохраняем в файл
    output_filename = f'signals_chart_5m_{model_type}.html'
    fig.write_html(output_filename)
    print(f"\n✅ График успешно сохранен в файл: {output_filename}")


# --- 2. ОСНОВНОЙ БЛОК ---
if __name__ == "__main__":
    load_dotenv()

    # --- ✅ ЭТАП 1: ЗАГРУЗКА ДАННЫХ ЧЕРЕЗ НОВУЮ ФУНКЦИЮ ---
    # Один вызов заменяет весь старый блок загрузки
    final_df, df_5m = prepare_master_dataframe(START_DATE, TICKER, DOWNLOAD_DATA)
    # --- ✅ КОНЕЦ ИЗМЕНЕНИЯ ---

    # Генерируем признаки из 1H данных
    feature_engine = FeatureEngine_LONG(final_df)
    X, y, price_data_for_backtest = feature_engine.run()

    # --- ЭТАП 2: ЗАГРУЗКА МОДЕЛИ И ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ (на 1H) ---
    print("\n--- ЭТАП 2: ПОЛУЧЕНИЕ 1H ПРЕДСКАЗАНИЙ ---")
    scout_model = LiquidityMLModel()
    model_filename = f"scout_model_{MODEL_TYPE}_{safe_ticker}.pkl"
    scout_model.load_model(model_filename)

    probabilities = scout_model.predict_proba(X)[:, 1]
    predictions = (probabilities >= best_prediction_threshold).astype(int)
    price_data_for_backtest['signal'] = predictions

    # --- ЭТАП 3: ФИЛЬТРАЦИЯ И ПОДГОТОВКА ДАННЫХ ДЛЯ ГРАФИКА ---
    print(f"\n--- ЭТАП 3: ПОИСК ПОСЛЕДНИХ {N_SIGNALS_TO_SHOW} СИГНАЛОВ ---")
    signal_candles = price_data_for_backtest[price_data_for_backtest['signal'] == 1]
    last_signals = signal_candles.tail(N_SIGNALS_TO_SHOW)

    if last_signals.empty:
        print("❌ Не найдено ни одного сигнала для отображения.")
    else:
        # Определяем диапазон для 5M графика
        chart_start_date = last_signals.index[0] - pd.Timedelta(hours=3)
        chart_end_date = last_signals.index[-1] + pd.Timedelta(hours=3)
        chart_data_5m = df_5m[(df_5m.index >= chart_start_date) & (df_5m.index <= chart_end_date)]

        # Вызываем функцию для создания графика
        create_and_save_plot(chart_data_5m, last_signals, MODEL_TYPE, TICKER, safe_ticker)


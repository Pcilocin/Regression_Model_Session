# ==============================================================================
# visualize_signals_SHORT.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ------------------------------------------------------------------------------
# ЗАДАЧА: Загрузить обученную SHORT модель, найти последние сигналы
# и отобразить их на 5M графике с корректной разметкой сессий.
# ==============================================================================

import json
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

# Убедитесь, что все эти классы и переменные доступны для импорта
from trading_tools import (
    LiquidityMLModel,
    FeatureEngineSMC,
    TICKER,
    START_DATE,
    safe_ticker,
    DOWNLOAD_DATA,
    prepare_master_dataframe  # <-- Мы используем обновленную функцию
)

# --- 1. НАСТРОЙКИ ---
MODEL_TYPE = 'SHORT'
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
def create_and_save_plot(chart_data_30m, last_signals_1h, model_type, ticker):
    """Создает и сохраняет интерактивный график Plotly."""
    print("\n--- ЭТАП 4: СОЗДАНИЕ 5M ГРАФИКА ---")
    fig = go.Figure()

    # 1. Добавляем 5-минутные свечи
    fig.add_trace(go.Candlestick(
        x=chart_data_30m.index,
        open=chart_data_30m['open'],
        high=chart_data_30m['high'],
        low=chart_data_30m['low'],
        close=chart_data_30m['close'],
        name=f'Цена {ticker} (30m)',
        increasing_line_color = 'rgba(76, 175, 80, 0.8)',
        decreasing_line_color = 'rgba(242, 54, 69, 0.8)',
        increasing_fillcolor = 'rgba(76, 175, 80, 0.4)',
        decreasing_fillcolor = 'rgba(242, 54, 69, 0.4)',
        increasing_line_width = 0.2,
        decreasing_line_width = 0.2
    ))

    # --- РАЗМЕТКА СЕССИЙ ---
    print("Нанесение разметки торговых сессий...")
    shapes_to_add = []

    # ИСПРАВЛЕНО: Группируем chart_data_30m (который теперь содержит нужные колонки)
    for date, day_data in chart_data_30m.groupby(chart_data_30m.index.date):

        # # Находим Азиатскую киллзону (LKZ)
        # asian_kz_data = day_data[day_data['is_asian_killzone'] == 1]
        # if not asian_kz_data.empty:
        #     x0_akz = asian_kz_data.index[0]
        #     x1_akz = asian_kz_data.index[-1] + pd.Timedelta(minutes=30)
        #     y0_akz = asian_kz_data['low'].min()
        #     y1_akz = asian_kz_data['high'].max()
        #     shapes_to_add.append(go.layout.Shape(
        #         type="rect", xref="x", yref="y", x0=x0_akz, y0=y0_akz, x1=x1_akz, y1=y1_akz,
        #         fillcolor="rgba(223, 115, 255, 0.2)", line=dict(width=0), layer='below'
        #     ))

        # Находим Азиатскую Сессию (AS)
        asian_session_data = day_data[day_data['is_asian_session'] == 1]
        if not asian_session_data.empty:
            x0_as = asian_session_data.index[0]
            x1_as = asian_session_data.index[-1] + pd.Timedelta(minutes=30)
            y0_as = asian_session_data['low'].min()
            y1_as = asian_session_data['high'].max()
            shapes_to_add.append(go.layout.Shape(
                type="rect", xref="x", yref="y", x0=x0_as, y0=y0_as, x1=x1_as, y1=y1_as,
                fillcolor="rgba(174, 129, 255, 0.1)", line=dict(width=0), layer='below'
            ))


        # Находим Лондонскую киллзону (LKZ)
        london_kz_data = day_data[day_data['is_london_killzone'] == 1]
        if not london_kz_data.empty:
            x0_lkz = london_kz_data.index[0]
            x1_lkz = london_kz_data.index[-1] + pd.Timedelta(minutes=30)
            y0_lkz = london_kz_data['low'].min()
            y1_lkz = london_kz_data['high'].max()
            shapes_to_add.append(go.layout.Shape(
                type="rect", xref="x", yref="y", x0=x0_lkz, y0=y0_lkz, x1=x1_lkz, y1=y1_lkz,
                fillcolor="rgba(135, 206, 250, 0.2)", line=dict(width=0), layer='below'
            ))

        # Находим Лондонскую Сессию (LS)
        london_session_data = day_data[day_data['is_london_session'] == 1]
        if not london_session_data.empty:
            x0_ls = london_session_data.index[0]
            x1_ls = london_session_data.index[-1] + pd.Timedelta(minutes=30)
            y0_ls = london_session_data['low'].min()
            y1_ls = london_session_data['high'].max()
            shapes_to_add.append(go.layout.Shape(
                type="rect", xref="x", yref="y", x0=x0_ls, y0=y0_ls, x1=x1_ls, y1=y1_ls,
                fillcolor="rgba(102, 217, 239, 0.1)", line=dict(width=0), layer='below'
            ))


        # # Находим Нью-Йоркскую киллзону (NKZ)
        # newyork_session_data = day_data[day_data['is_newyork_killzone'] == 1]
        # if not newyork_session_data.empty:
        #     x0_nkz = newyork_session_data.index[0]
        #     x1_nkz = newyork_session_data.index[-1] + pd.Timedelta(minutes=30)
        #     y0_nkz = newyork_session_data['low'].min()
        #     y1_nkz = newyork_session_data['high'].max()
        #     shapes_to_add.append(go.layout.Shape(
        #         type="rect", xref="x", yref="y", x0=x0_nkz, y0=y0_nkz, x1=x1_nkz, y1=y1_nkz,
        #         fillcolor="rgba(255, 185, 97, 0.2)", line=dict(width=0), layer='below'
        #     ))
        #
        # # Находим Нью-Йоркскую Сессию (NS)
        # newyork_session_data = day_data[day_data['is_newyork_session'] == 1]
        # if not newyork_session_data.empty:
        #     x0_ns = newyork_session_data.index[0]
        #     x1_ns = newyork_session_data.index[-1] + pd.Timedelta(minutes=30)
        #     y0_ns = newyork_session_data['low'].min()
        #     y1_ns = newyork_session_data['high'].max()
        #     shapes_to_add.append(go.layout.Shape(
        #         type="rect", xref="x", yref="y", x0=x0_ns, y0=y0_ns, x1=x1_ns, y1=y1_ns,
        #         fillcolor="rgba(253, 151, 31, 0.1)", line=dict(width=0), layer='below'
        #     ))


    fig.update_layout(shapes=shapes_to_add)

    # --- ОТРИСОВКА СИГНАЛОВ (1H) ---
    marker_x = []
    marker_y = []
    marker_text = []

    for signal_time, signal_row in last_signals_1h.iterrows():
        marker_x.append(signal_time)
        y_position = signal_row['high'] + signal_row['atr'] * 0.2
        marker_y.append(y_position)
        marker_text.append(f"Сигнал в {signal_time.hour}:00")

    fig.add_trace(go.Scatter(
        x=marker_x, y=marker_y, mode='markers', marker_symbol='triangle-down',
        marker_color='crimson', marker_size=6, hoverinfo='text',
        text=marker_text, name='Сигналы (1H)'
    ))

    # --- НАСТРОЙКА ВНЕШНЕГО ВИДА ---
    fig.update_layout(
        title=f'Визуализация 1H сигналов ({model_type}) на 30M графике для {ticker}',
        xaxis_title='Дата', yaxis_title='Цена', template='plotly_dark',
        xaxis_rangeslider_visible=False, showlegend=False
    )

    output_filename = f'signals_chart_30m_{model_type}.html'
    fig.write_html(output_filename)
    print(f"\n✅ График успешно сохранен в файл: {output_filename}")


# --- 2. ОСНОВНОЙ БЛОК ---
if __name__ == "__main__":
    load_dotenv()

    # --- ЭТАП 1: ЗАГРУЗКА ДАННЫХ ЧЕРЕЗ ОБНОВЛЕННУЮ ФУНКЦИЮ ---
    # Один вызов теперь корректно возвращает все, что нам нужно
    df_1h, df_30m, df_15m, df_5m, df_1m = prepare_master_dataframe(START_DATE, TICKER, DOWNLOAD_DATA)

    if df_1h.empty or df_30m.empty:
        print("❌ Не удалось загрузить данные 1H или 30M. Визуализация невозможна.")
        exit()

    # --- ЭТАП 2: ГЕНЕРАЦИЯ ПРИЗНАКОВ И ПОЛУЧЕНИЕ 1H "МАСТЕР-ФРЕЙМА" ---
    # ИСПРАВЛЕНО: Передаем правильные аргументы в FeatureEngineSMC
    feature_engine = FeatureEngineSMC(df_1h, df_1m)
    # Нам не нужен 'y' для визуализации, поэтому create_target=False экономит время.
    X, _, price_data_for_backtest = feature_engine.run(model_type=MODEL_TYPE, create_target=False)

    # --- ЭТАП 3: ЗАГРУЗКА МОДЕЛИ И ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ ---
    print("\n--- ЭТАП 3: ПОЛУЧЕНИЕ 1H ПРЕДСКАЗАНИЙ ---")
    scout_model = LiquidityMLModel()
    model_filename = f"scout_model_{MODEL_TYPE}_{safe_ticker}.pkl"
    scout_model.load_model(model_filename)

    probabilities = scout_model.predict_proba(X)[:, 1]
    predictions = (probabilities >= best_prediction_threshold).astype(int)
    price_data_for_backtest['signal'] = predictions

    # --- ЭТАП 4: ПОДГОТОВКА ДАННЫХ ДЛЯ ГРАФИКА ---
    print(f"\n--- ЭТАП 4: ПОИСК ПОСЛЕДНИХ {N_SIGNALS_TO_SHOW} СИГНАЛОВ ---")
    signal_candles_1h = price_data_for_backtest[price_data_for_backtest['signal'] == 1]
    last_signals_1h = signal_candles_1h.tail(N_SIGNALS_TO_SHOW)

    if last_signals_1h.empty:
        print("❌ Не найдено ни одного сигнала для отображения.")
    else:
        # 1. Определяем диапазон для 5M графика
        chart_start_date = last_signals_1h.index[0] - pd.Timedelta(hours=3)
        chart_end_date = last_signals_1h.index[-1] + pd.Timedelta(hours=3)
        chart_data_30m_slice = df_30m[(df_30m.index >= chart_start_date) & (df_30m.index <= chart_end_date)].copy()

        # 2. ИСПРАВЛЕНО: Обогащаем 30M-данные 1H-контекстом сессий
        context_cols = ['is_asian_session', 'is_asian_killzone', 'is_london_session', 'is_london_killzone',
                        # 'is_newyork_session', 'is_newyork_killzone'
                        ]
        context_data_1h = price_data_for_backtest[context_cols]
        chart_data_30m_enriched = pd.merge_asof(
            left=chart_data_30m_slice,
            right=context_data_1h,
            left_index=True,
            right_index=True,
            direction='backward'
        )
        chart_data_30m_enriched[context_cols] = chart_data_30m_enriched[context_cols].ffill().fillna(0)

        # 3. Вызываем функцию для создания графика
        create_and_save_plot(chart_data_30m_enriched, last_signals_1h, MODEL_TYPE, TICKER)

# Advanced_Trainer.py

import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Импортируем все из одного места
from trading_tools import (
    prepare_master_dataframe,
    FeatureEngineSMC,
    LiquidityMLModel,
    START_DATE,
    TICKER,
    DOWNLOAD_DATA,
    LOOK_FORWARD,
    TRESHOLD_PCT,
    CONFIDENCE_THRESHOLD
)



# --- 1. КОНСТАНТЫ ---
MODEL_TYPE = 'SHORT'
CONFIDENCE_FILTER_THRESHOLD = CONFIDENCE_THRESHOLD  # Порог для первой модели-фильтра


# --- 2. ФУНКЦИЯ РАЗМЕТКИ (остается той же) ---
def find_significant_moves(df, look_forward=LOOK_FORWARD, threshold_pct=TRESHOLD_PCT):
    print(f"Поиск будущих движений > {threshold_pct * 100}% в окне {look_forward} часов...")
    future_highs = df['high'].rolling(window=look_forward, min_periods=1).max().shift(-look_forward)
    future_lows = df['low'].rolling(window=look_forward, min_periods=1).min().shift(-look_forward)
    is_up_move = (future_highs / df['close']) - 1 >= threshold_pct
    is_down_move = 1 - (future_lows / df['close']) >= threshold_pct
    target = pd.Series(0, index=df.index, dtype=int)
    target.loc[is_up_move] = 1;
    target.loc[is_down_move] = -1
    target.loc[is_up_move & is_down_move] = -1
    return target


# --- 3. ОСНОВНОЙ БЛОК ---
if __name__ == "__main__":
    # --- ЭТАП 1: ПОДГОТОВКА ДАННЫХ (как и раньше) ---
    raw_final_df, df_4h, df_30m, df_15m, df_5m, df_1m = prepare_master_dataframe(START_DATE, TICKER, DOWNLOAD_DATA)
    feature_engine = FeatureEngineSMC(
        main_df_1h=raw_final_df, ltf_df_4h=df_4h, ltf_df_30m=df_30m,
        ltf_df_15m=df_15m, ltf_df_5m=df_5m, ltf_df_1m=df_1m
    )
    X, _, _, enriched_df = feature_engine.run(model_type=MODEL_TYPE, create_target=False)
    y = find_significant_moves(enriched_df, look_forward=LOOK_FORWARD, threshold_pct=TRESHOLD_PCT)

    data_for_training = X[y != 0].copy()
    y_binary = y[y != 0].replace({-1: 0})
    X_train, X_test, y_train, y_test = train_test_split(data_for_training, y_binary, test_size=0.2, shuffle=False)

    # --- ЭТАП 2: ОБУЧЕНИЕ МОДЕЛИ №1 (ФИЛЬТР) ---
    print("\n--- ЭТАП 2: ОБУЧЕНИЕ МОДЕЛИ №1 (Простой Фильтр) ---")
    num_neg = (y_train == 0).sum();
    num_pos = (y_train == 1).sum()
    base_scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1
    # Используем простые, "решительные" параметры
    filter_params = {'random_state': 33, 'scale_pos_weight': base_scale_pos_weight, 'learning_rate': 0.05,
                     'n_estimators': 200}
    filter_model = LiquidityMLModel(params=filter_params)
    filter_model.train(X_train, y_train)

    # --- ЭТАП 3: ФИЛЬТРАЦИЯ ДАННЫХ С ПОМОЩЬЮ МОДЕЛИ №1 ---
    print(f"\n--- ЭТАП 3: ФИЛЬТРАЦИЯ ДАННЫХ (порог уверенности > {CONFIDENCE_FILTER_THRESHOLD * 100:.0f}%) ---")
    probabilities = filter_model.predict_proba(X_train)

    high_confidence_indices = [
        X_train.index[i] for i, prob in enumerate(probabilities)
        if prob[0] > CONFIDENCE_FILTER_THRESHOLD or prob[1] > CONFIDENCE_FILTER_THRESHOLD
    ]

    X_train_filtered = X_train.loc[high_confidence_indices]
    y_train_filtered = y_train.loc[high_confidence_indices]

    print(f"Найдено {len(X_train_filtered)} сигналов с высокой уверенностью для обучения Эксперта.")

    if len(X_train_filtered) < 50:
        print("❌ Недостаточно данных для обучения Эксперта. Попробуйте понизить CONFIDENCE_FILTER_THRESHOLD.");
        exit()

    # --- ЭТАП 4: ОБУЧЕНИЕ МОДЕЛИ №2 (ЭКСПЕРТ) ---
    print("\n--- ЭТАП 4: ОБУЧЕНИЕ МОДЕЛИ №2 (Эксперт на отфильтрованных данных) ---")
    safe_ticker = TICKER.replace('/', '_')
    params_filename = f"best_params_{MODEL_TYPE}_{safe_ticker}.json"
    try:
        with open(params_filename, 'r') as f:
            best_params = json.load(f)
        print(f"✅ Лучшие параметры (от Optuna) загружены из {params_filename}")
    except FileNotFoundError:
        print(f"⚠️ Файл с параметрами не найден. Запустите Model_Tuner.py сначала!");
        exit()

    expert_model = LiquidityMLModel(params=best_params)
    # Обучаем Эксперта ТОЛЬКО на лучших, отфильтрованных данных
    expert_model.train(X_train_filtered, y_train_filtered)

    model_filename = f"expert_model_{MODEL_TYPE}_{safe_ticker}.pkl"
    joblib.dump(expert_model, model_filename)
    print(f"✅ Финальная модель 'Эксперт' сохранена: {model_filename}")

    # --- ЭТАП 5: ОЦЕНКА ЭКСПЕРТА НА ТЕСТОВЫХ ДАННЫХ ---
    print("\n--- ЭТАП 5: ОЦЕНКА ФИНАЛЬНОЙ МОДЕЛИ 'ЭКСПЕРТ' ---")
    expert_model.evaluate(X_test, y_test)

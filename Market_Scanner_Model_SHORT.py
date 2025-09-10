# Market_Scanner_Model_Short.py

import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from trading_tools import (
    prepare_master_dataframe,
    FeatureEngineSMC,
    LiquidityMLModel,
    START_DATE,
    TICKER,
    safe_ticker,
    DOWNLOAD_DATA,
    TRESHOLD_PCT,
    LOOK_FORWARD,
)

# Добавьте остальные ваши импорты

# --- 1. КОНСТАНТЫ ---
MODEL_TYPE = 'SHORT'


# --- 2. НОВАЯ ФУНКЦИЯ РАЗМЕТКИ ---
def find_significant_moves(df, look_forward=None, threshold_pct=None):
    """
    Размечает данные для классификации на основе будущих движений.
    y = 1: Ожидается значительный рост (> threshold_pct).
    y = -1: Ожидается значительное падение (> threshold_pct).
    y = 0: Боковик/шум.
    """
    print(f"Поиск будущих движений > {threshold_pct * 100}% в окне {look_forward} часов...")

    future_highs = df['high'].rolling(window=look_forward, min_periods=1).max().shift(-look_forward)
    future_lows = df['low'].rolling(window=look_forward, min_periods=1).min().shift(-look_forward)

    is_up_move = (future_highs / df['close']) - 1 >= threshold_pct
    is_down_move = 1 - (future_lows / df['close']) >= threshold_pct

    target = pd.Series(0, index=df.index)
    target.loc[is_up_move] = 1
    target.loc[is_down_move] = -1
    target.loc[is_up_move & is_down_move] = -1

    print("Разметка завершена.")
    return target


# --- 3. ОСНОВНОЙ БЛОК ---
if __name__ == "__main__":
    # --- ЭТАП 1: ДАННЫЕ И ПРИЗНАКИ ---
    # 1.1. Загружаем все необходимые DataFrame'ы
    raw_final_df, df_4h, df_30m, df_15m, df_5m, df_1m = prepare_master_dataframe(start_date=START_DATE, ticker=TICKER, download_data=DOWNLOAD_DATA)

    # 1.2. Правильно вызываем FeatureEngineSMC, передавая все данные
    feature_engine = FeatureEngineSMC(
        main_df_1h=raw_final_df,
        ltf_df_4h=df_4h,
        ltf_df_30m=df_30m,
        ltf_df_15m=df_15m,
        ltf_df_5m=df_5m,
        ltf_df_1m=df_1m
    )

    # 1.3. Правильно принимаем результаты: X (признаки) и enriched_df (полный DF)
    X, _, _, enriched_df = feature_engine.run(model_type=MODEL_TYPE, create_target=False)

    # --- ЭТАП 2: СОЗДАНИЕ ОБЪЕКТИВНОЙ ЦЕЛИ (y) ---
    # 2.1. Создаем цель на основе ПОЛНОГО DataFrame'а с ценами
    y = find_significant_moves(enriched_df, look_forward=LOOK_FORWARD, threshold_pct=TRESHOLD_PCT)

    print("\nБаланс классов в данных:")
    print(y.value_counts())

    # --- ЭТАП 3: ОБУЧЕНИЕ МОДЕЛИ-СКАНЕРА ---
    # 3.1. Готовим данные для обучения
    data_for_training = X.copy()
    data_for_training['target'] = y

    data_for_training = data_for_training[data_for_training['target'] != 0]

    if len(data_for_training) < 100:
        print("❌ Недостаточно данных о значительных движениях для обучения.")
        exit()

    X_filtered = data_for_training.drop(columns=['target'])
    y_binary = data_for_training['target'].replace({-1: 0})  # Down = 0, Up = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_binary, test_size=0.2, shuffle=False
    )

    # --- ЗАГРУЗКА ЛУЧШИХ ПАРАМЕТРОВ ---
    safe_ticker = TICKER.replace('/', '_')
    params_filename = f"best_params_{MODEL_TYPE}_{safe_ticker}.json"
    try:
        with open(params_filename, 'r') as f:
            best_params = json.load(f)
        print(f"✅ Лучшие параметры загружены из {params_filename}")
    except FileNotFoundError:
        print("⚠️ Файл с параметрами не найден. Используются параметры по умолчанию.")
        best_params = {'random_state': 33} # хотя бы random_state оставим

    # Передаем лучшие параметры в модель
    scanner_model = LiquidityMLModel(params=best_params)
    scanner_model.train(X_train, y_train)
    #
    # # 3.2. Балансируем классы и обучаем модель
    # num_negatives = (y_train == 0).sum()
    # num_positives = (y_train == 1).sum()
    # scale_pos_weight_value = num_negatives / num_positives if num_positives > 0 else 1
    #
    # print(f"\nДисбаланс в обучающей выборке: Down(0) -> {num_negatives}, Up(1) -> {num_positives}")
    # print(f"Рассчитан 'scale_pos_weight': {scale_pos_weight_value:.2f}")
    #
    # scanner_params = {'scale_pos_weight': scale_pos_weight_value, 'random_state': 33}
    # scanner_model = LiquidityMLModel(params=scanner_params)
    # scanner_model.train(X_train, y_train)

    # 3.3. Сохраняем обученную модель
    model_filename = f"scanner_model_{MODEL_TYPE}_{safe_ticker}.pkl"
    joblib.dump(scanner_model, model_filename)
    print(f"✅ Модель успешно сохранена в файл: {model_filename}")

    # --- ЭТАП 4: ОЦЕНКА И АНАЛИЗ ---
    # (Этот блок остается таким же, как в вашем коде, он корректен)
    print("\n--- ЭТАП 4: РАСШИРЕННАЯ ОЦЕНКА И ФИЛЬТРАЦИЯ СИГНАЛОВ ---")
    print("\n--- Базовая оценка (без фильтра) ---")
    scanner_model.evaluate(X_test, y_test)

    print("\n--- Анализ с фильтрацией по уверенности модели (predict_proba) ---")
    probabilities = scanner_model.predict_proba(X_test)
    confidence_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    for threshold in confidence_thresholds:
        print(f"\n--- Результаты при пороге уверенности > {threshold * 100:.0f}% ---")
        filtered_preds = []
        for prob_pair in probabilities:
            if prob_pair[0] >= threshold:
                filtered_preds.append(0)
            elif prob_pair[1] >= threshold:
                filtered_preds.append(1)
            else:
                filtered_preds.append(-1)

        results_df = pd.DataFrame({'y_true': y_test, 'prediction': filtered_preds})
        actionable_results = results_df[results_df['prediction'] != -1]

        if len(actionable_results) == 0:
            print("Сделок не найдено при данном пороге.")
            continue

        report = classification_report(actionable_results['y_true'], actionable_results['prediction'], labels=[0, 1])
        print(
            f"Сделок совершено: {len(actionable_results)} из {len(results_df)} ({len(actionable_results) / len(results_df):.1%})")
        print(report)

    print("\nВажность признаков для сканера:")
    print(scanner_model.get_feature_importance().head(15))
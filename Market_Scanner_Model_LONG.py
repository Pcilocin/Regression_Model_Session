import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from trading_tools import (
    prepare_master_dataframe,
    FeatureEngineSMC,
    LiquidityMLModel,
    START_DATE,
    TICKER,
    safe_ticker,
    DOWNLOAD_DATA
)

# Добавьте остальные ваши импорты

# --- 1. КОНСТАНТЫ ---
MODEL_TYPE = 'LONG'



# --- 2. НОВАЯ ФУНКЦИЯ РАЗМЕТКИ ---
def find_significant_moves(df, look_forward=None, threshold_pct=None):
    """
    Размечает данные для мультиклассовой классификации на основе будущих движений.
    y = 1: Ожидается значительный рост (> threshold_pct).
    y = -1: Ожидается значительное падение (> threshold_pct).
    y = 0: Боковик/шум.
    """
    print(f"Поиск будущих движений > {threshold_pct * 100}% в окне {look_forward} часов...")

    future_highs = df['high'].rolling(window=look_forward, min_periods=1).max().shift(-look_forward)
    future_lows = df['low'].rolling(window=look_forward, min_periods=1).min().shift(-look_forward)

    # Условия
    is_up_move = (future_highs / df['close']) - 1 >= threshold_pct
    is_down_move = 1 - (future_lows / df['close']) >= threshold_pct

    target = pd.Series(0, index=df.index)
    target.loc[is_up_move] = 1
    target.loc[is_down_move] = -1

    # Если и вверх и вниз (волатильность), отдаем приоритет падению (более рискованно)
    target.loc[is_up_move & is_down_move] = -1

    print("Разметка завершена.")
    return target


# --- 3. ОСНОВНОЙ БЛОК ---
if __name__ == "__main__":
    # --- ЭТАП 1: ДАННЫЕ И ПРИЗНАКИ ---
    final_df, df_30m, df_15m, df_5m, df_1m = prepare_master_dataframe(START_DATE, TICKER, DOWNLOAD_DATA)

    # ВАЖНО: Мы запускаем Feature Engine БЕЗ создания цели. Он только генерирует признаки (X).
    feature_engine = FeatureEngineSMC(final_df, ltf_df_30m=df_30m, ltf_df_5m=df_5m, ltf_df_1m=df_1m)
    X, _, _ = feature_engine.run(model_type=MODEL_TYPE, create_target=False)

    # --- ЭТАП 2: СОЗДАНИЕ ОБЪЕКТИВНОЙ ЦЕЛИ (y) ---
    y = find_significant_moves(final_df.loc[X.index], look_forward=8, threshold_pct=0.03)

    print("\nБаланс классов в данных:")
    print(y.value_counts())

    # --- ЭТАП 3: ОБУЧЕНИЕ МОДЕЛИ-СКАНЕРА ---
    # Мы будем предсказывать только UP (1) или DOWN (-1), поэтому отфильтруем боковик (0)
    data_for_training = X.copy()
    data_for_training['target'] = y

    # Удаляем боковик ИЗ ТРЕНИРОВОЧНОГО НАБОРА, чтобы модель сфокусировалась
    data_for_training = data_for_training[data_for_training['target'] != 0]

    if len(data_for_training) < 100:
        print("❌ Недостаточно данных о значительных движениях для обучения.")
        exit()

    X_filtered = data_for_training.drop(columns=['target'])
    y_filtered = data_for_training['target']

    # Меняем метки -1 на 0 для удобства бинарного классификатора (Down=0, Up=1)
    y_binary = y_filtered.replace({-1: 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_binary, test_size=0.2, shuffle=False
    )

    # Балансируем классы
    num_negatives = (y_train == 0).sum()
    num_positives = (y_train == 1).sum()
    scale_pos_weight_value = num_negatives / num_positives if num_positives > 0 else 1

    print(f"\nДисбаланс в обучающей выборке: Down(0) -> {num_negatives}, Up(1) -> {num_positives}")
    print(f"Рассчитан 'scale_pos_weight': {scale_pos_weight_value:.2f}")

    scanner_params = {'scale_pos_weight': scale_pos_weight_value, 'random_state': 33}
    scanner_model = LiquidityMLModel(params=scanner_params)
    scanner_model.train(X_train, y_train)

    # --- ЭТАП 4: РАСШИРЕННАЯ ОЦЕНКА И ФИЛЬТРАЦИЯ СИГНАЛОВ ---
    print("\n--- ЭТАП 4: РАСШИРЕННАЯ ОЦЕНКА И ФИЛЬТРАЦИЯ СИГНАЛОВ ---")

    print("\n--- Базовая оценка (без фильтра) ---")
    scanner_model.evaluate(X_test, y_test)

    print("\n--- Анализ с фильтрацией по уверенности модели (predict_proba) ---")

    # Получаем вероятности для каждого класса [P(Down), P(Up)]
    probabilities = scanner_model.predict_proba(X_test)

    # Задаем пороги уверенности, которые хотим протестировать
    confidence_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    for threshold in confidence_thresholds:
        print(f"\n--- Результаты при пороге уверенности > {threshold * 100:.0f}% ---")

        # Фильтруем предсказания. Делаем предсказание, только если уверены.
        filtered_preds = []
        for prob_pair in probabilities:
            prob_down = prob_pair[0]
            prob_up = prob_pair[1]

            if prob_down >= threshold:
                filtered_preds.append(0)  # Уверены, что будет падение
            elif prob_up >= threshold:
                filtered_preds.append(1)  # Уверены, что будет рост
            else:
                filtered_preds.append(-1)  # НЕ УВЕРЕНЫ - пропускаем сделку

        # Создаем DataFrame для анализа
        results_df = pd.DataFrame({
            'y_true': y_test,
            'prediction': filtered_preds
        })

        # Отфильтровываем пропущенные сделки для расчета метрик
        actionable_results = results_df[results_df['prediction'] != -1]

        if len(actionable_results) == 0:
            print("Сделок не найдено при данном пороге.")
            continue

        # Считаем и выводим отчет
        report = classification_report(
            actionable_results['y_true'],
            actionable_results['prediction'],
            labels=[0, 1]
        )

        total_signals = len(results_df)
        trades_taken = len(actionable_results)

        print(f"Сделок совершено: {trades_taken} из {total_signals} ({trades_taken / total_signals:.1%})")
        print(report)

    print("\nВажность признаков для сканера (включая новые):")
    print(scanner_model.get_feature_importance().head(20))
import joblib
from sklearn.model_selection import train_test_split
from trading_tools import (
    prepare_master_dataframe,
    FeatureEngineSMC,
    START_DATE,
    TICKER,
    safe_ticker,
    DOWNLOAD_DATA
)
from backtesting_engine import StrategyBacktester  # <-- Импортируем наш новый движок
from backtesting_engine import (
    CONFIDENCE_THRESHOLD,
    RISK_TO_REWARD,
    STOP_LOSS_ATR,
    TREND_FILTER_THRESHOLD,
    LOOK_FORWARD,
    TRESHOLD_PCT,
)


# --- 1. КОНСТАНТЫ И НАСТРОЙКИ ---
MODEL_TYPE = 'SHORT'  # Укажите, какую модель тестируем: 'SHORT' или 'LONG'
MODEL_FILENAME = f"scanner_model_{MODEL_TYPE}_{safe_ticker}.pkl"  # Имя файла с обученной моделью


if __name__ == "__main__":
    # --- ЭТАП 1: ЗАГРУЗКА МОДЕЛИ И ДАННЫХ ---
    print(f"--- Загрузка модели {MODEL_FILENAME} и подготовка данных ---")

    try:
        model = joblib.load(MODEL_FILENAME)
    except FileNotFoundError:
        print(f"❌ Ошибка: Модель не найдена по пути '{MODEL_FILENAME}'.")
        print("   Убедитесь, что вы сначала обучили и сохранили модель с помощью Market_Scanner_Model.py.")
        exit()

    final_df, df_4h, df_30m, df_15m, df_5m, df_1m = prepare_master_dataframe(START_DATE, TICKER, DOWNLOAD_DATA)
    feature_engine = FeatureEngineSMC(
            main_df_1h=final_df,
            ltf_df_4h=df_4h,
            ltf_df_30m=df_30m,
            ltf_df_15m=df_15m,
            ltf_df_5m=df_5m,
            ltf_df_1m=df_1m)

    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
    # Теперь мы получаем 4 элемента: X, y_class, y_regr и самое главное - enriched_df
    X, _, _, enriched_df = feature_engine.run(model_type=MODEL_TYPE, create_target=False)

    y_dummy = [0] * len(X)
    X_train, X_test, _, _ = train_test_split(X, y_dummy, test_size=0.2, shuffle=False)

    print(f"Данные для бэктеста готовы. Размер тестовой выборки: {len(X_test)}")

    # --- ЭТАП 2: ЗАПУСК БЭКТЕСТЕРА ---

    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
    # Мы передаем в бэктестер enriched_df, в котором гарантированно есть колонка 'atr'
    backtester = StrategyBacktester(
        model=model,
        price_data=enriched_df,  # <-- Используем обогащенный DataFrame
        features_data=X_test
    )

    # results = backtester.run(
    #     confidence_threshold=CONFIDENCE_THRESHOLD,
    #     risk_to_reward=RISK_TO_REWARD,
    #     stop_loss_atr_multiplier=STOP_LOSS_ATR
    # )

    results = backtester.run(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        risk_to_reward=RISK_TO_REWARD,
        stop_loss_atr_multiplier=STOP_LOSS_ATR,
        trend_structure_threshold=TREND_FILTER_THRESHOLD # <-- ПЕРЕДАЕМ НОВЫЙ ПАРАМЕТР
    )

    # --- ЭТАП 3: ВЫВОД РЕЗУЛЬТАТОВ ---
    print("\n--- РЕЗУЛЬТАТЫ БЭКТЕСТА ---")
    if results:
        for key, value in results.items():
            print(f"{key}: {value}")
        backtester.plot_equity_curve()
    else:
        print("Бэктест не вернул результатов.")
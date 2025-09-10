import pandas as pd
import joblib
import numpy as np
import optuna
import json
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import f1_score
from trading_tools import (
    prepare_master_dataframe,
    FeatureEngineSMC,
    LiquidityMLModel,
    START_DATE,
    TICKER,
    DOWNLOAD_DATA,
    LOOK_FORWARD,
    TRESHOLD_PCT
)
from Market_Scanner_Model_SHORT import find_significant_moves

# --- НАСТРОЙКИ ---
MODEL_TYPE = 'SHORT'
N_TRIALS = 100  # Количество попыток для поиска. Начните с 50, для хорошего результата можно 100-200.


# --- ФУНКЦИЯ ЦЕЛИ ДЛЯ OPTUNA ---
def objective(trial, X, y):
    """
    Эта функция определяет, что мы хотим оптимизировать.
    Мы будем максимизировать F1-score для класса 1 (сигналы на РОСТ).
    """
    # Определяем пространство поиска гиперпараметров
    params = {
        'objective': 'binary',
        'metric': 'logloss',
        'random_state': 33,
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 70),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)  # Optuna сама подберет вес
    }

    model = LiquidityMLModel(params=params)

    # Используем TimeSeriesSplit для кросс-валидации временных рядов
    tscv = TimeSeriesSplit(n_splits=7)
    scores = []

    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.train(X_train, y_train)
        preds = model.model.predict(X_val)

        # Наша главная цель - F1 score для класса 1 (Up moves)
        score = f1_score(y_val, preds, average='macro', zero_division=0)
        scores.append(score)

    return np.mean(scores)


# --- ОСНОВНОЙ БЛОК ---
if __name__ == "__main__":
    print("--- ЭТАП 1: ПОДГОТОВКА ДАННЫХ ДЛЯ ТЮНИНГА ---")
    raw_final_df, df_4h, df_30m, df_15m, df_5m, df_1m = prepare_master_dataframe(START_DATE, TICKER, DOWNLOAD_DATA)
    feature_engine = FeatureEngineSMC(
        main_df_1h=raw_final_df, ltf_df_4h=df_4h, ltf_df_30m=df_30m,
        ltf_df_15m=df_15m, ltf_df_5m=df_5m, ltf_df_1m=df_1m
    )
    X, _, _, enriched_df = feature_engine.run(model_type=MODEL_TYPE, create_target=False)
    y = find_significant_moves(enriched_df, look_forward=LOOK_FORWARD, threshold_pct=TRESHOLD_PCT)

    # Готовим данные так же, как в трейнере
    data_for_training = X.copy()
    data_for_training['target'] = y
    data_for_training = data_for_training[data_for_training['target'] != 0]
    X_filtered = data_for_training.drop(columns=['target'])
    y_binary = data_for_training['target'].replace({-1: 0})

    print(f"\n--- ЭТАП 2: ЗАПУСК OPTUNA ДЛЯ ПОИСКА ЛУЧШИХ ПАРАМЕТРОВ ({N_TRIALS} попыток) ---")
    study = optuna.create_study(direction='maximize')  # Мы МАКСИМИЗИРУЕМ F1-score
    study.optimize(lambda trial: objective(trial, X_filtered, y_binary), n_trials=N_TRIALS)

    print("\n✅ Оптимизация завершена!")
    print("🔥 Лучшие гиперпараметры найдены:", study.best_params)
    print(f"🏆 Лучшее значение F1-score (class 1): {study.best_value:.4f}")

    # Сохраняем лучшие параметры в файл
    safe_ticker = TICKER.replace('/', '_')
    params_filename = f"best_params_{MODEL_TYPE}_{safe_ticker}.json"
    with open(params_filename, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"💾 Гиперпараметры сохранены в {params_filename}")
# ==============================================================================
# Scout_Model_Trainer_REGRESSION_SHORT.py
# ------------------------------------------------------------------------------
# ЗАДАЧА: Обучить, оценить и сохранить лучшую РЕГРЕССИОННУЮ модель-"Разведчика",
# которая предсказывает РАЗМЕР движения после сетапа "London sweeps Asia".
# ==============================================================================

import pandas as pd
import numpy as np
import optuna
import os
import random
import warnings
from dotenv import load_dotenv
import plotly.graph_objects as go
import json

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Импортируем НОВЫЙ класс для регрессии
from trading_tools import (
    LiquidityMLRegressor,
    FeatureEngineSMC,
    prepare_master_dataframe,
    START_DATE,
    TICKER,
    safe_ticker,
    DOWNLOAD_DATA
)

warnings.filterwarnings('ignore')

# --- 1. НАСТРОЙКИ И КОНСТАНТЫ ---
MODEL_TYPE = 'SHORT'
N_TRAILS_MAE = 75  # Количество попыток для поиска лучшей регрессионной модели


def set_seed(seed=33):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    print(f"✅ Случайность зафиксирована с seed = {seed}")


# --- 2. НОВАЯ ФУНКЦИЯ ОПТИМИЗАЦИИ (MAE) ---
def objective_mae_cv(trial, X_train_full, y_train_full):
    """
    Ищем лучшие параметры РЕГРЕССИОННОЙ МОДЕЛИ по среднему MAE.
    """
    model_params = {
        'objective': 'regression_l1',  # L1 loss = MAE
        'metric': 'mae',
        'random_state': 33,
        'verbosity': -1,
        'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = []

    for train_index, val_index in tscv.split(X_train_full):
        X_train_fold, X_val_fold = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
        y_train_fold, y_val_fold = y_train_full.iloc[train_index], y_train_full.iloc[val_index]

        model = LiquidityMLRegressor(params=model_params)
        model.train(X_train_fold, y_train_fold)

        predictions = model.predict(X_val_fold)
        score = mean_absolute_error(y_val_fold, predictions)
        mae_scores.append(score)

    return np.mean(mae_scores)


# --- 3. ОСНОВНОЙ БЛОК ВЫПОЛНЕНИЯ ---
if __name__ == "__main__":
    set_seed(33)
    load_dotenv()

    # --- ЭТАП 1: ЗАГРУЗКА ДАННЫХ ---
    final_df, df_30m, df_15m, df_5m, df_1m = prepare_master_dataframe(START_DATE, TICKER, DOWNLOAD_DATA)

    # --- ЭТАП 2: СОЗДАНИЕ ПРИЗНАКОВ И ФИЛЬТРАЦИЯ ---
    print("\n--- ЭТАП 2: СОЗДАНИЕ ПРИЗНАКОВ И ФИЛЬТРАЦИЯ ДАННЫХ ---")
    feature_engine = FeatureEngineSMC(final_df, df_30m, df_5m, df_1m)
    X, y, _ = feature_engine.run(model_type=MODEL_TYPE, create_target=True)

    # КЛЮЧЕВОЙ ШАГ: Мы будем обучать модель ТОЛЬКО на тех данных, где был сетап
    # Убираем все строки, где y = NaN
    valid_indices = y.dropna().index
    X_filtered = X.loc[valid_indices]
    y_filtered = y.loc[valid_indices]

    print(f"Отфильтровано {len(X_filtered)} событий для обучения регрессионной модели.")

    if len(X_filtered) < 50:
        print("❌ Слишком мало данных для обучения. Скрипт остановлен.")
        exit()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.3, shuffle=False
    )

    # --- ЭТАП 3: ПОИСК ЛУЧШЕЙ РЕГРЕССИОННОЙ МОДЕЛИ ---
    print(f"\n--- ЭТАП 3: ПОИСК ЛУЧШЕЙ РЕГРЕССИОННОЙ МОДЕЛИ ({MODEL_TYPE}) ---")
    study = optuna.create_study(direction='minimize')  # МИНИМИЗИРУЕМ ошибку
    study.optimize(lambda trial: objective_mae_cv(trial, X_train_full, y_train_full), n_trials=N_TRAILS_MAE)

    best_model_hyperparams = study.best_params
    print("\n✅ Оптимизация завершена!")
    print("🔥 Лучшие гиперпараметры для РЕГРЕССОРА найдены:", best_model_hyperparams)

    # Сохраняем параметры в JSON
    params_filename = f"regression_params_{MODEL_TYPE}_{safe_ticker}.json"
    with open(params_filename, 'w') as f:
        json.dump(best_model_hyperparams, f, indent=4)
    print(f"💾 Гиперпараметры сохранены в {params_filename}")

    # --- ЭТАП 4: ФИНАЛЬНОЕ ОБУЧЕНИЕ И ОЦЕНКА ---
    print("\n--- ЭТАП 4: ФИНАЛЬНОЕ ОБУЧЕНИЕ И ОЦЕНКА РЕГРЕССОРА ---")
    final_model = LiquidityMLRegressor(params=best_model_hyperparams)
    print("Обучение финальной модели на полном наборе тренировочных данных...")
    final_model.train(X_train_full, y_train_full)

    model_filename = f"regressor_model_{MODEL_TYPE}_{safe_ticker}.pkl"
    final_model.save_model(model_filename)

    # Оцениваем точность на невидимых данных (X_test)
    results_df = final_model.evaluate(X_test, y_test)

    # --- ЭТАП 5: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ---
    print("\n--- ЭТАП 5: ВИЗУАЛИЗАЦИЯ ТОЧНОСТИ ПРЕДСКАЗАНИЙ ---")
    fig = go.Figure()

    # Scatter plot: Реальность vs Предсказание
    fig.add_trace(go.Scatter(
        x=results_df['y_true'],
        y=results_df['y_pred'],
        mode='markers',
        name='Предсказания',
        marker=dict(color='rgba(100, 181, 246, 0.7)', line=dict(width=1, color='DarkSlateGrey'))
    ))

    # Линия идеального предсказания (y=x)
    fig.add_trace(go.Scatter(
        x=[results_df['y_true'].min(), results_df['y_true'].max()],
        y=[results_df['y_true'].min(), results_df['y_true'].max()],
        mode='lines',
        name='Идеальное предсказание',
        line=dict(color='crimson', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f'Точность Регрессора: Реальный vs Предсказанный R ({TICKER})',
        xaxis_title='Реальное движение (в R)',
        yaxis_title='Предсказанное движение (в R)',
        template='plotly_dark'
    )

    plot_filename = f'regression_accuracy_plot_{MODEL_TYPE}.html'
    fig.write_html(plot_filename)
    print(f"✅ График точности сохранен в файл: {plot_filename}")

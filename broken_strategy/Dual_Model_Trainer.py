
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
    LiquidityMLModel,
    LiquidityMLRegressor,
    FeatureEngineSMC,
    prepare_master_dataframe,
    START_DATE,
    TICKER,
    safe_ticker,
    DOWNLOAD_DATA
)

# Добавьте остальные ваши импорты (dotenv, set_seed, etc.)

# --- 1. КОНСТАНТЫ ---

warnings.filterwarnings('ignore')

# --- 1. НАСТРОЙКИ И КОНСТАНТЫ ---
MODEL_TYPE = 'SHORT'
N_TRAILS_MAE = 100  # Количество попыток для поиска лучшей регрессионной модели


def set_seed(seed=33):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    print(f"✅ Случайность зафиксирована с seed = {seed}")



# --- 2. ОСНОВНОЙ БЛОК ВЫПОЛНЕНИЯ ---
if __name__ == "__main__":
    set_seed(33)
    load_dotenv()

    # --- ЭТАП 1: ДАННЫЕ И ПРИЗНАКИ ---
    final_df, df_30m, df_15m, df_5m, df_1m = prepare_master_dataframe(START_DATE, TICKER, False)
    feature_engine = FeatureEngineSMC(final_df, ltf_df_30m=df_30m, ltf_df_5m=df_5m, ltf_df_1m=df_1m)

    # Получаем X и ДВЕ цели: y_class для Разведчика, y_regr для Снайпера
    X, y_class, y_regr = feature_engine.run(model_type=MODEL_TYPE, create_target=True)

    # --- ЭТАП 2: ОБУЧЕНИЕ МОДЕЛИ №1 ("РАЗВЕДЧИК" - КЛАССИФИКАТОР) ---
    print("\n--- ЭТАП 2: ОБУЧЕНИЕ 'РАЗВЕДЧИКА' (КЛАССИФИКАТОР) ---")

    # 2.1. Готовим данные для классификатора
    classifier_data = X.copy()
    classifier_data['target'] = y_class
    classifier_data.dropna(subset=['target'], inplace=True)  # Удаляем свечи без сетапов

    X_class = classifier_data.drop(columns=['target'])
    y_class_filtered = classifier_data['target']


    # --- НОВЫЙ БЛОК: РАСЧЕТ ВЕСА ДЛЯ БАЛАНСИРОВКИ КЛАССОВ ---
    num_negatives = (y_class_filtered == 0).sum()
    num_positives = (y_class_filtered == 1).sum()

    if num_positives == 0:
        print("❌ КРИТИЧЕСКАЯ ОШИБКА: Не найдено ни одного положительного примера (y=1).")
        exit()

    scale_pos_weight_value = num_negatives / num_positives
    print(f"\nДисбаланс классов: 0 -> {num_negatives}, 1 -> {num_positives}")
    print(f"Рассчитан 'scale_pos_weight': {scale_pos_weight_value:.2f}")
    # --- КОНЕЦ НОВОГО БЛОКА ---

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class_filtered, test_size=0.2, shuffle=False, random_state=33
    )

    # 2.2. Обучаем и оцениваем классификатор
    # Создаем словарь параметров, куда добавляем наш вес
    scout_params = {
        'objective': 'binary',
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 33,
        'deterministic': True,
        'scale_pos_weight': scale_pos_weight_value # <-- ПЕРЕДАЕМ ВЕС СЮДА
    }

    scout_model = LiquidityMLModel()  # Используем параметры по умолчанию для начала
    scout_model.train(X_train_c, y_train_c)
    print("\nОценка 'Разведчика' на тестовых данных:")
    scout_model.evaluate(X_test_c, y_test_c)
    # scout_model.save_model(f"scout_model_{MODEL_TYPE}.pkl")

    # --- ЭТАП 3: ОБУЧЕНИЕ МОДЕЛИ №2 ("СНАЙПЕР" - РЕГРЕССОР) ---
    print("\n--- ЭТАП 3: ОБУЧЕНИЕ 'СНАЙПЕРА' (РЕГРЕССОР) ---")

    # 3.1. Готовим данные: ИСПОЛЬЗУЕМ ТОЛЬКО УСПЕШНЫЕ СДЕЛКИ
    regressor_data = X.copy()
    regressor_data['target'] = y_regr  # Цель - размер движения

    # Оставляем только те строки, где был УСПЕШНЫЙ сетап (y_class == 1)
    # и где есть цель для регрессии
    successful_trades_indices = y_class[y_class == 1].index
    regressor_data = regressor_data.loc[successful_trades_indices]
    regressor_data.dropna(subset=['target'], inplace=True)

    print(f"Найдено {len(regressor_data)} успешных сделок для обучения 'Снайпера'.")

    if len(regressor_data) < 20:
        print("❌ Недостаточно данных для обучения 'Снайпера'.")
    else:
        X_regr = regressor_data.drop(columns=['target'])
        y_regr_filtered = regressor_data['target']

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_regr, y_regr_filtered, test_size=0.2, shuffle=False, random_state=33
        )

        # 3.2. Обучаем и оцениваем регрессор
        sniper_model = LiquidityMLRegressor()
        sniper_model.train(X_train_r, y_train_r)
        print("\nОценка 'Снайпера' на тестовых данных (только успешные сделки):")
        sniper_model.evaluate(X_test_r, y_test_r)
        # sniper_model.save_model(f"sniper_model_{MODEL_TYPE}.pkl")

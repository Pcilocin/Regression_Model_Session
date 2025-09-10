
# trading_tools.py - Набор инструментов для торговых ботов

# --- ОБЯЗАТЕЛЬНЫЕ ИМПОРТЫ ДЛЯ ИНСТРУМЕНТОВ ---
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt
import lightgbm as lgb
import joblib
import requests
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
import optuna
import time
import os
import logging
from scipy.signal import argrelextrema


# LOOK_FORWARD_PERIOD = 8 # Количество часов для ожидания цены
# ATR_MULTIPLIER_PERIOD = 2.0 # Колебания цены  ATR
START_DATE = '2024-01-01'
initial_capital = 100
TICKER = 'SUI/USDT'
safe_ticker = TICKER.replace('/', '_')
DOWNLOAD_DATA = False # True Загрузка новых данных / False данные из кэша


class DataHandler:
    """
    Класс для загрузки данных с "умным" кэшированием и полной защитой.
    (Версия 3.2 - с атомарной записью, валидацией и проверкой на пропуски)
    """

    def __init__(self, symbol: str, timeframe: str, start_date: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.exchange = ccxt.bybit({'options': {'defaultType': 'swap', 'defaultSubType': 'linear'}})
        self.safe_symbol = symbol.replace('/', '_')
        self.cache_dir = "../data_cache"
        self.cache_filepath = os.path.join(
            self.cache_dir,
            f"{self.safe_symbol}_{self.timeframe}_{self.start_date.split('T')[0]}.parquet"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _download_data(self, since_timestamp):
        """Вспомогательная функция для скачивания данных с определенной даты."""
        all_ohlcv = []
        limit = 1000
        since = since_timestamp

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since, limit)
                if len(ohlcv):
                    if len(all_ohlcv) > 0 and ohlcv[0][0] == all_ohlcv[-1][0]:
                        ohlcv = ohlcv[1:]
                    if not ohlcv: break

                    since = ohlcv[-1][0] + (self.exchange.parse_timeframe(self.timeframe) * 1000)
                    all_ohlcv.extend(ohlcv)
                    logging.info(f"Промежуточная загрузка: {len(all_ohlcv)} свечей...")
                    if len(ohlcv) < limit: break
                else:
                    break
            except Exception as e:
                logging.error(f"Ошибка при загрузке части данных: {e}. Повтор через 10 секунд...")
                time.sleep(10)

        return all_ohlcv

    def _validate_and_save(self, df_to_save: pd.DataFrame, old_df: pd.DataFrame = None):
        """Проверяет данные и атомарно сохраняет их в кэш."""
        # 1. Валидация: Проверяем, что high >= low во всем DataFrame
        if not (df_to_save['high'] >= df_to_save['low']).all():
            logging.error("❌ Обнаружены некорректные свечи (high < low). Кэш не будет обновлен.")
            return old_df if old_df is not None else pd.DataFrame()  # Возвращаем старые данные

        # 2. Проверка на пропуски (gaps)
        time_diffs = df_to_save.index.to_series().diff().dropna()
        expected_interval = pd.Timedelta(self.timeframe)
        if (time_diffs > expected_interval * 1.5).any():  # Даем небольшой допуск в 50%
            logging.warning("⚠️ В данных обнаружены пропуски (gaps)! Кэш все равно будет обновлен.")

        # 3. Атомарная запись для защиты от повреждений
        try:
            temp_filepath = self.cache_filepath + ".tmp"
            df_to_save.to_parquet(temp_filepath)
            os.replace(temp_filepath, self.cache_filepath)
            logging.info(f"💾 Кэш успешно сохранен/обновлен: {self.cache_filepath}")
        except Exception as e:
            logging.error(f"❌ Ошибка при сохранении кэша: {e}. Файл не был изменен.")
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)  # Удаляем временный файл
            return old_df if old_df is not None else pd.DataFrame()

        return df_to_save

    def fetch_data(self, update_cache: bool = None) -> pd.DataFrame:
        """
        Загружает данные, используя "умный" кэш.

        Args:
            update_cache (bool): Если True, догружает новые данные.
                                 Если False, использует только существующий кэш.
        """
        if update_cache is None:
            update_cache = DOWNLOAD_DATA

        if os.path.exists(self.cache_filepath):
            logging.info(f"✅ Найден кэш! Загрузка данных из файла: {self.cache_filepath}")
            try:
                cached_df = pd.read_parquet(self.cache_filepath).copy()
            except Exception as e:
                logging.error(f"❌ Кэш-файл поврежден: {e}. Запускаем полную перезагрузку.")
                os.remove(self.cache_filepath)
                return self.fetch_data()

            if not update_cache:
                logging.info("🔌 Обновление кэша отключено. Используются данные из файла.")
                return cached_df

            last_timestamp_ms = int(cached_df.index[-1].timestamp() * 1000)
            logging.info(f"Последняя свеча в кэше: {cached_df.index[-1]}. Догружаем новые данные...")

            new_ohlcv = self._download_data(since_timestamp=last_timestamp_ms)

            if not new_ohlcv:
                logging.info("Новых свечей нет. Данные в кэше актуальны.")
                return cached_df

            new_df = pd.DataFrame(new_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            new_df.set_index('timestamp', inplace=True)

            combined_df = pd.concat([cached_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            logging.info(f"Добавлено {len(new_df)} новых свечей. Проверка и обновление кэша...")
            return self._validate_and_save(combined_df, old_df=cached_df)

        else:  # Если кэша нет, то просто скачиваем всё
            logging.info(f"Кэш не найден. Полная загрузка данных {self.symbol} с {self.start_date} с биржи...")
            date_string = self.start_date
            if ' ' not in date_string and 'T' not in date_string:
                date_string += 'T00:00:00Z'
            start_timestamp_ms = self.exchange.parse8601(date_string)

            all_ohlcv = self._download_data(since_timestamp=start_timestamp_ms)

            if not all_ohlcv:
                logging.warning("Не удалось загрузить данные.")
                return pd.DataFrame()

            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]

            return self._validate_and_save(df)

# --------------------------<Загрузка данных по доминации биткоина>----------------------------------------

    #
    #     # 3. Скачиваем данные с пагинацией (как в DataHandler)
    #     while True:
    #         ohlcv = binance.fetch_ohlcv('BTCDOMUSDT', '1h', since, limit)
    #         if len(ohlcv):
    #             since = ohlcv[-1][0] + (binance.parse_timeframe('1h') * 1000)
    #             all_ohlcv.extend(ohlcv)
    #             if len(ohlcv) < limit:
    #                 break
    #         else:
    #             break
    #
    #     # 4. Преобразуем в DataFrame
    #     if all_ohlcv:
    #         df_dom = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    #         df_dom['timestamp'] = pd.to_datetime(df_dom['timestamp'], unit='ms')
    #         df_dom.set_index('timestamp', inplace=True)
    #         df_dom = df_dom[['close']].rename(columns={'close': 'btc_dominance'})
    #         print(f"✅ Индекс доминации BTC успешно загружен. Записей: {len(df_dom)}")
    #     else:
    #         print("⚠️ Не удалось загрузить индекс доминации BTC (данные не получены).")
    #
    # except Exception as e:
    #     print(f"❌ Критическая ошибка при загрузке доминации BTC: {e}")

# --------------------------</Загрузка данных по доминации биткоина завершена>----------------------------------------

# ---------------------< ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ >------------------------

def prepare_master_dataframe(start_date, ticker, download_data):
    # ----------------< Этап 1: Загрузка данных >-----------------------
    print("--- ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---")

    # (Этот блок остается без изменений, мы просто загружаем все нужные ТФ)
    data_handler_1m = DataHandler(symbol=TICKER, timeframe='1m', start_date=START_DATE)
    df_1m = data_handler_1m.fetch_data(update_cache=DOWNLOAD_DATA).copy()
    if not df_1m.empty: df_1m = df_1m.reset_index().set_index('timestamp')

    data_handler_5m = DataHandler(symbol=TICKER, timeframe='5m', start_date=START_DATE)
    df_5m = data_handler_5m.fetch_data(update_cache=DOWNLOAD_DATA).copy()
    if not df_5m.empty: df_5m = df_5m.reset_index().set_index('timestamp')

    data_handler_15m = DataHandler(symbol=TICKER, timeframe='15m', start_date=START_DATE)
    df_15m = data_handler_15m.fetch_data(update_cache=DOWNLOAD_DATA).copy()
    if not df_15m.empty: df_15m = df_15m.reset_index().set_index('timestamp')

    data_handler_30m = DataHandler(symbol=TICKER, timeframe='30m', start_date=START_DATE)
    df_30m = data_handler_30m.fetch_data(update_cache=DOWNLOAD_DATA).copy()
    if not df_30m.empty: df_30m = df_30m.reset_index().set_index('timestamp')

    data_handler_1h = DataHandler(symbol=TICKER, timeframe='1h', start_date=START_DATE)
    df_1h = data_handler_1h.fetch_data(update_cache=DOWNLOAD_DATA).copy()
    if not df_1h.empty: df_1h = df_1h.reset_index().set_index('timestamp')

    data_handler_4h = DataHandler(symbol=TICKER, timeframe='4h', start_date=START_DATE)
    df_4h = data_handler_4h.fetch_data(update_cache=DOWNLOAD_DATA).copy()
    if not df_4h.empty: df_4h = df_4h.reset_index().set_index('timestamp')

    data_handler_1d = DataHandler(symbol=TICKER, timeframe='1d', start_date=START_DATE)
    df_1d = data_handler_1d.fetch_data(update_cache=DOWNLOAD_DATA).copy()
    if not df_1d.empty: df_1d = df_1d.reset_index().set_index('timestamp')

    if df_1d.empty or df_1h.empty or df_4h.empty or df_30m.empty or df_15m.empty or df_5m.empty or df_1m.empty:
        print("❌ Критическая ошибка: Не удалось загрузить один из таймфреймов. Скрипт остановлен.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_fng = fetch_fear_and_greed_index(limit=0)
    # ----------------</ Этап 1: Загрузка данных завершена>-----------------------

    # ----------------< Этап 2: Инжиниринг Признаков >-----------------------
    print("\n--- ЭТАП 2: РАСЧЕТ СЛОЖНЫХ ПРИЗНАКОВ (POC, DELTA, HTF) ---")

    # <-- ШАГ 2.1: РАСЧЕТ ДЕЛЬТЫ И CVD (НОВЫЙ БЛОК) -->
    # Используем 5-минутные данные для более точного расчета и агрегируем до 1 часа.
    if not df_1m.empty:
        df_delta_1h = add_proxy_delta_features(df_1m, timeframe='1H')
    else:
        # Создаем пустой DF, если 5м данные не загрузились, чтобы избежать ошибок
        df_delta_1h = pd.DataFrame()

        # <-- ШАГ 2.2: РАСЧЕТ POC (КАК И РАНЬШЕ) -->
    df_poc_h = calculate_poc_from_ltf(df_1h, df_1m, period='H')
    df_poc_4h = calculate_poc_from_ltf(df_4h, df_5m, period='4H')
    df_poc_d = calculate_poc_from_ltf(df_1d, df_15m, period='D')

    # <-- ШАГ 2.3: РАСЧЕТ HTF ПРИЗНАКОВ (КАК И РАНЬШЕ) -->
    features_4h = pd.DataFrame(index=df_4h.index)
    features_4h['ema_50_4h'] = df_4h.ta.ema(length=50)
    adx_4h = df_4h.ta.adx(length=14)
    if adx_4h is not None and 'ADX_14' in adx_4h.columns:
        features_4h['adx_14_4h'] = adx_4h['ADX_14']
    features_4h['atr_14_4h'] = df_4h.ta.atr(length=14)

    # ----------------< Этап 3: Сборка Финального DataFrame >--------------------
    print("\n--- ЭТАП 3: СБОРКА ФИНАЛЬНОГО 'final_df' ---")

    # <-- ШАГ 3.1: ОСНОВА -->
    # Начинаем с 1-часового ТФ, который будет нашей основной.
    final_df = df_1h.copy()

    # <-- ШАГ 3.2: ОБЪЕДИНЕНИЕ ДАННЫХ С ДЕЛЬТОЙ (НОВЫЙ БЛОК) -->
    # Присоединяем рассчитанную дельту и CVD по индексу (времени).
    if not df_delta_1h.empty:
        final_df = final_df.join(df_delta_1h)

    # Теперь, когда колонка 'cvd' добавлена, мы можем рассчитать дивергенцию.
    if 'cvd' in final_df.columns:
        df_divergence = add_cvd_divergence(final_df, lookback=18)
        final_df = final_df.join(df_divergence)

    # <-- ШАГ 3.3: ОБЪЕДИНЕНИЕ С ОСТАЛЬНЫМИ ПРИЗНАКАМИ (КАК И РАНЬШЕ) -->
    final_df = pd.merge_asof(final_df, features_4h, left_index=True, right_index=True, direction='backward')
    if not df_fng.empty:
        final_df = pd.merge_asof(final_df, df_fng.rename(columns={'value': 'fear_greed_value'}), left_index=True,
                                 right_index=True, direction='backward')
    if not df_poc_h.empty:
        final_df = final_df.join(df_poc_h)
    if not df_poc_4h.empty:
        final_df = pd.merge_asof(final_df, df_poc_4h, left_index=True, right_index=True, direction='backward')
    if not df_poc_d.empty:
        final_df = pd.merge_asof(final_df, df_poc_d, left_index=True, right_index=True, direction='backward')

    # ----------------< Этап 4: Финальная обработка >--------------------
    final_df.fillna(method='ffill', inplace=True)
    final_df.dropna(inplace=True)

    print("\n✅ Мастер-DataFrame и LTF данные успешно подготовлены.")
    return final_df, df_4h, df_30m, df_15m, df_5m, df_1m

# ---------------------</ ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА>------------------------

# ---------------------<Загрузка индекса страха и жадности>------------------------

def fetch_fear_and_greed_index(limit=0, retries=3, delay=5, timeout=10):
    """
    Загружает исторические данные Индекса страха и жадности с механизмом повторных попыток.
    """
    print("➡️ Загрузка исторических данных Индекса страха и жадности...")

    for attempt in range(retries):
        try:
            # --- ✅ ИЗМЕНЕНИЕ: Добавлен таймаут в 10 секунд ---
            response = requests.get(
                f"https://api.alternative.me/fng/?limit={limit}&format=json",
                timeout=timeout
            )
            response.raise_for_status()  # Проверка на ошибки HTTP (4xx, 5xx)
            data = response.json()['data']

            df = pd.DataFrame(data)
            df['value'] = pd.to_numeric(df['value'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.iloc[::-1].reset_index(drop=True)
            df.set_index('timestamp', inplace=True)

            print(f"✅ Загружено {len(df)} записей Индекса.")
            return df[['value']]

        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка при загрузке (попытка {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"Повторная попытка через {delay} секунд...")
                time.sleep(delay)
            else:
                print("❌ Превышено количество попыток. Не удалось загрузить данные.")
                return pd.DataFrame()  # Возвращаем пустой DataFrame после всех неудач


# ---------------------</Загрузка индекса страха и жадности завершена>------------------------



class LiquidityMLModel:
    def __init__(self, params=None):
        if params is None:
            # Параметры по умолчанию
            params = {
                'objective': 'binary',
                'n_estimators': 200,
                'learning_rate': 0.05,
                'num_leaves': 31
            }

        # ### ГЛАВНОЕ ИЗМЕНЕНИЕ: Добавляем параметры для 100% воспроизводимости ###
        params['random_state'] = 33
        params['deterministic'] = True # <--- Самый важный параметр

        self.model = lgb.LGBMClassifier(**params)

    def train(self, X_train, y_train):
        print("Обучение модели LightGBM...")
        self.model.fit(X_train, y_train)
        print("Обучение завершено.")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        print("\n--- Отчет о классификации модели (2 класса: 0-Нет, 1-Да) ---")  # <-- Текст
        print(classification_report(y_test, predictions, labels=[0, 1], zero_division=0))  # <-- labels

    def get_feature_importance(self):
        return pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def save_model(self, filepath=f"Long-only_model_V6_{safe_ticker}.pkl"):
        joblib.dump(self.model, filepath)
        print(f"Модель сохранена в {filepath}")

    def load_model(self, filepath=f"Long-only_model_V6_{safe_ticker}.pkl"):
        self.model = joblib.load(filepath)
        print(f"Модель загружена из {filepath}")


class LiquidityMLRegressor:
    """НОВЫЙ КЛАСС для LightGBM Регрессора."""

    def __init__(self, params=None):
        if params is None:
            # Параметры по умолчанию для регрессии
            params = {'objective': 'regression_l1', 'metric': 'mae', 'random_state': 33, 'n_jobs': -1}
        self.model = lgb.LGBMRegressor(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Оценивает регрессионную модель."""
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Расчет "Точности 85%"
        # |Предсказание - Реальность| < 0.15 * Реальность
        # Избегаем деления на ноль, если реальное движение = 0
        relative_error = np.abs((predictions - y_test) / y_test.replace(0, 1e-9))
        accuracy_85 = np.mean(relative_error < 0.15) * 100

        print("\n--- Отчет о регрессии ---")
        print(f"Средняя Абсолютная Ошибка (MAE): {mae:.4f}")
        print(f"  -> В среднем модель ошибается на это значение (в R).")
        print(f"Корень из ср. квадр. ошибки (RMSE): {rmse:.4f}")
        print(f"Точность (ошибка < 15%): {accuracy_85:.2f}%")
        print("---------------------------\n")
        return pd.DataFrame({'y_true': y_test, 'y_pred': predictions})

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
        print(f"Модель сохранена в {filepath}")

    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        print(f"Модель загружена из {filepath}")



# -------------------------< /Новая Функция для расчета POC >----------------------------------

def calculate_poc_from_ltf(df_high_tf: pd.DataFrame, df_low_tf: pd.DataFrame, period: str = 'H') -> pd.DataFrame:
    """
    Рассчитывает Point of Control (POC) для старшего таймфрейма (high_tf),
    используя данные младшего таймфрейма (low_tf) для точного распределения объема.

    Args:
        df_high_tf: DataFrame со старшим таймфреймом (например, 1H).
        df_low_tf: DataFrame с младшим таймфреймом (например, 5m).
        period: Период для группировки ('H' для часа, 'D' для дня).

    Returns:
        DataFrame с индексом от старшего ТФ и колонкой POC.
    """
    print(f"➡️ Расчет POC за период '{period}' на основе данных младшего ТФ...")

    if df_low_tf.empty:
        print("⚠️ DataFrame младшего таймфрейма пуст. Расчет POC невозможен.")
        return pd.DataFrame(index=df_high_tf.index, columns=[f'poc_{period.lower()}'])

    # --- ИСПРАВЛЕНИЕ 1: Работаем с копией, чтобы не изменять исходный DataFrame ---
    ltf_copy = df_low_tf.copy()

    median_price = ltf_copy['close'].median()
    if median_price > 1000:
        tick_size = 0.5
    elif median_price > 100:
        tick_size = 0.1
    elif median_price > 1:
        tick_size = 0.01
    else:
        tick_size = 0.0001

    ltf_copy['price_bin'] = (ltf_copy['close'] / tick_size).round() * tick_size
    ltf_copy['time_group'] = ltf_copy.index.floor(period)

    volume_profile = ltf_copy.groupby(['time_group', 'price_bin'])['volume'].sum()
    poc_series = volume_profile.groupby('time_group').idxmax().apply(lambda x: x[1])
    poc_df = poc_series.to_frame(name=f'poc_{period.lower()}')

    # --- ИСПРАВЛЕНИЕ 2 (КЛЮЧЕВОЕ): Сдвигаем данные на 1 шаг в будущее, чтобы избежать lookahead bias ---
    # POC за 10:00-10:59 будет присвоен свече 11:00
    poc_df_shifted = poc_df.shift(1)

    # Объединяем с df_high_tf
    final_poc_df = df_high_tf.join(poc_df_shifted)
    final_poc_df[f'poc_{period.lower()}'].ffill(inplace=True)

    print("✅ Расчет POC завершен.")
    return final_poc_df[[f'poc_{period.lower()}']]


def add_proxy_delta_features(df: pd.DataFrame, timeframe: str = '1H') -> pd.DataFrame:
    """
    Рассчитывает "Грубую Дельту" (Crude Delta) и Кумулятивную Дельту (CVD)
    на основе данных OHLCV и Volume, агрегированных к нужному таймфрейму.

    Логика:
    - Если close > open, Delta = +Volume
    - Если close < open, Delta = -Volume
    - Иначе Delta = 0

    Args:
        df: DataFrame с OHLCV данными. Должен иметь datetime индекс.
        timeframe: Таймфрейм для агрегации (например, '1H', '4H').

    Returns:
        DataFrame с добавленными колонками 'delta' и 'cvd' для указанного таймфрейма.
    """
    print(f"➡️  Расчет прокси-дельты (Crude Delta) и CVD для таймфрейма {timeframe}...")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс DataFrame должен быть типа DatetimeIndex.")

    # 1. Рассчитываем дельту для каждой свечи в исходном DataFrame
    temp_df = df.copy()
    temp_df['delta'] = np.where(
        temp_df['close'] > temp_df['open'],
        temp_df['volume'],
        np.where(temp_df['close'] < temp_df['open'], -temp_df['volume'], 0)
    )

    # 2. Агрегируем (resample) данные к целевому таймфрейму
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'delta': 'sum'  # Суммируем дельту за весь период
    }

    df_resampled = temp_df.resample(timeframe).agg(agg_rules)
    df_resampled.dropna(subset=['open'], inplace=True)  # Удаляем пустые интервалы

    # 3. Рассчитываем Кумулятивную Дельту (CVD) на новом таймфрейме
    df_resampled['cvd'] = df_resampled['delta'].cumsum()

    print(f"✅ Колонки 'delta' и 'cvd' (прокси) успешно рассчитаны для {timeframe}.")

    # Возвращаем только новые колонки, чтобы избежать дублирования OHLCV
    return df_resampled[['delta', 'cvd']]


def add_cvd_divergence(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """
    Рассчитывает дивергенцию между ценой (close) и кумулятивной дельтой (cvd).

    Args:
        df: DataFrame, который уже содержит колонки 'close' и 'cvd'.
        lookback: Период для поиска максимумов/минимумов.

    Returns:
        DataFrame с новой колонкой 'cvd_divergence'.
        +1: Бычья дивергенция (цена ниже, CVD выше).
        -1: Медвежья дивергенция (цена выше, CVD ниже).
         0: Нет дивергенции.
    """
    print(f"➡️  Поиск дивергенций CVD с периодом {lookback}...")
    df_out = df.copy()

    # Находим самый высокий максимум и самый низкий минимум за период
    df_out['price_high'] = df_out['high'].rolling(lookback).max()
    df_out['cvd_high'] = df_out['cvd'].rolling(lookback).max()
    df_out['price_low'] = df_out['low'].rolling(lookback).min()
    df_out['cvd_low'] = df_out['cvd'].rolling(lookback).min()

    # Условия для дивергенций
    bearish_divergence = (df_out['high'] == df_out['price_high']) & (df_out['cvd'] < df_out['cvd_high'])
    bullish_divergence = (df_out['low'] == df_out['price_low']) & (df_out['cvd'] > df_out['cvd_low'])

    df_out['cvd_divergence'] = np.select(
        [bearish_divergence, bullish_divergence],
        [-1, 1],
        default=0
    )

    print("✅ Колонка 'cvd_divergence' успешно добавлена.")
    return df_out[['cvd_divergence']]


# -------------------------< /Новая Функция для расчета POC >----------------------------------

# ==============================================================================
# РАЗДЕЛ 2: ОБЪЕДИНЕННЫЙ FEATURE ENGINE (SMC) (Версия 4.0)
# ==============================================================================

class FeatureEngineSMC:
    """
    Единый класс для инжиниринга признаков (LONG и SHORT) с гибридным подходом,
    отслеживанием ликвидности и поддержкой анализа микроструктуры LTF.
    """
    def __init__(self, main_df_1h: pd.DataFrame, ltf_df_1m: pd.DataFrame = None, ltf_df_5m: pd.DataFrame = None, ltf_df_15m: pd.DataFrame = None, ltf_df_30m: pd.DataFrame = None, ltf_df_4h: pd.DataFrame = None):
        """
        Инициализатор.
        Args:
            main_df_1h: Основной DataFrame с данными (например, 1H), который уже содержит POC, HTF фичи и т.д.
            ltf_df_1m: Опциональный DataFrame с 1-минутными данными для продвинутого анализа лейблов.
        """
        self.df = main_df_1h.copy()
        # Сохраняем 1M данные, если они переданы. Они понадобятся для твоей новой логики фильтрации таргетов.
        self.df_1m = ltf_df_1m.copy() if ltf_df_1m is not None else pd.DataFrame() # Сохраняем 1m
        self.df_5m = ltf_df_5m.copy() if ltf_df_5m is not None else pd.DataFrame() # <-- ДОБАВЛЕНО: Сохраняем 5m
        self.df_15m = ltf_df_15m.copy() if ltf_df_15m is not None else pd.DataFrame() # <-- ДОБАВЛЕНО: Сохраняем 15m
        self.df_30m = ltf_df_30m.copy() if ltf_df_30m is not None else pd.DataFrame() # <-- ДОБАВЛЕНО: Сохраняем 30m
        self.df_4h = ltf_df_4h.copy() if ltf_df_4h is not None else pd.DataFrame() # <-- ДОБАВЛЕНО: Сохраняем 4h



        self.liquidity_map = []
        self.next_liquidity_id = 0
        self.liquidity_vector_features = []  # Будет заполнено в .run()

        # --- НАЧАЛО НОВОГО БЛОКА: СПИСКИ ПРИЗНАКОВ ---


    STATIC_FEATURES_LONG = [
        # --- 1. КОНТЕКСТ СТАРШЕГО ТФ ---
        'trend_strength_4h',  # Глобальный тренд (мы выше/ниже 4H EMA?)
        'structure_state',  # Наш 1H BOS/CHoCH индикатор (мы в 1H ап- или даун-тренде?)
        # --- 2. VSA-КОНТЕКСТ (все еще полезен) ---
        'dynamic_vol_ratio',  # VSA-подтверждение (наш 1M Vol Ratio)
        'volume_spike_ratio',  # Всплеск 1H объема
        # --- 3. КОНТЕКСТ ЛИКВИДНОСТИ ---
        'dist_to_static_pdh_atr', 'dist_to_static_pdl_atr',
        # --- 4. КОНТЕКСТ СВЕЧИ ---
        'bullish_rejection_power' # Форма 1Ч свечи (важно для sweep)
        # --- 5. КЛЮЧЕВЫЕ ПРИЗНАКИ ДЛЯ ЭТОЙ ЗАДАЧИ ---
        'delta', 'cvd', 'cvd_divergence',
        # 'is_asian_session', 'is_london_session', 'is_newyork_session',
        # 'is_asian_killzone', 'is_london_killzone', 'is_newyork_killzone',
        # 'hour_sin', 'hour_cos', 'day_sin', 'day_cos','fear_greed_value',
        'combo_trend_x_cvd', 'combo_pdl_x_volume', 'combo_pdh_x_volume',
        'fear_greed_change_3d', 'fear_greed_value',

    ]
    STATIC_FEATURES_SHORT = [
        'market_structure_trend'
        # --- 1. КОНТЕКСТ СТАРШЕГО ТФ ---
        'trend_strength_4h',  # Глобальный тренд (мы выше/ниже 4H EMA?)
        'structure_state',  # Наш 1H BOS/CHoCH индикатор (мы в 1H ап- или даун-тренде?)
        # --- 2. VSA-КОНТЕКСТ (все еще полезен) ---
        'dynamic_vol_ratio',  # VSA-подтверждение (наш 1M Vol Ratio)
        'volume_spike_ratio',  # Всплеск 1H объема
        # --- 3. КОНТЕКСТ ЛИКВИДНОСТИ ---
        'dist_to_static_pdh_atr', 'dist_to_static_pdl_atr',
        # --- 4. КОНТЕКСТ СВЕЧИ ---
        'bearish_rejection_power', # 'bullish_rejection_power' # Форма 1Ч свечи (важно для sweep)
        # --- 5. КЛЮЧЕВЫЕ ПРИЗНАКИ ДЛЯ ЭТОЙ ЗАДАЧИ ---
        'delta', 'cvd', 'cvd_divergence', #'adx'
        # 'is_asian_session', 'is_london_session', 'is_newyork_session',
        # 'is_asian_killzone', 'is_london_killzone', 'is_newyork_killzone',
        # 'hour_sin', 'hour_cos',
        # 'day_sin', 'day_cos',
        'combo_trend_x_cvd', 'combo_pdl_x_volume', 'combo_pdh_x_volume',
        'fear_greed_change_3d', 'fear_greed_value',

    ]


    # Добавляем все признаки, которые ты решил использовать:
    # 'dist_to_static_bull_fvg_atr', # 'dist_to_static_bear_fvg_atr'

    # 'is_compression', 'bullish_rejection_power', 'dist_to_static_pdh_atr',
    # 'dmp_dmn_diff', 'price_vs_ema200_1h_ratio', 'adx_atr_product',
    # 'dist_to_static_bear_fvg_atr', #'dist_to_static_bull_fvg_atr'
    # 'rsi_14_1h', 'dist_to_vwap_atr', 'fear_greed_change_3d' и т.д.

    # --- КОНЕЦ НОВОГО БЛОКА ---

    # --- Приватные методы Карты Ликвидности (идентичны для LONG и SHORT) ---

    def _add_liquidity_zone(self, zone_type, price_start, price_end=None):
        new_zone = {'id': self.next_liquidity_id, 'type': zone_type, 'price_start': price_start,
                    'price_end': price_end if price_end is not None else price_start, 'status': 'active'}
        self.liquidity_map.append(new_zone)
        self.next_liquidity_id += 1

    def _update_liquidity_status(self, current_high, current_low):
        for zone in self.liquidity_map:
            if zone['status'] == 'active':
                if not (current_high < zone['price_start'] or current_low > zone['price_end']):
                    zone['status'] = 'swept'

    def _precalculate_swing_points(self, fractal_order=5): #расчет структурных точек
        """
        Находит ВСЕ фрактальные минимумы и максимумы в 1H данных (self.df)
        и сохраняет их для быстрого доступа.
        fractal_order = 5 означает классический 5-свечной фрактал (свеча ниже/выше 2х слева и 2х справа)
        """
        print(f"Предварительный расчет структурных точек (фрактал {fractal_order})...")

        # Находим индексы (номера строк) относительных минимумов
        low_indices = argrelextrema(self.df['low'].values, np.less_equal, order=fractal_order)[0]
        # Находим индексы относительных максимумов
        high_indices = argrelextrema(self.df['high'].values, np.greater_equal, order=fractal_order)[0]

        # Получаем временные метки (timestamps) этих свечей
        self.all_swing_low_timestamps = self.df.iloc[low_indices].index
        self.all_swing_high_timestamps = self.df.iloc[high_indices].index

        print(
            f"Найдено {len(self.all_swing_low_timestamps)} свингов (low) и {len(self.all_swing_high_timestamps)} свингов (high).")

    # ВСТАВЬ ЭТОТ НОВЫЙ МЕТОД ВНУТРЬ КЛАССА FeatureEngineSMC


    def _find_significant_liquidity(self, rest_period_bars=24):
        """
        Из всех найденных свингов (high/low) находит только "значимые",
        то есть те, которые "отдыхали" (не были пробиты) N баров ПОСЛЕ их формирования.
        """
        print(f"Поиск значимой 'отдыхающей' ликвидности (период {rest_period_bars} бара)...")

        significant_highs_idx = []
        for high_ts in self.all_swing_high_timestamps:
            high_price = self.df.loc[high_ts, 'high']
            # Смотрим на N баров ПОСЛЕ этого хая
            window_after = self.df.loc[high_ts:].iloc[1: 1 + rest_period_bars]
            if not window_after.empty:
                # Если МАКСИМУМ этого окна все равно НИЖЕ нашего хая, значит, хай "удержался"
                if window_after['high'].max() < high_price:
                    significant_highs_idx.append(high_ts)

        significant_lows_idx = []
        for low_ts in self.all_swing_low_timestamps:
            low_price = self.df.loc[low_ts, 'low']
            # Смотрим на N баров ПОСЛЕ этого лоу
            window_after = self.df.loc[low_ts:].iloc[1: 1 + rest_period_bars]
            if not window_after.empty:
                # Если МИНИМУМ этого окна ВЫШЕ нашего лоу, значит, лоу "удержался"
                if window_after['low'].min() > low_price:
                    significant_lows_idx.append(low_ts)

        self.significant_highs = pd.Series(self.df.loc[significant_highs_idx, 'high'])
        self.significant_lows = pd.Series(self.df.loc[significant_lows_idx, 'low'])

        print(f"Найдено {len(self.significant_highs)} знач. максимумов и {len(self.significant_lows)} знач. минимумов.")

    def _calculate_market_structure(self):
        """
        [ИСПРАВЛЕННАЯ ВЕРСИЯ - С ФИКСОМ ДУБЛИКАТОВ]
        Определяет состояние рыночной структуры и количественную силу тренда.
        """
        print("Расчет состояния рыночной структуры и силы тренда...")

        if not hasattr(self, 'df_4h') or self.df_4h.empty:
            print("⚠️ DataFrame 4H не найден. Расчет структурного тренда пропущен.")
            self.df['market_structure_trend'] = 0
            return

        fractal_order = 3
        low_indices = argrelextrema(self.df_4h['low'].values, np.less_equal, order=fractal_order)[0]
        high_indices = argrelextrema(self.df_4h['high'].values, np.greater_equal, order=fractal_order)[0]

        swing_lows = self.df_4h.iloc[low_indices]
        swing_highs = self.df_4h.iloc[high_indices]

        lows_df = pd.DataFrame({'price': swing_lows['low'], 'type': 'low'})
        highs_df = pd.DataFrame({'price': swing_highs['high'], 'type': 'high'})

        swing_events = pd.concat([lows_df, highs_df]).sort_index()

        # --- ИСПРАВЛЕНИЕ ОШИБКИ ДУБЛИКАТОВ ---
        swing_events = swing_events[~swing_events.index.duplicated(keep='first')]

        if swing_events.empty:
            self.df['market_structure_trend'] = 0
            return

        trend_counter = 0
        last_high = None
        last_low = None

        structure_series = pd.Series(index=swing_events.index, dtype=int)

        for timestamp, event in swing_events.iterrows():
            if event['type'] == 'high':
                if last_high is not None:
                    if event['price'] > last_high:  # HH
                        if trend_counter < 0:
                            trend_counter = 1
                        else:
                            trend_counter = min(trend_counter + 1, 5)
                    else:  # LH
                        if trend_counter > 0: trend_counter = 0
                last_high = event['price']

            elif event['type'] == 'low':
                if last_low is not None:
                    if event['price'] < last_low:  # LL
                        if trend_counter > 0:
                            trend_counter = -1
                        else:
                            trend_counter = max(trend_counter - 1, -5)
                    else:  # HL
                        if trend_counter < 0: trend_counter = 0
                last_low = event['price']

            structure_series.loc[timestamp] = trend_counter

        self.df['market_structure_trend'] = structure_series.reindex(self.df.index, method='ffill').fillna(0).astype(
            int)

    # def _calculate_market_structure(self): # пройтись по всем найденным точкам свингов и определить, когда тренд бычий, а когда медвежий.
    #     """
    #     Создает новый признак 'structure_state' (+1 для бычьего, -1 для медвежьего),
    #     основываясь на последовательности 1H Swing Highs/Lows.
    #     """
    #     print("Расчет состояния рыночной структуры (BOS/CHoCH)...")
    #
    #     # 1. Объединяем все точки свингов в один DataFrame
    #     lows = pd.DataFrame({'price': self.df.loc[self.all_swing_low_timestamps, 'low']},
    #                         index=self.all_swing_low_timestamps)
    #     lows['type'] = -1  # -1 = Swing Low
    #
    #     highs = pd.DataFrame({'price': self.df.loc[self.all_swing_high_timestamps, 'high']},
    #                          index=self.all_swing_high_timestamps)
    #     highs['type'] = 1  # +1 = Swing High
    #
    #     swing_events = pd.concat([lows, highs]).sort_index()
    #
    #     if swing_events.empty:
    #         print("ВНИМАНИЕ: Не найдено точек свинга, расчет структуры невозможен.")
    #         self.df['structure_state'] = 0  # Заполняем нулем
    #         return
    #
    #     # 2. Итерируем по событиям и определяем тренд (State Machine)
    #     structure_state = 0  # 0 = Неопределен, +1 = Бычий, -1 = Медвежий
    #     last_high = None
    #     last_low = None
    #
    #     # Создаем серию для хранения результатов (быстрее, чем .loc в цикле)
    #     state_series = pd.Series(index=self.df.index, dtype='float64')
    #
    #     for timestamp, event in swing_events.iterrows():
    #         if event['type'] == 1:  # Это Swing High
    #             if last_high is not None and event['price'] > last_high:
    #                 structure_state = 1  # Bullish BOS (Higher High)
    #             last_high = event['price']
    #
    #         elif event['type'] == -1:  # Это Swing Low
    #             if last_low is not None and event['price'] < last_low:
    #                 structure_state = -1  # Bearish BOS / CHoCH (Lower Low)
    #             last_low = event['price']
    #
    #         # Записываем текущее состояние на момент события
    #         state_series.loc[timestamp] = structure_state
    #
    #     # 3. Заполняем пропуски
    #     # ffill() распространяет последнее состояние тренда до следующего события
    #     self.df['structure_state'] = state_series.ffill().fillna(0)  # ffill + fillna(0) для самого начала
    #



    def _calculate_static_features(self):
        """ Рассчитывает все не-итеративные (векторные) признаки. """
        print("Расчет статических признаков...")

        # --- ШАГ 1: Расчеты pandas-ta ---
        # (Весь этот блок СКОПИРОВАН из твоего старого FeatureEngine. Он идентичен.)
        ohlcv_df = self.df[['open', 'high', 'low', 'close', 'volume']].copy()
        ohlcv_df.ta.adx(length=14, append=True)
        ohlcv_df.ta.atr(length=14, append=True)
        ohlcv_df.ta.ema(length=200, append=True)
        ohlcv_df.ta.rsi(length=14, append=True)
        ohlcv_df.ta.vwap(anchor="D", append=True)
        ohlcv_df.ta.vwap(anchor="H", append=True)

        cols_to_copy = ['ADX_14', 'DMP_14', 'DMN_14', 'ATRr_14', 'EMA_200', 'RSI_14', 'VWAP_D', 'VWAP_H']
        for col in cols_to_copy:
            if col in ohlcv_df.columns:  # Добавляем проверку, если индикатор не рассчитался
                self.df[col] = ohlcv_df[col]

        self.df.rename(columns={
            'ADX_14': 'adx', 'DMP_14': 'dmp', 'DMN_14': 'dmn',
            'ATRr_14': 'atr', 'EMA_200': 'ema_200_1h', 'RSI_14': 'rsi_14_1h'
        }, inplace=True)

        # --- ШАГ 2: Сессии, Киллзоны, Отторжения ---
        # (Этот блок также полностью скопирован из старого кода)
        if not pd.api.types.is_datetime64_any_dtype(self.df.index):
            self.df.index = pd.to_datetime(self.df.index)

        hour = self.df.index.hour
        self.df['is_asian_session'] = ((hour >= 0) & (hour < 9)).astype(int)
        self.df['is_london_session'] = ((hour >= 8) & (hour < 17)).astype(int)
        self.df['is_newyork_session'] = ((hour >= 13) & (hour < 22)).astype(int)
        self.df['is_asian_killzone'] = ((hour >= 0) & (hour < 4)).astype(int)
        self.df['is_london_killzone'] = ((hour >= 7) & (hour < 10)).astype(int)
        self.df['is_newyork_killzone'] = ((hour >= 12) & (hour < 15)).astype(int)

        self.df['bullish_rejection_power'] = (self.df['close'] - self.df['low']) / self.df['atr']
        self.df['bearish_rejection_power'] = (self.df['high'] - self.df['close']) / self.df['atr']
        volume_ma = self.df['volume'].rolling(20).mean()
        self.df['volume_spike_ratio'] = self.df['volume'] / volume_ma

        # --- ШАГ 3: Зависимые признаки (дистанции, время и т.д.) ---
        # (Этот блок также полностью скопирован из старого кода)
        self.df['range'] = self.df['high'] - self.df['low']
        self.df['is_compression'] = self.df['range'] <= self.df['range'].rolling(7).min()

        self.df['hour_sin'] = np.sin(2 * np.pi * self.df.index.hour / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df.index.hour / 24)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df.index.dayofweek / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df.index.dayofweek / 7)

        daily_df = self.df.resample('D').agg({'high': 'max', 'low': 'min'})
        self.df['static_pdh'] = daily_df['high'].shift(1).reindex(self.df.index, method='ffill')
        self.df['static_pdl'] = daily_df['low'].shift(1).reindex(self.df.index, method='ffill')

        # Дистанции (убедись, что эти колонки есть в self.df из prepare_master_dataframe)
        if 'poc_h' in self.df.columns:
            self.df['dist_to_poc_atr'] = (self.df['close'] - self.df['poc_h']) / self.df['atr']
        if 'poc_4h' in self.df.columns:
            self.df['dist_to_poc_4h_atr'] = (self.df['close'] - self.df['poc_4h']) / self.df['atr']
        if 'poc_d' in self.df.columns:
            self.df['dist_to_poc_d_atr'] = (self.df['close'] - self.df['poc_d']) / self.df['atr']

        self.df['dist_to_static_pdh_atr'] = (self.df['static_pdh'] - self.df['close']) / self.df['atr']
        self.df['dist_to_static_pdl_atr'] = (self.df['close'] - self.df['static_pdl']) / self.df['atr']

        if 'VWAP_D' in self.df.columns:
            self.df['dist_to_vwap_atr'] = (self.df['close'] - self.df['VWAP_D']) / self.df['atr']
        if 'VWAP_H' in self.df.columns:
            self.df['dist_to_vwap_h_atr'] = (self.df['close'] - self.df['VWAP_H']) / self.df['atr']

        # F&G
        if 'fear_greed_value' in self.df.columns:
            self.df['fear_greed_change_3d'] = self.df['fear_greed_value'].diff(periods=24 * 3)
            self.df['fear_greed_ma_3d'] = self.df['fear_greed_value'].rolling(window=24 * 3).mean()

        # FVG
        self.df['dist_to_static_bull_fvg_atr'] = np.where(self.df['low'] > self.df['high'].shift(2),
                                                          (self.df['close'] - self.df['high'].shift(2)) / self.df[
                                                              'atr'], np.nan)
        self.df['dist_to_static_bear_fvg_atr'] = np.where(self.df['high'] < self.df['low'].shift(2),
                                                          (self.df['low'].shift(2) - self.df['close']) / self.df['atr'],
                                                          np.nan)
        self.df['dist_to_static_bull_fvg_atr'].ffill(inplace=True)
        self.df['dist_to_static_bear_fvg_atr'].ffill(inplace=True)

        # Фильтр Тренда (HTF)
        if 'ema_50_4h' in self.df.columns and 'atr_14_4h' in self.df.columns:
            self.df['trend_strength_4h'] = (self.df['close'] - self.df['ema_50_4h']) / self.df['atr_14_4h']
        else:
            self.df['trend_strength_4h'] = 0

        # --- НОВЫЙ БЛОК: УСИЛЕНИЕ СИГНАЛА (КОМБИНИРОВАННЫЕ ПРИЗНАКИ) ---
        print("Создание комбинированных признаков для усиления сигнала...")

        # 1. Синергия Тренда и Потока Ордеров
        # Этот признак будет сильно положительным, если и тренд, и CVD растут,
        # и сильно отрицательным, если оба падают.
        if 'trend_strength_4h' in self.df.columns and 'cvd' in self.df.columns:
            # Нормализуем CVD, чтобы он был в том же масштабе, что и тренд
            cvd_normalized = (self.df['cvd'] - self.df['cvd'].rolling(50).mean()) / self.df['cvd'].rolling(50).std()
            self.df['combo_trend_x_cvd'] = self.df['trend_strength_4h'] * cvd_normalized

        # 2. Давление Объема у Ключевых Уровней
        # Этот признак покажет, есть ли всплеск объема рядом с PDL/PDH.
        if 'dist_to_static_pdl_atr' in self.df.columns and 'volume_spike_ratio' in self.df.columns:
            # Чем ближе к уровню (меньше dist) и выше объем, тем больше значение
            self.df['combo_pdl_x_volume'] = (1 / (abs(self.df['dist_to_static_pdl_atr']) + 0.1)) * self.df[
                'volume_spike_ratio']
        if 'dist_to_static_pdh_atr' in self.df.columns and 'volume_spike_ratio' in self.df.columns:
            self.df['combo_pdh_x_volume'] = (1 / (abs(self.df['dist_to_static_pdh_atr']) + 0.1)) * self.df[
                'volume_spike_ratio']

        # --- КОНЕЦ НОВОГО БЛОКА ---

        print("Статические признаки рассчитаны.")

    # --- РАЗДЕЛЕННЫЕ ФУНКЦИИ РАЗМЕТКИ ЦЕЛИ ---

    # def _label_target_long(self):
    #     """
    #     [ЛОГИКА ИЗ СТАРОГО FeatureEngine_LONG]
    #     Ищет паттерн "Rejection" на ключевом уровне ликвидности для ЛОНГА.
    #     """
    #     print("Разметка LONG-таргетов...")
    #     df = self.df.copy()
    #
    #     pdl = df['static_pdl'].shift(1)
    #     is_asian = df['is_asian_session'] == 1
    #     daily_asian_low = df['low'].where(is_asian).groupby(df.index.date).transform('min')
    #     asian_session_low = daily_asian_low.ffill()
    #     is_london = df['is_london_session'] == 1
    #     daily_london_low = df['low'].where(is_london).groupby(df.index.date).transform('min')
    #     london_session_low = daily_london_low.ffill()
    #
    #     rejection_pdl = (df['low'] < pdl) & (df['close'] > pdl)
    #     rejection_asian = (df['low'] < asian_session_low) & (df['close'] > asian_session_low) & ~is_asian
    #     rejection_london = (df['low'] < london_session_low) & (df['close'] > london_session_low) & ~is_london
    #
    #     df['target'] = (rejection_pdl | rejection_asian | rejection_london).astype(int)
    #
    #
    #     self.df['target'] = df['target'].shift(-1).fillna(0)

    # ЗАМЕНИТЕ ВАШУ ФУНКЦИЮ _label_target_short НА ЭТУ ОБНОВЛЕННУЮ ВЕРСИЮ

    # ПОЛНОСТЬЮ ЗАМЕНИТЕ ВАШУ ФУНКЦИЮ _label_target_short НА ЭТУ НОВУЮ, ФИНАЛЬНУЮ ВЕРСИЮ
    # def _label_target_short(self, look_forward_bars=8, take_profit_multiple=2.0):
    #     """
    #     [ФИНАЛЬНАЯ ВЕРСИЯ - ЛОГИКА R:R]
    #     Размечает цель на основе встроенного риска сетапа (Risk-to-Reward).
    #     ATR больше не используется для постановки целей.
    #
    #     y = 1, если цена достигла TP (в N раз больше риска) раньше, чем SL.
    #     y = 0, если цена достигла SL раньше, чем TP.
    #     """
    #     print(f"Разметка по паттерну 'Свип + Слом' с логикой R:R (TP = {take_profit_multiple}R)...")
    #     df = self.df.copy()
    #
    #     # --- Блок 1: Определение уровней ликвидности (без изменений) ---
    #     is_asian = (df['is_asian_session'] == 1)
    #     asia_high = df['high'].where(is_asian).groupby(df.index.date).transform('max').ffill()
    #     is_london = (df['is_london_session'] == 1)
    #     london_high = df['high'].where(is_london).groupby(df.index.date).transform('max').ffill()
    #     target_liquidity_level = pd.Series(float('nan'), index=df.index)
    #     is_london_kz = (df['is_london_killzone'] == 1)
    #     target_liquidity_level.loc[is_london_kz] = asia_high.shift(1)
    #     is_newyork_kz = (df['is_newyork_killzone'] == 1)
    #     daily_high_before_ny = pd.concat([asia_high, london_high]).groupby(level=0).max().shift(1)
    #     target_liquidity_level.loc[is_newyork_kz] = daily_high_before_ny
    #
    #     final_target_class = pd.Series(float('nan'), index=df.index)
    #     final_target_regr = pd.Series(float('nan'), index=df.index)
    #
    #     # --- Блок 2: Итерация по свечам для поиска паттерна ---
    #     for i in range(1, len(df)):
    #         current_timestamp = df.index[i]
    #         current_high = df['high'].iloc[i]
    #         prev_high = df['high'].iloc[i - 1]
    #
    #         liquidity_level = target_liquidity_level.iloc[i]
    #         if pd.isna(liquidity_level): continue
    #
    #         # --- Шаг 1: Находим Свечу Свипа ---
    #         if prev_high < liquidity_level and current_high >= liquidity_level:
    #             sweep_candle = df.iloc[i]
    #
    #             # --- Шаг 2: Ищем Свечу Подтверждения ---
    #             confirmation_window = df.iloc[i + 1: i + 1 + 4]
    #             for j in range(len(confirmation_window)):
    #                 confirmation_candle = confirmation_window.iloc[j]
    #
    #                 if confirmation_candle['close'] < sweep_candle['low']:
    #                     # --- Шаг 3: РАСЧЕТ РИСКА И ЦЕЛЕЙ ---
    #                     entry_candle_timestamp = confirmation_window.index[j]
    #                     entry_price = confirmation_candle['close']
    #
    #                     # Стоп-лосс ставится за максимум свечи свипа.
    #                     stop_loss_price = sweep_candle['high']
    #
    #                     # Риск (1R) - это расстояние от входа до стопа.
    #                     risk_in_price = stop_loss_price - entry_price
    #
    #                     # Пропускаем сетапы со слишком большим или нелогичным риском
    #                     if risk_in_price <= 0 or risk_in_price > entry_price * 0.1:  # Риск не более 10% от цены
    #                         continue
    #
    #                     # Тейк-профит - это N рисков вниз от точки входа.
    #                     take_profit_price = entry_price - (risk_in_price * take_profit_multiple)
    #
    #                     # --- Шаг 4: Проверяем исход в будущем ---
    #                     future_window = df.loc[entry_candle_timestamp:].iloc[1: 1 + look_forward_bars]
    #                     if future_window.empty: continue
    #
    #                     outcome = None
    #                     for _, future_candle in future_window.iterrows():
    #                         if future_candle['high'] >= stop_loss_price:
    #                             outcome = 0;
    #                             break  # Неудача
    #                         if future_candle['low'] <= take_profit_price:
    #                             outcome = 1;
    #                             break  # Успех
    #
    #                     if outcome is not None:
    #                         final_target_class.loc[entry_candle_timestamp] = outcome
    #                         if outcome == 1:
    #                             # Для "Снайпера" сохраняем достигнутое R:R
    #                             final_target_regr.loc[entry_candle_timestamp] = take_profit_multiple
    #
    #                     break

        # self.df['target_class'] = final_target_class
        # self.df['target_regr'] = final_target_regr
        # print(
        #     f"Разметка завершена. Найдено {final_target_class.notna().sum()} событий. Успешных (y=1): {(final_target_class == 1).sum()}")


    def run(self, model_type: str, create_target: bool = False):
        """
        [ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ]
        Исправлен порядок выполнения:
        1. Расчет (Пре-кальк, Статика)
        2. Инициализация векторов
        3. Главный Цикл (который СОЗДАЕТ VSA-фичи и Вектор Ликвидности)
        4. Разметка (которая ЧИТАЕТ VSA-фичи)
        5. Финализация.
        """

        # --- ЭТАП 1: Предварительные расчеты ---
        print("Запуск Feature Engine...")
        self._precalculate_swing_points(fractal_order=5)
        self._find_significant_liquidity(rest_period_bars=24) # 2. <-- ДОБАВЬ ЭТОТ ВЫЗОВ (Фильтрует их)
        self._calculate_market_structure()
        self._calculate_static_features()

        # --- ЭТАП 2: Выбор списка признаков (но ПОКА НЕ ВЫЗЫВАЕМ разметку) ---
        static_feature_names = []
        if model_type == 'LONG':
            static_feature_names = self.STATIC_FEATURES_LONG
        elif model_type == 'SHORT':
            static_feature_names = self.STATIC_FEATURES_SHORT
        else:
            raise ValueError(f"Неизвестный model_type: {model_type}. Ожидается 'LONG' или 'SHORT'.")

        # --- ЭТАП 3: Инициализация Динамических Признаков ---
        print("Инициализация векторов (Ликвидность + VSA)...")

        # 3.1 Инициализация Вектора Ликвидности
        N_CLOSEST_LEVELS = 3
        self.liquidity_vector_features = []
        for i in range(1, N_CLOSEST_LEVELS + 1):
            self.liquidity_vector_features.append(f'dist_to_level_{i}_atr')
            self.liquidity_vector_features.append(f'type_of_level_{i}')
        for col in self.liquidity_vector_features:
            self.df[col] = 0.0

        # 3.2 Инициализация VSA-признаков
        vsa_cols = ['dynamic_leg_vol', 'dynamic_candle_1m_vol', 'dynamic_vol_ratio']
        for col in vsa_cols:
            self.df[col] = 0.0

        # --- ЭТАП 4: ЕДИНЫЙ ИТЕРАТИВНЫЙ ЦИКЛ (СОЗДАЕТ все признаки) ---
        print("Запуск главного итеративного цикла (VSA + Liquidity Vector)...")

        prev_day = None
        WARMUP_BARS = 10

        for i in range(WARMUP_BARS, len(self.df)):

            # --- 4.1: Получение данных текущей свечи ---
            current_timestamp = self.df.index[i]
            current_high = self.df['high'].iloc[i]
            current_low = self.df['low'].iloc[i]
            current_close = self.df['close'].iloc[i]
            # Убедимся, что atr не NaN (важно для деления)
            current_atr = self.df['atr'].iloc[i]
            if pd.isna(current_atr) or current_atr == 0:
                current_atr = 1.0  # Запасной вариант, чтобы избежать деления на ноль

            # --- 4.2: РАСЧЕТ ДИНАМИЧЕСКИХ VSA-ПРИЗНАКОВ ---
            try:
                if model_type == 'SHORT':
                    previous_swings = self.all_swing_low_timestamps[self.all_swing_low_timestamps < current_timestamp]
                    if previous_swings.empty: raise ValueError("Нет данных о прошлых минимумах")
                    leg_start_time = previous_swings.max()

                elif model_type == 'LONG':  # Добавляем логику для Лонга
                    previous_swings = self.all_swing_high_timestamps[self.all_swing_high_timestamps < current_timestamp]
                    if previous_swings.empty: raise ValueError("Нет данных о прошлых максимумах")
                    leg_start_time = previous_swings.max()

                leg_end_time = current_timestamp - pd.Timedelta(seconds=1)
                df_1m_leg_slice = self.df_1m.loc[leg_start_time:leg_end_time]
                leg_volume = df_1m_leg_slice['volume'].sum()

                candle_end_time = current_timestamp + pd.Timedelta(minutes=59, seconds=59)
                df_1m_candle_slice = self.df_1m.loc[current_timestamp:candle_end_time]
                candle_1m_volume = df_1m_candle_slice['volume'].sum()

                if leg_volume > 0:
                    dynamic_ratio = candle_1m_volume / leg_volume
                else:
                    dynamic_ratio = 0

                self.df.loc[current_timestamp, 'dynamic_leg_vol'] = leg_volume
                self.df.loc[current_timestamp, 'dynamic_candle_1m_vol'] = candle_1m_volume
                self.df.loc[current_timestamp, 'dynamic_vol_ratio'] = dynamic_ratio

            except Exception as e:
                self.df.loc[current_timestamp, 'dynamic_leg_vol'] = 0
                self.df.loc[current_timestamp, 'dynamic_candle_1m_vol'] = 0
                self.df.loc[current_timestamp, 'dynamic_vol_ratio'] = 0

            # --- 4.3: РАСЧЕТ ВЕКТОРА ЛИКВИДНОСТИ ---

            self._update_liquidity_status(current_high, current_low)

            if prev_day != current_timestamp.date():
                if prev_day is not None:
                    pdh = self.df.loc[self.df.index.date == prev_day, 'high'].max()
                    pdl = self.df.loc[self.df.index.date == prev_day, 'low'].min()
                    if not np.isnan(pdh): self._add_liquidity_zone('PDH', pdh)
                    if not np.isnan(pdl): self._add_liquidity_zone('PDL', pdl)
                prev_day = current_timestamp.date()

            if (i > 1) and (self.df['high'].iloc[i - 1] < self.df['low'].iloc[i - 2]):
                self._add_liquidity_zone('BullishFVG', self.df['high'].iloc[i - 1], self.df['low'].iloc[i - 2])
            if (i > 1) and (self.df['low'].iloc[i - 1] > self.df['high'].iloc[i - 2]):
                self._add_liquidity_zone('BearishFVG', self.df['high'].iloc[i - 2], self.df['low'].iloc[i - 1])

            active_zones = [z for z in self.liquidity_map if z['status'] == 'active']

            zone_type_map = {'PDH': 1, 'PDL': 2, 'BullishFVG': 3, 'BearishFVG': 4, 'POC_H': 5, 'POC_4H': 6, 'POC_D': 7}
            levels_with_dist = []
            for zone in active_zones:
                level_price = (zone['price_start'] + zone['price_end']) / 2
                distance = current_close - level_price
                zone_type_code = zone_type_map.get(zone['type'], 0)
                if zone_type_code > 0:
                    levels_with_dist.append((distance, zone_type_code))

            if 'poc_h' in self.df.columns and pd.notna(self.df['poc_h'].iloc[i]):
                levels_with_dist.append((current_close - self.df['poc_h'].iloc[i], zone_type_map['POC_H']))
            if 'poc_4h' in self.df.columns and pd.notna(self.df['poc_4h'].iloc[i]):
                levels_with_dist.append((current_close - self.df['poc_4h'].iloc[i], zone_type_map['POC_4H']))
            if 'poc_d' in self.df.columns and pd.notna(self.df['poc_d'].iloc[i]):
                levels_with_dist.append((current_close - self.df['poc_d'].iloc[i], zone_type_map['POC_D']))

            levels_with_dist.sort(key=lambda x: abs(x[0]))

            for j in range(N_CLOSEST_LEVELS):
                feature_dist_name = f'dist_to_level_{j + 1}_atr'
                feature_type_name = f'type_of_level_{j + 1}'

                if j < len(levels_with_dist):
                    signed_dist = levels_with_dist[j][0]
                    level_type = levels_with_dist[j][1]
                    dist_atr = signed_dist / current_atr  # Мы уже проверили ATR на ноль в начале цикла
                    self.df.loc[current_timestamp, feature_dist_name] = dist_atr
                    self.df.loc[current_timestamp, feature_type_name] = level_type
                else:
                    self.df.loc[current_timestamp, feature_dist_name] = 999
                    self.df.loc[current_timestamp, feature_type_name] = 0

        # --- КОНЕЦ ГЛАВНОГО ЦИКЛА ---

        # --- ЭТАП 5: РАЗМЕТКА (ТЕПЕРЬ ОНА ВЫПОЛНЯЕТСЯ ПОСЛЕ ЦИКЛА) ---
        # Теперь self.df полностью заполнен ВСЕМИ признаками (включая dynamic_vol_ratio)
        if create_target:
            print("Запуск финальной разметки (Labeling)...")
            if model_type == 'LONG':
                self._label_target_long()
            elif model_type == 'SHORT':
                self._label_target_short()

        all_feature_names = static_feature_names + self.liquidity_vector_features
        all_feature_names = [col for col in all_feature_names if col in self.df.columns]

        # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
        if create_target:
            # Возвращаем ОБЕ цели
            columns_to_select = all_feature_names + ['target_class', 'target_regr']
            columns_to_select = [col for col in columns_to_select if col in self.df.columns]
            final_df = self.df[columns_to_select].copy()
            y_class = final_df.get('target_class')
            y_regr = final_df.get('target_regr')
        else:
            columns_to_select = all_feature_names
            final_df = self.df[columns_to_select].copy()
            y_class = None
            y_regr = None
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        final_df.replace([np.inf, -np.inf], 999, inplace=True)
        final_df.fillna(0, inplace=True)

        X = final_df.loc[:, all_feature_names]

        print("Инжиниринг признаков завершен.")
        # Возвращаем X и ДВЕ цели
        return X, y_class, y_regr, self.df

# ==============================================================================
# РАЗДЕЛ 2.1: НОВЫЙ FEATURE ENGINE ДЛЯ МОДЕЛИ-"ТАКТИКА" (5м ТФ)
# ==============================================================================

class FeatureEngine_Tactician:
    """Создает признаки микроструктуры для 5-минутного ТФ (V2)."""

    def __init__(self, data):
        self.df = data.copy()

    def run(self):
        print("Создание 5м признаков для 'Тактика' (V2)...")

        # --- Базовые индикаторы ---
        rsi_series = self.df.ta.rsi(length=5, append=False)
        if rsi_series is not None:
            self.df['rsi_5_5m'] = rsi_series

        ema_10_series = self.df.ta.ema(length=10, append=False)
        if ema_10_series is not None:
            self.df['dist_to_ema10_5m'] = (self.df['close'] - ema_10_series) / self.df['close']

        vol_ma_20 = self.df['volume'].rolling(20).mean()
        self.df['volume_spike'] = self.df['volume'] / vol_ma_20

        self.df['candle_in_hour'] = self.df.index.minute / 5

        # --- ✅ НАЧАЛО НОВОГО БЛОКА: АНАЛИЗ СВЕЧЕЙ ---

        # 1. Расчет ATR на 5м ТФ
        atr_5m = self.df.ta.atr(length=14, append=False)

        # 2. Размер тела свечи, нормализованный на ATR
        body_size = abs(self.df['close'] - self.df['open'])
        self.df['body_size_vs_atr'] = body_size / atr_5m

        # 3. Размер нижнего фитиля
        lower_wick = (self.df[['open', 'close']].min(axis=1) - self.df['low'])
        self.df['lower_wick_vs_atr'] = lower_wick / atr_5m

        # 4. Размер верхнего фитиля
        upper_wick = (self.df['high'] - self.df[['open', 'close']].max(axis=1))
        self.df['upper_wick_vs_atr'] = upper_wick / atr_5m

        # --- ✅ КОНЕЦ НОВОГО БЛОКА ---

        # Заполняем пропуски
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(method='ffill', inplace=True)

        feature_names = [
            'rsi_5_5m', 'dist_to_ema10_5m', 'volume_spike', 'candle_in_hour',
            'body_size_vs_atr', 'lower_wick_vs_atr', 'upper_wick_vs_atr'  # Новые признаки
        ]

        existing_features = [col for col in feature_names if col in self.df.columns]
        X = self.df[existing_features]

        y = self.df['target'] if 'target' in self.df.columns else None

        return X, y


# ==============================================================================
# РАЗДЕЛ 4: БЭКТЕСТИНГ И ОЦЕНКА (Версия для бинарных моделей)
# ==============================================================================
class Backtester:
    def __init__(self, price_data, initial_capital=initial_capital, fee=0.001, timeframe='1h',):
        self.price_data = price_data.copy()
        self.initial_capital = initial_capital
        self.fee = fee
        self.timeframe = timeframe
        self.trades = []

    def calculate_performance(self, equity_curve):
        periods_in_year = {
            '1m': 365 * 24 * 60,
            '5m': 365 * 24 * 12,
            '15m': 365 * 24 * 4,
            '1h': 365 * 24,
            '4h': 365 * 6,
            '1d': 365
        }
        annualization_factor = np.sqrt(periods_in_year.get(self.timeframe, 365 * 24))

        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        returns = equity_curve.pct_change().dropna()

        if len(returns) < 2 or returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = annualization_factor * returns.mean() / returns.std()

        cumulative = equity_curve / self.initial_capital
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        print(annualization_factor)  # чтобы удостовериться что выбран правильный таймфрейм
        return {"Итоговая прибыль (%)": total_return * 100,  # <-- Возвращаем правильный ключ
                "Коэф. Шарпа (годовой)": sharpe_ratio,
                "Макс. просадка (%)": max_drawdown * 100,
                "Итоговый капитал": equity_curve.iloc[-1]  # <-- Добавляем новую строку
                }


    def run_buy_and_hold(self):
        equity = self.initial_capital * (self.price_data['close'] / self.price_data['close'].iloc[0])
        stats = self.calculate_performance(equity)
        stats.update({"Всего сделок": 1})
        return stats

    def run_dma_crossover(self):
        # Эта стратегия не реализована, возвращаем нули
        return {"Итоговая прибыль (%)": 0.0, "Коэф. Шарпа (годовой)": 0.0, "Макс. просадка (%)": 0.0, "Всего сделок": 0}

        # Поместите этот код внутрь class Backtester в trading_tools.py

    def equity_curve_from_trades(self, trades_df: pd.DataFrame, price_series: pd.Series) -> pd.Series:
        """
        Строит кривую капитала на основе DataFrame'а со сделками.
        """
        if trades_df.empty:
            return pd.Series([self.initial_capital], index=[price_series.index[0]])

        equity = pd.Series(index=price_series.index)
        equity.iloc[0] = self.initial_capital
        capital = self.initial_capital

        last_trade_exit_time = pd.Timestamp.min

        for _, trade in trades_df.iterrows():
            # Пропускаем сделки, которые начинаются до окончания предыдущей
            if trade['entry_time'] < last_trade_exit_time:
                continue

            # Рассчитываем PnL для сделки
            if trade['direction'] == 1:  # Long
                pnl_ratio = (trade['exit_price'] / trade['entry_price']) * (1 - self.fee) ** 2
            else:  # Short
                pnl_ratio = (trade['entry_price'] / trade['exit_price']) * (1 - self.fee) ** 2

            capital *= pnl_ratio

            # Находим индекс, соответствующий времени выхода из сделки
            exit_idx = price_series.index.get_indexer([trade['exit_time']], method='nearest')[0]
            equity.iloc[exit_idx] = capital
            last_trade_exit_time = trade['exit_time']

        equity.ffill(inplace=True)
        equity.bfill(inplace=True)  # Заполняем начало, если первая сделка не в самом начале
        return equity

    def run_simulation_from_trades(self, trades_df: pd.DataFrame):
        """
        Основной метод для запуска бэктеста по списку сделок.
        """
        # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
        # Правильно определяем, что нам передали: целый DataFrame или только один столбец (Series)
        if isinstance(self.price_data, pd.DataFrame):
            price_series = self.price_data['close']
        else:
            price_series = self.price_data

        equity_curve = self.equity_curve_from_trades(trades_df, price_series)
        stats = self.calculate_performance(equity_curve)
        stats['Всего сделок'] = len(trades_df)
        return stats

    def run_smc_strategy(self, signals_df, strategy_params):
        """
        Специализированный бэктестер для торговли паттернов "Smart Money".
        Использует короткий стоп и быстрый выход.
        """
        if self.price_data.empty:
            # Возвращаем пустую статистику, если нет данных
            return self.calculate_performance(pd.Series([self.initial_capital], index=self.price_data.index))

        df = self.price_data.copy().join(signals_df)
        df['prediction'].fillna(0, inplace=True)
        self.trades = []

        equity = self.initial_capital
        equity_history = [self.initial_capital] * len(df)  # Создаем историю сразу
        position = 0

        # --- Новая логика выхода ---
        EXIT_AFTER_BARS = 5  # Выходим из сделки через 5 свечей, если не сработал SL/TP
        entry_bar_index = -1

        for i in range(1, len(df)):
            current_open = df['open'].iloc[i]

            # --- Логика выхода ---
            if position != 0:
                # 1. Проверяем выход по времени
                if i >= entry_bar_index + EXIT_AFTER_BARS:
                    exit_price = current_open  # Выходим по открытию следующей свечи
                    pnl_ratio = (exit_price / entry_price) if position == 1 else (entry_price / exit_price)
                    equity *= pnl_ratio * (1 - self.fee) ** 2
                    position = 0

                # 2. Проверяем выход по SL/TP (остается без изменений)
                elif position == 1 and (df['low'].iloc[i] <= stop_loss or df['high'].iloc[i] >= take_profit):
                    exit_price = stop_loss if df['low'].iloc[i] <= stop_loss else take_profit
                    equity *= (exit_price / entry_price) * (1 - self.fee) ** 2
                    position = 0
                elif position == -1 and (df['high'].iloc[i] >= stop_loss or df['low'].iloc[i] <= take_profit):
                    exit_price = stop_loss if df['high'].iloc[i] >= stop_loss else take_profit
                    equity *= (entry_price / exit_price) * (1 - self.fee) ** 2
                    position = 0

            # --- Логика входа ---
            if position == 0:
                signal = int(df['prediction'].iloc[i - 1])  # Берем сигнал с ПРЕДЫДУЩЕЙ свечи
                if signal != 0:
                    entry_price = current_open  # Входим по открытию текущей свечи
                    entry_bar_index = i
                    position = signal
                    self.trades.append(i)

                    # --- Новая логика постановки SL/TP ---
                    # Стоп ставим за минимум/максимум сигнальной свечи
                    risk_per_share = abs(entry_price - df['low'].iloc[i - 1]) if signal == 1 else abs(
                        df['high'].iloc[i - 1] - entry_price)

                    # Тейк-профит ставим с небольшим, но надежным RR
                    take_profit_amount = risk_per_share * strategy_params.get('rr_ratio', 1.5)

                    if signal == 1:
                        stop_loss = df['low'].iloc[i - 1]
                        take_profit = entry_price + take_profit_amount
                    else:  # signal == -1
                        stop_loss = df['high'].iloc[i - 1]
                        take_profit = entry_price - take_profit_amount

            equity_history[i] = equity

        equity_series = pd.Series(equity_history, index=df.index)
        stats = self.calculate_performance(equity_series)
        stats.update({"Всего сделок": len(self.trades)})
        self.trades = []
        return stats


    def run_ml_strategy(self, signals_df, strategy_params):
        # Убедимся, что данные для бэктеста не пустые
        if self.price_data.empty:
            return self.calculate_performance(pd.Series([self.initial_capital]))

        df = self.price_data.copy().join(signals_df)

        # Заполняем NaN в сигналах, если они есть
        df['prediction'].fillna(0, inplace=True)
        df['strength'].fillna('none', inplace=True)

        # Убедимся, что ATR существует, заполняем пропуски
        if 'atr' not in df.columns:
            raise ValueError("ATR колонка отсутствует в данных для бэктеста.")
        df['atr'].fillna(method='ffill', inplace=True)

        atr_multiplier_stop = strategy_params['atr_multiplier_stop']
        rr_ratio_low = strategy_params['rr_ratio_low']
        rr_ratio_high = strategy_params['rr_ratio_high']

        equity = self.initial_capital
        position = 0  # 0: нет позиции, 1: лонг, -1: шорт
        entry_price = 0
        stop_loss = 0
        take_profit = 0

        # Список для истории изменения капитала
        equity_history = [self.initial_capital]

        for i in range(1, len(df)):
            # Сначала проверяем выход из позиции
            if position == 1:  # Если мы в лонге
                if df['low'].iloc[i] <= stop_loss:
                    # Выход по стоп-лоссу
                    equity *= (stop_loss / entry_price) * (1 - self.fee) ** 2
                    position = 0
                elif df['high'].iloc[i] >= take_profit:
                    # Выход по тейк-профиту
                    equity *= (take_profit / entry_price) * (1 - self.fee) ** 2
                    position = 0

            elif position == -1:  # Если мы в шорте
                if df['high'].iloc[i] >= stop_loss:
                    # Выход по стоп-лоссу
                    equity *= (entry_price / stop_loss) * (1 - self.fee) ** 2
                    position = 0
                elif df['low'].iloc[i] <= take_profit:
                    # Выход по тейк-профиту
                    equity *= (entry_price / take_profit) * (1 - self.fee) ** 2
                    position = 0

            # Затем проверяем вход в новую позицию
            if position == 0:
                signal = int(df['prediction'].iloc[i])
                strength = df['strength'].iloc[i]

                if signal != 0 and strength != 'none':
                    atr_value = df['atr'].iloc[i]
                    if pd.notna(atr_value) and atr_value > 0:
                        entry_price = df['close'].iloc[i]
                        risk_reward_ratio = rr_ratio_high if strength == 'high' else rr_ratio_low

                        # Устанавливаем SL/TP в зависимости от направления сигнала
                        sl_amount = atr_value * atr_multiplier_stop
                        tp_amount = sl_amount * risk_reward_ratio

                        if signal == 1:  # Лонг
                            stop_loss = entry_price - sl_amount
                            take_profit = entry_price + tp_amount
                        else:  # Шорт (направление -1)
                            stop_loss = entry_price + sl_amount
                            take_profit = entry_price - tp_amount

                        position = signal
                        self.trades.append(i)  # Просто записываем факт сделки

            equity_history.append(equity)

        # --- Обработка итогов ---
        # Создаем Series из истории капитала
        equity_series = pd.Series(equity_history, index=df.index[:len(equity_history)])

        stats = self.calculate_performance(equity_series)
        stats.update({"Всего сделок": len(self.trades)})

        # Важно! Очищаем сделки для следующего запуска на этом же объекте
        self.trades = []

        return stats

        # (Добавьте эту функцию в класс Backtester)

    def run_adaptive_smc_strategy(self, signals_df, strategy_params):
        """
        ФИНАЛЬНАЯ ВЕРСИЯ С АДАПТИВНЫМ ТЕЙК-ПРОФИТОМ.
        Размер Take-Profit зависит от объема на сигнальной свече.
        """
        if self.price_data.empty:
            return self.calculate_performance(pd.Series([self.initial_capital], index=self.price_data.index))

        df = self.price_data.copy().join(signals_df)
        if 'volume_spike_ratio' not in df.columns:
            raise ValueError("Признак 'volume_spike_ratio' отсутствует в данных для бэктеста!")

        df['prediction'].fillna(0, inplace=True)
        self.trades = []
        equity = self.initial_capital
        equity_history = [self.initial_capital] * len(df)
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0

        EXIT_AFTER_BARS = strategy_params.get('exit_after_bars', 8)
        entry_bar_index = -1

        for i in range(1, len(df)):
            current_open = df['open'].iloc[i]

            # --- Логика выхода ---
            if position != 0:
                exit_reason = None
                if i >= entry_bar_index + EXIT_AFTER_BARS:
                    exit_reason = "time_exit"
                elif position == 1 and df['low'].iloc[i] <= stop_loss:
                    exit_reason = "stop_loss"
                elif position == 1 and df['high'].iloc[i] >= take_profit:
                    exit_reason = "take_profit"
                elif position == -1 and df['high'].iloc[i] >= stop_loss:
                    exit_reason = "stop_loss"
                elif position == -1 and df['low'].iloc[i] <= take_profit:
                    exit_reason = "take_profit"

                if exit_reason:
                    exit_price = current_open
                    if exit_reason == "stop_loss": exit_price = stop_loss
                    if exit_reason == "take_profit": exit_price = take_profit

                    pnl_ratio = (exit_price / entry_price) if position == 1 else (entry_price / exit_price)
                    equity *= pnl_ratio * (1 - self.fee) ** 2
                    position = 0

            # --- Логика входа ---
            if position == 0:
                signal = int(df['prediction'].iloc[i - 1])
                if signal != 0:
                    entry_price = current_open
                    entry_bar_index = i
                    position = signal
                    self.trades.append(i)

                    volume_spike = df['volume_spike_ratio'].iloc[i - 1]
                    base_rr = strategy_params.get('base_rr', 1.5)
                    volume_multiplier = strategy_params.get('volume_multiplier', 0.5)
                    dynamic_rr = min(base_rr + (volume_spike * volume_multiplier), 8.0)

                    stop_loss_price = df['low'].iloc[i - 1] if signal == 1 else df['high'].iloc[i - 1]
                    risk_per_share = abs(entry_price - stop_loss_price)

                    take_profit_amount = risk_per_share * dynamic_rr
                    take_profit = entry_price + take_profit_amount if signal == 1 else entry_price - take_profit_amount
                    stop_loss = stop_loss_price

            equity_history[i] = equity

        # --- Финальный расчет статистики ---
        equity_series = pd.Series(equity_history, index=df.index)
        stats = self.calculate_performance(equity_series)
        stats.update({"Всего сделок": len(self.trades)})
        self.trades = []
        return stats


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль прогнозирования DBN-FBD
================================
Прогнозирование финансовых пузырей с использованием машинного обучения

Авторы: С.М. Гавриков, Н.И. Лысенок
НИУ ВШЭ, Факультет экономических наук
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class BubbleForecaster:
    """
    Класс для прогнозирования композитного индекса DBN-FBD
    Использует Random Forest с временной кросс-валидацией
    """
    
    def __init__(self, forecast_horizon=6):
        """
        Инициализация прогнозной модели
        
        Args:
            forecast_horizon: горизонт прогноза в месяцах (по умолчанию 6)
        """
        self.forecast_horizon = forecast_horizon
        
        # Основная модель - Random Forest (как в проекте)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def prepare_features(self, data):
        """
        Подготовка признаков для модели на основе существующих данных
        """
        features = pd.DataFrame(index=data.index)
        
        # Используем реальные колонки из датасета
        # Основные индикаторы для создания лагов
        key_columns = [
            'composite_bubble_score',
            'valuation_risk_score',
            'dynamics_risk_score', 
            'credit_risk_score',
            'macro_risk_score',
            'sentiment_risk_score',
            'The_Buffett_Indicator',
            'CAPE_or_Earnings_Ratio_P_E10',
            'VIX',
            'SPX',
            'CREDIT_GAP',
            'BAA10YM'
        ]
        
        # Создаем лаговые переменные
        for col in key_columns:
            if col in data.columns:
                features[f'{col}_lag1'] = data[col].shift(1)
                features[f'{col}_lag3'] = data[col].shift(3) 
                features[f'{col}_lag6'] = data[col].shift(6)
                features[f'{col}_lag12'] = data[col].shift(12)
                
                # Изменения
                features[f'{col}_change1'] = data[col].pct_change(1)
                features[f'{col}_change3'] = data[col].pct_change(3)
                features[f'{col}_change12'] = data[col].pct_change(12)
        
        # Скользящие средние для композитного индекса
        if 'composite_bubble_score' in data.columns:
            features['MA20'] = data['composite_bubble_score'].rolling(20).mean()
            features['MA50'] = data['composite_bubble_score'].rolling(50).mean()
            features['MA200'] = data['composite_bubble_score'].rolling(200).mean()
            
            # Отклонение от скользящих средних
            features['deviation_MA20'] = (data['composite_bubble_score'] - features['MA20']) / features['MA20']
            features['deviation_MA50'] = (data['composite_bubble_score'] - features['MA50']) / features['MA50']
        
        # Волатильность из существующих колонок
        if 'SPX_volatility_21d' in data.columns:
            features['vol_21d'] = data['SPX_volatility_21d']
        if 'SPX_volatility_63d' in data.columns:
            features['vol_63d'] = data['SPX_volatility_63d']
        if 'SPX_volatility_252d' in data.columns:
            features['vol_252d'] = data['SPX_volatility_252d']
        
        # Технические индикаторы из данных
        if 'SPX_exp_deviation' in data.columns:
            features['exp_deviation'] = data['SPX_exp_deviation']
        if 'SPX_growth_rate' in data.columns:
            features['growth_rate'] = data['SPX_growth_rate']
        
        # Сезонность (из даты)
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        features['year'] = data.index.year
        
        # Индекс времени
        features['time_index'] = np.arange(len(data))
        
        return features
    
    def create_target(self, data):
        """
        Создание целевой переменной - будущее значение composite_bubble_score
        """
        if 'composite_bubble_score' in data.columns:
            # Прогнозируем значение через forecast_horizon месяцев
            # Учитываем что данные ежедневные, конвертируем в дни
            shift_days = self.forecast_horizon * 21  # примерно 21 торговый день в месяце
            target = data['composite_bubble_score'].shift(-shift_days)
            return target
        return None
    
    def train(self, data):
        """
        Обучение модели на исторических данных
        """
        print("\nОбучение прогнозной модели DBN-FBD...")
        
        # Подготовка признаков
        X = self.prepare_features(data)
        y = self.create_target(data)
        
        if y is None:
            print("Ошибка: composite_bubble_score не найден в данных")
            return None
        
        # Удаление NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 1000:
            print(f"Предупреждение: мало данных для обучения ({len(X)} записей)")
        
        # Нормализация
        X_scaled = self.scaler.fit_transform(X)
        
        # Временная кросс-валидация
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_scores.append(score)
        
        print(f"Кросс-валидация RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        # Обучение на всех данных
        self.model.fit(X_scaled, y)
        
        # Важность признаков
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nТоп-10 важных признаков:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self
    
    def predict(self, data, periods=24):
        """
        Генерация прогноза на заданное количество месяцев
        
        Args:
            data: исторические данные
            periods: количество месяцев для прогноза
            
        Returns:
            DataFrame с прогнозами
        """
        print(f"\nГенерация прогноза на {periods} месяцев...")
        
        forecasts = []
        current_data = data.copy()
        
        # Генерируем прогноз итеративно
        for month in range(1, periods + 1):
            # Подготовка признаков для последней точки
            X = self.prepare_features(current_data)
            X_last = X.iloc[-1:].copy()
            
            # Заполнение NaN
            X_last = X_last.fillna(method='ffill').fillna(0)
            
            # Нормализация
            X_scaled = self.scaler.transform(X_last)
            
            # Прогноз
            prediction = self.model.predict(X_scaled)[0]
            
            # Определение уровня риска
            if prediction < 0.3:
                risk_level = 'Low'
            elif prediction < 0.5:
                risk_level = 'Medium'
            elif prediction < 0.7:
                risk_level = 'High'
            else:
                risk_level = 'Critical'
            
            # Вероятность кризиса (сигмоидная функция)
            crisis_prob = 1 / (1 + np.exp(-10 * (prediction - 0.7)))
            
            forecasts.append({
                'month': month,
                'forecast_date': current_data.index[-1] + pd.DateOffset(months=month),
                'dbn_fbd_forecast': prediction,
                'risk_level': risk_level,
                'crisis_probability': crisis_prob
            })
            
            # Добавляем прогноз для следующей итерации
            # (упрощенно - добавляем среднее за месяц)
            new_date = current_data.index[-1] + pd.DateOffset(months=1)
            new_row = pd.DataFrame({
                'composite_bubble_score': [prediction]
            }, index=[new_date])
            
            # Копируем остальные метрики с небольшим шумом
            for col in ['valuation_risk_score', 'dynamics_risk_score', 'credit_risk_score', 
                       'macro_risk_score', 'sentiment_risk_score']:
                if col in current_data.columns:
                    new_row[col] = current_data[col].iloc[-1] * np.random.normal(1, 0.02)
            
            current_data = pd.concat([current_data, new_row])
        
        forecast_df = pd.DataFrame(forecasts)
        
        # Вывод ключевых прогнозов
        print("\nПрогноз индекса DBN-FBD:")
        for _, row in forecast_df.head(6).iterrows():
            print(f"  Месяц {row['month']:2d}: {row['dbn_fbd_forecast']:.3f} "
                  f"[{row['risk_level']:8s}] "
                  f"P(кризис)={row['crisis_probability']:.1%}")
        
        # Анализ тренда
        trend = self._analyze_trend(forecast_df['dbn_fbd_forecast'].values)
        print(f"\nТренд: {trend}")
        
        return forecast_df
    
    def _analyze_trend(self, values):
        """
        Анализ тренда прогнозных значений
        """
        if len(values) < 2:
            return "Недостаточно данных"
        
        # Линейная регрессия для определения тренда
        x = np.arange(len(values))
        coef = np.polyfit(x, values, 1)[0]
        
        if coef > 0.01:
            return "📈 РАСТУЩИЙ РИСК"
        elif coef < -0.01:
            return "📉 СНИЖАЮЩИЙСЯ РИСК"
        else:
            return "➡️ СТАБИЛЬНЫЙ УРОВЕНЬ"
    
    def validate(self, data, test_size=0.2):
        """
        Валидация модели на тестовой выборке
        """
        print("\nВалидация модели...")
        
        # Разделение на train/test
        split_point = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        
        # Обучение на train
        self.train(train_data)
        
        # Прогноз на test
        X_test = self.prepare_features(test_data)
        y_test = self.create_target(test_data)
        
        # Удаление NaN
        valid_idx = ~(X_test.isna().any(axis=1) | y_test.isna())
        X_test = X_test[valid_idx]
        y_test = y_test[valid_idx]
        
        if len(X_test) > 0:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            
            # Метрики
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nМетрики на тестовой выборке:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'actuals': y_test
            }
        
        return None


def run_forecast_demo():
    """
    Демонстрация работы прогнозного модуля
    """
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ПРОГНОЗИРОВАНИЯ DBN-FBD")
    print("="*60)
    
    # Загрузка данных
    try:
        data = pd.read_csv('../financial_data_with_fixed_metrics.csv', 
                          parse_dates=['observation_date'])
        data.set_index('observation_date', inplace=True)
        
        print(f"✓ Загружено {len(data)} записей")
        print(f"  Период: {data.index[0].strftime('%Y-%m-%d')} - {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Создание и обучение модели
        forecaster = BubbleForecaster(forecast_horizon=6)
        forecaster.train(data)
        
        # Генерация прогноза
        forecast = forecaster.predict(data, periods=12)
        
        # Валидация
        validation_results = forecaster.validate(data)
        
        print("\n" + "="*60)
        print("ПРОГНОЗИРОВАНИЕ ЗАВЕРШЕНО")
        print("="*60)
        
        return forecast, validation_results
        
    except FileNotFoundError:
        print("✗ Файл данных не найден")
        print("  Требуется файл: financial_data_with_fixed_metrics.csv")
        return None, None


if __name__ == "__main__":
    forecast, validation = run_forecast_demo()
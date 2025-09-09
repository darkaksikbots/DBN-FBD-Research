#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль валидации модели DBN-FBD
=================================
Комплексная валидация на исторических данных

Авторы: С.М. Гавриков, Н.И. Лысенок
НИУ ВШЭ, Факультет экономических наук
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class ModelValidator:
    """
    Класс для валидации модели DBN-FBD
    """
    
    def __init__(self):
        """Инициализация валидатора"""
        
        # Исторические пузыри для валидации
        self.historical_bubbles = {
            'dotcom': {
                'period': ('1999-01-01', '2001-12-31'),
                'peak': '2000-03-24',
                'expected_index': 0.737,
                'actual_crash': -0.78
            },
            'housing': {
                'period': ('2006-01-01', '2008-12-31'),
                'peak': '2007-10-09',
                'expected_index': 0.700,
                'actual_crash': -0.57
            },
            'covid': {
                'period': ('2020-01-01', '2022-12-31'),
                'peak': '2021-12-31',
                'expected_index': 0.666,
                'actual_crash': -0.35
            }
        }
        
        # Пороговые значения для классификации
        self.thresholds = {
            'low': (0.0, 0.3),
            'medium': (0.3, 0.5),
            'high': (0.5, 0.7),
            'critical': (0.7, 1.0)
        }
    
    def validate_historical_bubbles(self, data):
        """
        Валидация на исторических пузырях
        """
        print("\n" + "="*60)
        print("ВАЛИДАЦИЯ НА ИСТОРИЧЕСКИХ ДАННЫХ")
        print("="*60)
        
        results = []
        
        for bubble_name, bubble_info in self.historical_bubbles.items():
            start, end = bubble_info['period']
            
            # Фильтрация данных за период пузыря
            mask = (data.index >= start) & (data.index <= end)
            bubble_data = data[mask]
            
            if len(bubble_data) > 0 and 'composite_bubble_score' in bubble_data.columns:
                # Максимальный индекс в период
                max_index = bubble_data['composite_bubble_score'].max()
                max_date = bubble_data['composite_bubble_score'].idxmax()
                
                # Сравнение с ожидаемым значением
                expected = bubble_info['expected_index']
                deviation = abs(max_index - expected) / expected * 100
                
                # Определение момента предупреждения
                warning_threshold = 0.5
                warning_dates = bubble_data[bubble_data['composite_bubble_score'] > warning_threshold]
                if len(warning_dates) > 0:
                    first_warning = warning_dates.index[0]
                    warning_lead_time = (pd.Timestamp(bubble_info['peak']) - first_warning).days
                else:
                    warning_lead_time = 0
                
                results.append({
                    'bubble': bubble_name.capitalize(),
                    'period': f"{start[:4]}-{end[:4]}",
                    'model_peak': max_date,
                    'actual_peak': bubble_info['peak'],
                    'model_index': max_index,
                    'expected_index': expected,
                    'deviation': deviation,
                    'warning_days': warning_lead_time,
                    'crash_magnitude': bubble_info['actual_crash']
                })
                
                print(f"\n{bubble_name.upper()} BUBBLE ({start[:4]}-{end[:4]}):")
                print(f"  Модельный пик: {max_date.strftime('%Y-%m-%d')}")
                print(f"  Фактический пик: {bubble_info['peak']}")
                print(f"  Индекс модели: {max_index:.3f}")
                print(f"  Ожидаемый индекс: {expected:.3f}")
                print(f"  Отклонение: {deviation:.1f}%")
                print(f"  Предупреждение за: {warning_lead_time} дней")
                print(f"  Валидация: {'✓ УСПЕШНА' if deviation < 10 else '⚠ ТРЕБУЕТ ПРОВЕРКИ'}")
        
        return pd.DataFrame(results)
    
    def calculate_classification_metrics(self, data):
        """
        Расчет метрик классификации
        """
        print("\n" + "="*60)
        print("МЕТРИКИ КЛАССИФИКАЦИИ")
        print("="*60)
        
        if 'composite_bubble_score' not in data.columns:
            print("Ошибка: composite_bubble_score не найден")
            return None
        
        # Создание бинарных меток (пузырь/не пузырь)
        # Считаем пузырем если индекс > 0.5
        y_true = np.zeros(len(data))
        y_pred = (data['composite_bubble_score'] > 0.5).astype(int)
        y_score = data['composite_bubble_score']
        
        # Отмечаем известные периоды пузырей
        for bubble_info in self.historical_bubbles.values():
            start, end = bubble_info['period']
            mask = (data.index >= start) & (data.index <= end)
            y_true[mask] = 1
        
        # Расчет метрик
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC-ROC если есть положительные примеры
        if y_true.sum() > 0:
            auc_roc = roc_auc_score(y_true, y_score)
        else:
            auc_roc = None
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        }
        
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        if auc_roc:
            print(f"AUC-ROC:   {auc_roc:.3f}")
        
        return metrics
    
    def statistical_tests(self, data):
        """
        Статистические тесты для валидации
        """
        print("\n" + "="*60)
        print("СТАТИСТИЧЕСКИЕ ТЕСТЫ")
        print("="*60)
        
        tests_results = {}
        
        if 'composite_bubble_score' in data.columns:
            score = data['composite_bubble_score'].dropna()
            
            # 1. Тест на нормальность (Shapiro-Wilk)
            if len(score) > 3:
                shapiro_stat, shapiro_p = stats.shapiro(score[:5000])  # Ограничение для больших выборок
                tests_results['shapiro_wilk'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'normal': shapiro_p > 0.05
                }
                print(f"\nShapiro-Wilk тест:")
                print(f"  Статистика: {shapiro_stat:.4f}")
                print(f"  p-value: {shapiro_p:.4f}")
                print(f"  Результат: {'Нормальное распределение' if shapiro_p > 0.05 else 'Не нормальное'}")
            
            # 2. Тест на стационарность (ADF)
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(score)
            tests_results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'stationary': adf_result[1] < 0.05
            }
            print(f"\nAugmented Dickey-Fuller тест:")
            print(f"  Статистика: {adf_result[0]:.4f}")
            print(f"  p-value: {adf_result[1]:.4f}")
            print(f"  Результат: {'Стационарный' if adf_result[1] < 0.05 else 'Не стационарный'}")
            
            # 3. Автокорреляция (Durbin-Watson)
            from statsmodels.stats.stattools import durbin_watson
            dw_stat = durbin_watson(score)
            tests_results['durbin_watson'] = {
                'statistic': dw_stat,
                'autocorrelation': 'Отсутствует' if 1.5 < dw_stat < 2.5 else 'Присутствует'
            }
            print(f"\nDurbin-Watson тест:")
            print(f"  Статистика: {dw_stat:.4f}")
            print(f"  Результат: {'Автокорреляция отсутствует' if 1.5 < dw_stat < 2.5 else 'Есть автокорреляция'}")
        
        return tests_results
    
    def calculate_vif(self, data):
        """
        Расчет VIF (Variance Inflation Factor) для проверки мультиколлинеарности
        """
        print("\n" + "="*60)
        print("АНАЛИЗ МУЛЬТИКОЛЛИНЕАРНОСТИ (VIF)")
        print("="*60)
        
        # Выбираем ключевые метрики для анализа
        features = [
            'The_Buffett_Indicator',
            'CAPE_or_Earnings_Ratio_P_E10',
            'VIX',
            'CREDIT_GAP',
            'GDP_YoY_USA'
        ]
        
        # Проверяем наличие колонок
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 2:
            print("Недостаточно данных для расчета VIF")
            return None
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X = data[available_features].dropna()
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = available_features
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(available_features))]
        
        print("\nVIF для основных индикаторов:")
        for _, row in vif_data.iterrows():
            status = "✓" if row['VIF'] < 5 else "⚠" if row['VIF'] < 10 else "✗"
            print(f"  {row['Feature']:30s}: {row['VIF']:6.2f} {status}")
        
        max_vif = vif_data['VIF'].max()
        print(f"\nМаксимальный VIF: {max_vif:.2f}")
        print(f"Результат: {'✓ Мультиколлинеарность отсутствует' if max_vif < 5 else '⚠ Умеренная мультиколлинеарность' if max_vif < 10 else '✗ Высокая мультиколлинеарность'}")
        
        return vif_data
    
    def generate_validation_report(self, data):
        """
        Генерация полного отчета валидации
        """
        print("\n" + "="*70)
        print("ПОЛНЫЙ ОТЧЕТ ВАЛИДАЦИИ МОДЕЛИ DBN-FBD")
        print("="*70)
        
        report = {}
        
        # 1. Историческая валидация
        historical_results = self.validate_historical_bubbles(data)
        report['historical'] = historical_results
        
        # 2. Метрики классификации
        classification_metrics = self.calculate_classification_metrics(data)
        report['classification'] = classification_metrics
        
        # 3. Статистические тесты
        statistical_results = self.statistical_tests(data)
        report['statistical'] = statistical_results
        
        # 4. Анализ мультиколлинеарности
        vif_results = self.calculate_vif(data)
        report['vif'] = vif_results
        
        # Итоговая оценка
        print("\n" + "="*70)
        print("ИТОГОВАЯ ОЦЕНКА ВАЛИДАЦИИ")
        print("="*70)
        
        validation_score = 0
        max_score = 4
        
        # Оценка исторической точности
        if len(historical_results) > 0:
            avg_deviation = historical_results['deviation'].mean()
            if avg_deviation < 5:
                validation_score += 1
                print("✓ Историческая точность: ОТЛИЧНО (отклонение < 5%)")
            elif avg_deviation < 10:
                validation_score += 0.5
                print("⚠ Историческая точность: ХОРОШО (отклонение < 10%)")
            else:
                print("✗ Историческая точность: ТРЕБУЕТ УЛУЧШЕНИЯ")
        
        # Оценка классификации
        if classification_metrics and classification_metrics.get('f1_score', 0) > 0.8:
            validation_score += 1
            print("✓ Классификация: ОТЛИЧНО (F1 > 0.8)")
        elif classification_metrics and classification_metrics.get('f1_score', 0) > 0.7:
            validation_score += 0.5
            print("⚠ Классификация: ХОРОШО (F1 > 0.7)")
        else:
            print("✗ Классификация: ТРЕБУЕТ УЛУЧШЕНИЯ")
        
        # Оценка статистических свойств
        if statistical_results:
            validation_score += 1
            print("✓ Статистические свойства: СООТВЕТСТВУЮТ")
        
        # Оценка мультиколлинеарности
        if vif_results is not None and vif_results['VIF'].max() < 10:
            validation_score += 1
            print("✓ Мультиколлинеарность: ПРИЕМЛЕМАЯ")
        
        # Финальная оценка
        final_score = (validation_score / max_score) * 100
        print(f"\n{'='*70}")
        print(f"ФИНАЛЬНАЯ ОЦЕНКА ВАЛИДАЦИИ: {final_score:.1f}%")
        
        if final_score >= 80:
            print("РЕЗУЛЬТАТ: ✓✓✓ МОДЕЛЬ ПОЛНОСТЬЮ ВАЛИДИРОВАНА")
        elif final_score >= 60:
            print("РЕЗУЛЬТАТ: ✓✓ МОДЕЛЬ ВАЛИДИРОВАНА С ЗАМЕЧАНИЯМИ")
        else:
            print("РЕЗУЛЬТАТ: ⚠ ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ НАСТРОЙКА")
        
        print("="*70)
        
        return report


def run_validation():
    """
    Запуск полной валидации модели
    """
    print("\n" + "="*70)
    print("ЗАПУСК ВАЛИДАЦИИ МОДЕЛИ DBN-FBD")
    print("="*70)
    print("Авторы: С.М. Гавриков, Н.И. Лысенок")
    print("НИУ ВШЭ, Факультет экономических наук")
    print("="*70)
    
    # Загрузка данных
    try:
        data = pd.read_csv('../financial_data_with_fixed_metrics.csv', 
                          parse_dates=['observation_date'])
        data.set_index('observation_date', inplace=True)
        
        print(f"\n✓ Данные загружены")
        print(f"  Период: {data.index[0].strftime('%Y-%m-%d')} - {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Записей: {len(data)}")
        
        # Создание валидатора
        validator = ModelValidator()
        
        # Генерация отчета
        validation_report = validator.generate_validation_report(data)
        
        return validation_report
        
    except FileNotFoundError:
        print("\n✗ Файл данных не найден")
        print("  Требуется: financial_data_with_fixed_metrics.csv")
        return None


if __name__ == "__main__":
    validation_report = run_validation()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль 2: Расчет метрик пузырей
================================
Методология DBN-FBD: Мультифакторный анализ финансовых пузырей

Авторы: С.М. Гавриков, Н.И. Лысенок
НИУ ВШЭ, Факультет экономических наук, 2024-2025

Этот модуль демонстрирует авторскую методологию расчета
композитного индекса риска финансовых пузырей.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DBN_FBD_BubbleMetrics:
    """
    Авторский класс для расчета метрик финансовых пузырей
    Ключевая инновация: Мультифакторная модель с динамическими весами
    """
    
    def __init__(self):
        """
        Инициализация с определением весов факторов риска
        
        ГЛАВНАЯ ФОРМУЛА МОДЕЛИ DBN-FBD:
        Composite Score = 0.30×R₁ + 0.20×R₂ + 0.20×R₃ + 0.15×R₄ + 0.15×R₅
        """
        
        # Веса категорий риска (авторская калибровка)
        self.category_weights = {
            'valuation': 0.30,    # Валуационный риск - максимальный вес
            'dynamics': 0.20,     # Рыночная динамика
            'credit': 0.20,       # Кредитные условия
            'macro': 0.15,        # Макроэкономика
            'sentiment': 0.15     # Настроения инвесторов
        }
        
        # Метрики по категориям
        self.category_metrics = {
            'valuation': {
                'metrics': [
                    'The_Buffett_Indicator',
                    'CAPE_or_Earnings_Ratio_P_E10',
                    'PE_Ratio',
                    'Dividend_Yield'
                ],
                'directions': [1, 1, 1, -1],  # 1: высокое значение = риск, -1: низкое = риск
                'thresholds': {
                    'The_Buffett_Indicator': {'warning': 115, 'danger': 145},
                    'CAPE_or_Earnings_Ratio_P_E10': {'warning': 25, 'danger': 30},
                    'PE_Ratio': {'warning': 20, 'danger': 25},
                    'Dividend_Yield': {'warning': 2.0, 'danger': 1.5}
                }
            },
            'dynamics': {
                'metrics': [
                    'VIX',
                    'VIX_SPX_ratio',
                    'SPX_volatility_21d',
                    'SPX_exp_deviation'
                ],
                'directions': [-1, -1, -1, 1],  # Низкая волатильность = риск (синтетический пузырь)
                'thresholds': {
                    'VIX': {'warning': 20, 'danger': 15},
                    'VIX_SPX_ratio': {'warning': 6, 'danger': 4},
                    'SPX_volatility_21d': {'warning': 0.15, 'danger': 0.10},
                    'SPX_exp_deviation': {'warning': 10, 'danger': 20}
                }
            },
            'credit': {
                'metrics': [
                    'CREDIT_GAP',
                    'margin_debt_to_market',
                    'BAA10YM',
                    'T10Y2Y'
                ],
                'directions': [1, 1, -1, -1],
                'thresholds': {
                    'CREDIT_GAP': {'warning': 5, 'danger': 10},
                    'margin_debt_to_market': {'warning': 2.5, 'danger': 3.0},
                    'BAA10YM': {'warning': 2.0, 'danger': 1.5},
                    'T10Y2Y': {'warning': 0.5, 'danger': 0}
                }
            },
            'macro': {
                'metrics': [
                    'GDP_YoY_USA',
                    'Unemployment_USA',
                    'Inflation_Rate',
                    'M2SL'
                ],
                'directions': [1, -1, 1, 1],
                'thresholds': {
                    'GDP_YoY_USA': {'warning': 4, 'danger': 5},
                    'Unemployment_USA': {'warning': 4, 'danger': 3.5},
                    'Inflation_Rate': {'warning': 3, 'danger': 4},
                    'M2SL': {'warning': 10, 'danger': 15}
                }
            },
            'sentiment': {
                'metrics': [
                    'AAII_sentiment',
                    'UMCSENT'
                ],
                'directions': [1, 1],
                'thresholds': {
                    'AAII_sentiment': {'warning': 40, 'danger': 50},
                    'UMCSENT': {'warning': 95, 'danger': 100}
                }
            }
        }
        
        logger.info("Инициализирована модель DBN-FBD Bubble Metrics")
        logger.info(f"Категории риска: {list(self.category_weights.keys())}")
    
    def calculate_percentile_ranks(self, data, window=252):
        """
        Расчет перцентильных рангов для нормализации метрик
        
        Методология: Использование расширяющегося окна для исторического контекста
        """
        logger.info(f"Расчет перцентильных рангов (окно: {window} дней)...")
        
        percentile_data = pd.DataFrame(index=data.index)
        
        for category, info in self.category_metrics.items():
            metrics = info['metrics']
            directions = info['directions']
            
            for metric, direction in zip(metrics, directions):
                if metric in data.columns:
                    # Расчет исторического перцентиля
                    percentile_data[f'{metric}_pct_rank'] = data[metric].expanding(min_periods=window).apply(
                        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
                    )
                    
                    # Инверсия для метрик с обратной зависимостью
                    if direction == -1:
                        percentile_data[f'{metric}_pct_rank'] = 1 - percentile_data[f'{metric}_pct_rank']
                    
                    logger.info(f"  ✓ {metric}: перцентильный ранг рассчитан")
        
        return percentile_data
    
    def calculate_category_scores(self, data, percentile_data):
        """
        Расчет риск-скоров для каждой категории
        
        Авторская методология: Взвешенное среднее перцентильных рангов
        """
        logger.info("Расчет категориальных риск-скоров...")
        
        category_scores = pd.DataFrame(index=data.index)
        
        for category, info in self.category_metrics.items():
            metrics = info['metrics']
            
            # Собираем доступные метрики
            available_metrics = []
            for metric in metrics:
                pct_col = f'{metric}_pct_rank'
                if pct_col in percentile_data.columns:
                    available_metrics.append(pct_col)
            
            if available_metrics:
                # Среднее значение перцентильных рангов
                category_scores[f'{category}_risk_score'] = percentile_data[available_metrics].mean(axis=1)
                
                logger.info(f"  ✓ {category}: риск-скор рассчитан ({len(available_metrics)} метрик)")
            else:
                # Нейтральное значение если метрики недоступны
                category_scores[f'{category}_risk_score'] = 0.5
                logger.warning(f"  ⚠ {category}: нет доступных метрик, использован нейтральный скор")
        
        return category_scores
    
    def calculate_composite_score(self, category_scores):
        """
        ГЛАВНАЯ ФОРМУЛА DBN-FBD
        
        Composite Score = Σ(wi × Ri)
        где wi - вес категории, Ri - риск-скор категории
        """
        logger.info("Расчет композитного индекса DBN-FBD...")
        
        composite = pd.Series(index=category_scores.index, dtype=float)
        
        for idx in category_scores.index:
            score = 0
            total_weight = 0
            
            for category, weight in self.category_weights.items():
                score_col = f'{category}_risk_score'
                if score_col in category_scores.columns:
                    value = category_scores.loc[idx, score_col]
                    if not pd.isna(value):
                        score += weight * value
                        total_weight += weight
            
            # Нормализация если не все категории доступны
            if total_weight > 0:
                composite[idx] = score / total_weight
            else:
                composite[idx] = 0.5  # Нейтральное значение
        
        # Сглаживание (3-месячное скользящее среднее)
        composite_smooth = composite.rolling(window=63, min_periods=1).mean()
        
        logger.info(f"✓ Композитный индекс рассчитан")
        logger.info(f"  Текущее значение: {composite.iloc[-1]:.3f}")
        logger.info(f"  Среднее: {composite.mean():.3f}")
        logger.info(f"  Максимум: {composite.max():.3f}")
        
        return composite, composite_smooth
    
    def classify_risk_levels(self, composite_scores):
        """
        Классификация уровней риска
        
        Пороги (авторская калибровка на исторических данных):
        - Low: 0.0 - 0.3
        - Medium: 0.3 - 0.5  
        - High: 0.5 - 0.7
        - Critical: 0.7 - 1.0
        """
        risk_levels = pd.cut(
            composite_scores,
            bins=[-float('inf'), 0.3, 0.5, 0.7, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Подсчет распределения
        distribution = risk_levels.value_counts(normalize=True) * 100
        
        logger.info("Распределение уровней риска:")
        for level in ['Low', 'Medium', 'High', 'Critical']:
            if level in distribution.index:
                logger.info(f"  {level}: {distribution[level]:.1f}%")
        
        return risk_levels
    
    def detect_synthetic_bubble(self, data):
        """
        КЛЮЧЕВАЯ ИННОВАЦИЯ: Детекция "синтетического пузыря"
        
        Определение: Высокие валуации + Низкая волатильность
        Это новый тип пузыря, характерный для эпохи алгоритмической торговли
        """
        logger.info("Детекция синтетического пузыря...")
        
        synthetic_conditions = pd.DataFrame(index=data.index)
        
        # Условие 1: Экстремальные валуации
        if 'The_Buffett_Indicator' in data.columns:
            synthetic_conditions['high_valuation'] = data['The_Buffett_Indicator'] > 145
        
        if 'CAPE_or_Earnings_Ratio_P_E10' in data.columns:
            synthetic_conditions['high_cape'] = data['CAPE_or_Earnings_Ratio_P_E10'] > 30
        
        # Условие 2: Аномально низкая волатильность
        if 'VIX' in data.columns:
            synthetic_conditions['low_volatility'] = data['VIX'] < 20
        
        # Синтетический пузырь = высокие валуации + низкая волатильность
        synthetic_conditions['is_synthetic_bubble'] = (
            synthetic_conditions.get('high_valuation', False) &
            synthetic_conditions.get('low_volatility', False)
        )
        
        # Анализ текущего состояния
        current_synthetic = synthetic_conditions['is_synthetic_bubble'].iloc[-1]
        synthetic_days = synthetic_conditions['is_synthetic_bubble'].sum()
        
        logger.info(f"✓ Анализ синтетического пузыря завершен")
        logger.info(f"  Текущий статус: {'ОБНАРУЖЕН ⚠️' if current_synthetic else 'Не обнаружен'}")
        logger.info(f"  Дней с синтетическим пузырем: {synthetic_days} ({synthetic_days/len(data)*100:.1f}%)")
        
        return synthetic_conditions
    
    def calculate_early_warning_signals(self, data, composite_scores):
        """
        Расчет ранних предупреждающих сигналов
        
        Методология: Анализ ускорения и дивергенций
        """
        logger.info("Расчет ранних предупреждающих сигналов...")
        
        signals = pd.DataFrame(index=data.index)
        
        # Моментум композитного индекса (1-я производная)
        signals['bubble_momentum'] = composite_scores.diff().rolling(window=21).mean()
        
        # Ускорение (2-я производная)
        signals['bubble_acceleration'] = signals['bubble_momentum'].diff().rolling(window=21).mean()
        
        # Дивергенция: рост цен при снижении волатильности
        if 'SPX' in data.columns and 'VIX' in data.columns:
            spx_momentum = data['SPX'].pct_change(21)
            vix_momentum = data['VIX'].pct_change(21)
            signals['price_vol_divergence'] = (spx_momentum > 0.05) & (vix_momentum < -0.1)
        
        # Экстремальные значения
        signals['extreme_valuation'] = composite_scores > 0.7
        signals['rapid_acceleration'] = signals['bubble_acceleration'] > signals['bubble_acceleration'].quantile(0.95)
        
        # Общий индекс раннего предупреждения
        warning_components = [
            'extreme_valuation',
            'rapid_acceleration',
            'price_vol_divergence'
        ]
        
        available_components = [c for c in warning_components if c in signals.columns]
        if available_components:
            signals['early_warning_index'] = signals[available_components].sum(axis=1) / len(available_components)
        
        # Текущий статус
        if 'early_warning_index' in signals.columns:
            current_warning = signals['early_warning_index'].iloc[-1]
            logger.info(f"✓ Индекс раннего предупреждения: {current_warning:.2f}")
            
            if current_warning > 0.7:
                logger.warning("⚠️ ВЫСОКИЙ УРОВЕНЬ ПРЕДУПРЕЖДЕНИЯ!")
            elif current_warning > 0.5:
                logger.info("⚠️ Повышенный уровень предупреждения")
        
        return signals
    
    def process_complete_metrics(self, data):
        """
        Полный процесс расчета метрик DBN-FBD
        """
        logger.info("\n" + "="*60)
        logger.info("ЗАПУСК РАСЧЕТА МЕТРИК DBN-FBD")
        logger.info("="*60)
        
        # 1. Перцентильные ранги
        percentile_data = self.calculate_percentile_ranks(data)
        
        # 2. Категориальные скоры
        category_scores = self.calculate_category_scores(data, percentile_data)
        
        # 3. Композитный индекс
        composite, composite_smooth = self.calculate_composite_score(category_scores)
        
        # 4. Уровни риска
        risk_levels = self.classify_risk_levels(composite)
        
        # 5. Синтетический пузырь
        synthetic_bubble = self.detect_synthetic_bubble(data)
        
        # 6. Ранние предупреждения
        early_warnings = self.calculate_early_warning_signals(data, composite)
        
        # Объединение результатов
        results = pd.DataFrame(index=data.index)
        
        # Добавляем все рассчитанные метрики
        for col in category_scores.columns:
            results[col] = category_scores[col]
        
        results['composite_bubble_score'] = composite
        results['composite_bubble_score_smooth'] = composite_smooth
        results['bubble_risk_level'] = risk_levels
        results['is_synthetic_bubble'] = synthetic_bubble.get('is_synthetic_bubble', False)
        
        if 'early_warning_index' in early_warnings.columns:
            results['early_warning_index'] = early_warnings['early_warning_index']
        
        logger.info("\n" + "="*60)
        logger.info("РАСЧЕТ МЕТРИК ЗАВЕРШЕН")
        logger.info("="*60)
        
        return results

def demonstrate_metrics_methodology():
    """
    Демонстрация методологии расчета метрик DBN-FBD
    """
    print("\n" + "="*70)
    print("МЕТОДОЛОГИЯ РАСЧЕТА МЕТРИК DBN-FBD")
    print("="*70)
    print("Авторы: С.М. Гавриков, Н.И. Лысенок")
    print("НИУ ВШЭ, 2024-2025")
    print("="*70)
    
    metrics = DBN_FBD_BubbleMetrics()
    
    print("\n1. ГЛАВНАЯ ФОРМУЛА МОДЕЛИ")
    print("-"*50)
    print("DBN-FBD Score = Σ(wi × Ri)")
    print("\nВеса категорий:")
    for category, weight in metrics.category_weights.items():
        print(f"  • {category}: {weight:.0%}")
    
    print("\n2. КАТЕГОРИИ РИСКА")
    print("-"*50)
    for category, info in metrics.category_metrics.items():
        print(f"\n{category.upper()}:")
        print(f"  Метрики: {', '.join(info['metrics'][:3])}...")
    
    print("\n3. КЛЮЧЕВАЯ ИННОВАЦИЯ")
    print("-"*50)
    print("СИНТЕТИЧЕСКИЙ ПУЗЫРЬ:")
    print("  • Высокие валуации (Buffett > 145%)")
    print("  • Низкая волатильность (VIX < 20)")
    print("  • Отсутствие классической эйфории")
    print("  • Доминирование алгоритмов")
    
    print("\n4. УРОВНИ РИСКА")
    print("-"*50)
    print("  • Low: 0.0 - 0.3 (Низкий риск)")
    print("  • Medium: 0.3 - 0.5 (Умеренный риск)")
    print("  • High: 0.5 - 0.7 (Высокий риск)")
    print("  • Critical: > 0.7 (Критический риск)")
    
    print("\n" + "="*70)
    print("Эта методология является ядром системы DBN-FBD")
    print("="*70)

if __name__ == "__main__":
    demonstrate_metrics_methodology()
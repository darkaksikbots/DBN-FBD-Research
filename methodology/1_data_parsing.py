#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль 1: Парсинг и подготовка данных
=====================================
Методология DBN-FBD: Этап сбора и парсинга данных

Авторы: С.М. Гавриков, Н.И. Лысенок
НИУ ВШЭ, Факультет экономических наук, 2024-2025

Этот модуль демонстрирует авторскую методологию сбора и парсинга
финансовых данных из 15+ источников для системы DBN-FBD.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from bs4 import BeautifulSoup
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DBN_FBD_DataParser:
    """
    Авторский класс для парсинга финансовых данных
    Ручной сбор → Парсинг → Валидация → Объединение
    """
    
    def __init__(self):
        """Инициализация парсера с определением структуры данных"""
        
        # Определение 200+ индикаторов по категориям
        self.indicators = {
            'valuation': {
                'The_Buffett_Indicator': 'Рыночная капитализация / ВВП',
                'CAPE_or_Earnings_Ratio_P_E10': 'Циклически скорректированный P/E',
                'PE_Ratio': 'Отношение цены к прибыли',
                'Dividend_Yield': 'Дивидендная доходность',
                'Excess_CAPE_Yield': 'Избыточная доходность CAPE'
            },
            'market_dynamics': {
                'SPX': 'Индекс S&P 500',
                'VIX': 'Индекс волатильности',
                'SPX_volatility_21d': 'Волатильность 21 день',
                'SPX_volatility_63d': 'Волатильность 63 дня',
                'SPX_volatility_252d': 'Волатильность 252 дня',
                'VIX_SPX_ratio': 'Отношение VIX к S&P 500'
            },
            'credit': {
                'CREDIT_GAP': 'Кредитный разрыв',
                'margin_debt_to_market': 'Маржинальный долг / Рынок',
                'BAA10YM': 'Спред корпоративных облигаций',
                'FEDFUNDS': 'Ставка ФРС',
                'T10Y2Y': 'Кривая доходности'
            },
            'macro': {
                'GDP_YoY_USA': 'Рост ВВП год к году',
                'Unemployment_USA': 'Безработица',
                'Inflation_Rate': 'Инфляция',
                'Core_CPI_USA': 'Базовая инфляция',
                'M2SL': 'Денежная масса M2'
            },
            'sentiment': {
                'AAII_sentiment': 'Настроения AAII',
                'UMCSENT': 'Потребительские настроения Michigan'
            }
        }
        
        # Исторические периоды пузырей для маркировки
        self.bubble_periods = {
            'dotcom': ('1995-01-01', '2002-10-09'),
            'housing': ('2003-01-01', '2009-03-09'),
            'covid': ('2020-03-23', '2022-10-12'),
            'synthetic': ('2023-01-01', '2025-12-31')
        }
        
        logger.info(f"Инициализирован парсер DBN-FBD с {sum(len(v) for v in self.indicators.values())} индикаторами")
    
    def parse_shiller_data(self, excel_file_path):
        """
        Парсинг данных Robert Shiller (Yale University)
        Исторические данные с 1871 года
        
        Источник: http://www.econ.yale.edu/~shiller/data.htm
        Метод: Ручная загрузка Excel файла → Парсинг
        """
        logger.info("Парсинг данных Shiller...")
        
        try:
            # Чтение Excel файла (скачан вручную)
            df = pd.read_excel(excel_file_path, sheet_name='Data', skiprows=7)
            
            # Извлечение ключевых колонок
            shiller_data = pd.DataFrame({
                'Date': pd.to_datetime(df.iloc[:, 0], format='%Y.%m'),
                'S_P_Comp_P': df.iloc[:, 1],  # S&P Composite Price
                'Dividend_D': df.iloc[:, 2],  # Дивиденды
                'Earnings_E': df.iloc[:, 3],  # Прибыль
                'CPI': df.iloc[:, 4],          # Индекс потребительских цен
                'Rate_GS10': df.iloc[:, 5],   # 10-летние казначейские
                'Real_Price': df.iloc[:, 6],  # Реальная цена
                'Real_Dividend': df.iloc[:, 7], # Реальные дивиденды
                'Real_Earnings': df.iloc[:, 8], # Реальная прибыль
                'CAPE_or_Earnings_Ratio_P_E10': df.iloc[:, 9]  # CAPE Ratio
            })
            
            # Очистка данных
            shiller_data = shiller_data.dropna(subset=['Date'])
            shiller_data.set_index('Date', inplace=True)
            
            # Расчет производных метрик
            shiller_data['PE_Ratio'] = shiller_data['S_P_Comp_P'] / shiller_data['Earnings_E']
            shiller_data['Dividend_Yield'] = (shiller_data['Dividend_D'] / shiller_data['S_P_Comp_P']) * 100
            shiller_data['Excess_CAPE_Yield'] = (1 / shiller_data['CAPE_or_Earnings_Ratio_P_E10']) - (shiller_data['Rate_GS10'] / 100)
            
            logger.info(f"✓ Распарсено {len(shiller_data)} записей Shiller (с {shiller_data.index[0]} по {shiller_data.index[-1]})")
            
            return shiller_data
            
        except Exception as e:
            logger.error(f"Ошибка парсинга Shiller: {e}")
            return None
    
    def parse_fred_csv_files(self, csv_directory):
        """
        Парсинг CSV файлов из FRED (Federal Reserve Economic Data)
        
        Источник: https://fred.stlouisfed.org/
        Метод: Ручная выгрузка CSV → Парсинг → Объединение
        """
        logger.info("Парсинг данных FRED...")
        
        fred_data = pd.DataFrame()
        
        # Список индикаторов FRED
        fred_indicators = {
            'GDP.csv': 'GDP_YoY_USA',
            'UNRATE.csv': 'Unemployment_USA',
            'CPIAUCSL.csv': 'Inflation_Rate',
            'FEDFUNDS.csv': 'FEDFUNDS',
            'M2SL.csv': 'M2SL',
            'T10Y2Y.csv': 'T10Y2Y',
            'BAA10YM.csv': 'BAA10YM'
        }
        
        for filename, indicator_name in fred_indicators.items():
            filepath = os.path.join(csv_directory, filename)
            
            if os.path.exists(filepath):
                try:
                    # Парсинг CSV
                    df = pd.read_csv(filepath, parse_dates=['DATE'])
                    df.columns = ['Date', indicator_name]
                    df.set_index('Date', inplace=True)
                    
                    # Объединение данных
                    if fred_data.empty:
                        fred_data = df
                    else:
                        fred_data = fred_data.join(df, how='outer')
                    
                    logger.info(f"  ✓ {indicator_name}: {len(df)} записей")
                    
                except Exception as e:
                    logger.error(f"  ✗ Ошибка парсинга {filename}: {e}")
        
        # Расчет производных индикаторов
        if 'GDP_YoY_USA' in fred_data.columns:
            fred_data['GDP_YoY_USA'] = fred_data['GDP_YoY_USA'].pct_change(periods=4) * 100
        
        if 'CPIAUCSL' in fred_data.columns:
            fred_data['Inflation_Rate'] = fred_data['CPIAUCSL'].pct_change(periods=12) * 100
        
        logger.info(f"✓ Всего распарсено {len(fred_data.columns)} индикаторов FRED")
        
        return fred_data
    
    def parse_yahoo_html(self, html_files_directory):
        """
        Парсинг HTML страниц Yahoo Finance
        
        Источник: https://finance.yahoo.com/
        Метод: Сохранение HTML → BeautifulSoup парсинг
        """
        logger.info("Парсинг данных Yahoo Finance...")
        
        yahoo_data = pd.DataFrame()
        
        # Тикеры для парсинга
        tickers = {
            'SPX.html': 'SPX',
            'VIX.html': 'VIX',
            'NASDAQ.html': 'NASDAQ'
        }
        
        for filename, ticker in tickers.items():
            filepath = os.path.join(html_files_directory, filename)
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                    
                    # Поиск таблицы с историческими данными
                    table = soup.find('table', {'data-test': 'historical-prices'})
                    
                    if table:
                        # Парсинг таблицы
                        rows = table.find_all('tr')
                        data = []
                        
                        for row in rows[1:]:  # Пропускаем заголовок
                            cols = row.find_all('td')
                            if len(cols) >= 6:
                                date = cols[0].text
                                close = cols[4].text.replace(',', '')
                                volume = cols[6].text.replace(',', '') if len(cols) > 6 else '0'
                                
                                data.append({
                                    'Date': pd.to_datetime(date),
                                    ticker: float(close),
                                    f'{ticker}_Volume': float(volume)
                                })
                        
                        df = pd.DataFrame(data)
                        df.set_index('Date', inplace=True)
                        
                        # Объединение
                        if yahoo_data.empty:
                            yahoo_data = df
                        else:
                            yahoo_data = yahoo_data.join(df, how='outer')
                        
                        logger.info(f"  ✓ {ticker}: {len(df)} записей")
                    
                except Exception as e:
                    logger.error(f"  ✗ Ошибка парсинга {filename}: {e}")
        
        # Расчет производных метрик
        if 'SPX' in yahoo_data.columns:
            yahoo_data['SPX_log_return'] = np.log(yahoo_data['SPX'] / yahoo_data['SPX'].shift(1))
            yahoo_data['SPX_volatility_21d'] = yahoo_data['SPX_log_return'].rolling(21).std() * np.sqrt(252)
            yahoo_data['SPX_volatility_63d'] = yahoo_data['SPX_log_return'].rolling(63).std() * np.sqrt(252)
            yahoo_data['SPX_volatility_252d'] = yahoo_data['SPX_log_return'].rolling(252).std() * np.sqrt(252)
        
        if 'VIX' in yahoo_data.columns and 'SPX' in yahoo_data.columns:
            yahoo_data['VIX_SPX_ratio'] = yahoo_data['VIX'] / yahoo_data['SPX'] * 100
        
        logger.info(f"✓ Всего распарсено {len(yahoo_data.columns)} индикаторов Yahoo")
        
        return yahoo_data
    
    def calculate_buffett_indicator(self, market_cap_data, gdp_data):
        """
        Расчет индикатора Баффетта
        
        Формула: (Рыночная капитализация / ВВП) × 100
        
        Интерпретация:
        < 75%: Недооценен
        75-90%: Справедливо оценен  
        90-115%: Умеренно переоценен
        115-145%: Существенно переоценен
        > 145%: Экстремально переоценен (пузырь)
        """
        buffett_indicator = (market_cap_data / gdp_data) * 100
        
        logger.info(f"✓ Рассчитан индикатор Баффетта. Текущее значение: {buffett_indicator.iloc[-1]:.1f}%")
        
        return buffett_indicator
    
    def mark_bubble_periods(self, data):
        """
        Маркировка исторических периодов пузырей
        """
        data['historical_bubble'] = 0
        data['bubble_phase'] = 'Normal'
        
        for bubble_name, (start, end) in self.bubble_periods.items():
            mask = (data.index >= start) & (data.index <= end)
            data.loc[mask, 'historical_bubble'] = 1
            data.loc[mask, 'bubble_phase'] = bubble_name.capitalize()
        
        bubble_days = data['historical_bubble'].sum()
        logger.info(f"✓ Маркировано {bubble_days} дней пузырей из {len(data)} записей")
        
        return data
    
    def merge_all_data(self, shiller_data, fred_data, yahoo_data, other_data=None):
        """
        Объединение всех источников данных в единый датасет
        """
        logger.info("Объединение всех источников данных...")
        
        # Начинаем с Shiller как базы (самые длинные исторические данные)
        merged = shiller_data.copy()
        
        # Добавляем FRED
        if fred_data is not None:
            merged = merged.join(fred_data, how='outer')
        
        # Добавляем Yahoo
        if yahoo_data is not None:
            merged = merged.join(yahoo_data, how='outer')
        
        # Добавляем другие источники
        if other_data is not None:
            merged = merged.join(other_data, how='outer')
        
        # Сортировка по дате
        merged.sort_index(inplace=True)
        
        # Заполнение пропусков
        # Forward fill для большинства индикаторов
        merged.fillna(method='ffill', inplace=True)
        
        # Интерполяция для плавных переходов
        numerical_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numerical_cols] = merged[numerical_cols].interpolate(method='linear')
        
        # Маркировка пузырей
        merged = self.mark_bubble_periods(merged)
        
        logger.info(f"✓ Объединено в единый датасет: {len(merged)} записей, {len(merged.columns)} индикаторов")
        logger.info(f"  Период: {merged.index[0]} - {merged.index[-1]}")
        
        return merged
    
    def validate_data_quality(self, data):
        """
        Валидация качества данных
        """
        logger.info("Валидация качества данных...")
        
        quality_report = {
            'total_records': len(data),
            'total_indicators': len(data.columns),
            'date_range': f"{data.index[0]} to {data.index[-1]}",
            'missing_values': {},
            'outliers': {},
            'data_types': {}
        }
        
        # Проверка пропущенных значений
        for col in data.columns:
            missing_pct = (data[col].isna().sum() / len(data)) * 100
            if missing_pct > 0:
                quality_report['missing_values'][col] = f"{missing_pct:.2f}%"
        
        # Проверка выбросов (значения за пределами 3 сигм)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            mean = data[col].mean()
            std = data[col].std()
            outliers = ((data[col] < mean - 3*std) | (data[col] > mean + 3*std)).sum()
            if outliers > 0:
                quality_report['outliers'][col] = outliers
        
        logger.info(f"✓ Валидация завершена:")
        logger.info(f"  - Записей: {quality_report['total_records']}")
        logger.info(f"  - Индикаторов: {quality_report['total_indicators']}")
        logger.info(f"  - Колонок с пропусками: {len(quality_report['missing_values'])}")
        logger.info(f"  - Колонок с выбросами: {len(quality_report['outliers'])}")
        
        return quality_report

def demonstrate_parsing_methodology():
    """
    Демонстрация полной методологии парсинга данных DBN-FBD
    """
    print("\n" + "="*70)
    print("МЕТОДОЛОГИЯ ПАРСИНГА ДАННЫХ DBN-FBD")
    print("="*70)
    print("Авторы: С.М. Гавриков, Н.И. Лысенок")
    print("НИУ ВШЭ, 2024-2025")
    print("="*70)
    
    # Инициализация парсера
    parser = DBN_FBD_DataParser()
    
    print("\n1. СТРУКТУРА ДАННЫХ")
    print("-"*50)
    for category, indicators in parser.indicators.items():
        print(f"\n{category.upper()} ({len(indicators)} индикаторов):")
        for key, description in list(indicators.items())[:3]:  # Показываем первые 3
            print(f"  • {key}: {description}")
    
    print("\n2. ИСТОЧНИКИ ДАННЫХ")
    print("-"*50)
    print("• Robert Shiller (Yale): Исторические данные с 1871")
    print("• FRED: Макроэкономические индикаторы США")
    print("• Yahoo Finance: Рыночные данные в реальном времени")
    print("• AAII: Настроения инвесторов")
    print("• BIS: Кредитные разрывы")
    
    print("\n3. ПРОЦЕСС ПАРСИНГА")
    print("-"*50)
    print("Шаг 1: Ручная загрузка данных из источников")
    print("Шаг 2: Парсинг с помощью специализированных функций")
    print("Шаг 3: Очистка и валидация данных")
    print("Шаг 4: Расчет производных метрик")
    print("Шаг 5: Объединение в единый датасет")
    print("Шаг 6: Маркировка исторических пузырей")
    print("Шаг 7: Финальная валидация качества")
    
    print("\n" + "="*70)
    print("Эта методология является основой системы DBN-FBD")
    print("="*70)

if __name__ == "__main__":
    demonstrate_parsing_methodology()
# Методология сбора и обработки данных

## Процесс формирования базы данных

Данные для модели DBN-FBD были собраны **вручную** из публичных источников и обработаны авторским парсингом. База данных создана **единократно** для исследования.

---

## Основные источники данных

### 1. Robert Shiller (Yale University)
**Файл:** `ie_data.xls`  
**Индикаторы из датасета:**
- `CAPE_or_Earnings_Ratio_P_E10` - Shiller P/E Ratio
- `S_P_Comp_P` - S&P Composite Index  
- `Dividend_D`, `Earnings_E` - Дивиденды и прибыль
- `CPI` - Consumer Price Index
- `Rate_GS10` - 10-year Treasury Rate

### 2. FRED (Federal Reserve Economic Data)
**Индикаторы:**
- `GDP_YoY_USA`, `GDP_QoQ_USA` - Рост ВВП
- `Unemployment_USA` - Уровень безработицы
- `Core_CPI_USA` - Базовая инфляция
- `M2SL` - Денежная масса M2
- `FEDFUNDS` - Ставка ФРС
- `BAA10YM` - Корпоративный спред
- `T10Y2Y` - Спред кривой доходности
- `NFCI` - Chicago Fed Financial Conditions Index
- `TOTALSL` - Total Consumer Credit

### 3. Yahoo Finance  
**Индикаторы:**
- `SPX` - S&P 500 Index
- `VIX` - CBOE Volatility Index
- `NASDAQ` - NASDAQ Composite

### 4. FINRA (Margin Statistics)
**Индикаторы:**
- `debit_balances_in_customers_securities_margin_acco` - Маржинальный долг
- `free_credit_balances_in_customers_cash_accounts` - Свободные средства на счетах
- `free_credit_balances_in_customers_securities_margi` - Свободные маржинальные средства

### 5. Дополнительные источники
- `AAII_sentiment` - AAII Bull/Bear Sentiment
- `UMCSENT` - University of Michigan Consumer Sentiment
- `wilshire_index` - Wilshire 5000 Total Market Index
- `The_Buffett_Indicator` - Рассчитан как Wilshire 5000 / GDP

---

## Структура финального датасета

### Файл: `financial_data_with_fixed_metrics.csv`

**Параметры:**
- **Период:** 1990-01-01 по 2024-12-31
- **Частота:** Ежедневная (daily)
- **Количество строк:** ~12,500
- **Количество колонок:** 218

### Ключевые колонки датасета:

#### Базовые рыночные данные:
- `observation_date` - Дата наблюдения
- `SPX`, `NASDAQ` - Основные индексы
- `wilshire_index`, `close_wilshire` - Wilshire 5000

#### Валуационные метрики (используются в модели):
- `The_Buffett_Indicator` - Market Cap/GDP
- `CAPE_or_Earnings_Ratio_P_E10` - CAPE Ratio
- `PE_Ratio` - P/E Ratio
- `Dividend_Yield` - Дивидендная доходность
- `Excess_CAPE_Yield` - Избыточная доходность CAPE

#### Показатели риска и волатильности:
- `VIX` - Индекс страха
- `Volatility` - Реализованная волатильность
- `VIX_SPX_ratio` - Отношение VIX/SPX
- `SPX_volatility_21d`, `SPX_volatility_63d`, `SPX_volatility_252d` - Волатильность на разных горизонтах

#### Кредитные условия:
- `CREDIT_GAP` - Credit-to-GDP Gap
- `margin_debt_to_market` - Отношение маржинального долга к рынку (не прямая колонка, рассчитывается)
- `BAA10YM` - Корпоративный спред
- `MDOAH` - Mortgage Debt Outstanding

#### Макроэкономика:
- `GDP_YoY_USA`, `GDP_QoQ_USA` - Рост ВВП
- `Unemployment_USA` - Безработица
- `Inflation_Rate` - Инфляция (рассчитана из CPI)
- `Core_CPI_USA` - Базовая инфляция
- `Gov_Debt_GDP_USA` - Госдолг к ВВП

#### Настроения инвесторов:
- `AAII_sentiment` - AAII Bull-Bear индикатор
- `UMCSENT` - Michigan Consumer Sentiment

---

## Рассчитанные метрики в датасете

### Нормализованные значения (_norm):
Для каждого основного индикатора рассчитаны нормализованные значения с суффиксом `_norm`

### Z-scores (_zscore):
Стандартизированные значения для ключевых метрик:
- `The_Buffett_Indicator_zscore`
- `CAPE_or_Earnings_Ratio_P_E10_zscore`
- `VIX_zscore`
- `CREDIT_GAP_zscore`

### Процентильные ранги (_pct_rank):
Исторические процентили для индикаторов:
- `The_Buffett_Indicator_pct_rank`
- `CAPE_or_Earnings_Ratio_P_E10_pct_rank`
- `SPX_exp_deviation_pct_rank`

### Композитные риск-скоры (основа модели DBN-FBD):
- `valuation_risk_score` - Валуационный риск (вес 30%)
- `dynamics_risk_score` - Динамический риск (вес 20%)
- `credit_risk_score` - Кредитный риск (вес 20%)
- `macro_risk_score` - Макроэкономический риск (вес 15%)
- `sentiment_risk_score` - Риск настроений (вес 15%)

### Итоговые индикаторы:
- `composite_bubble_score` - Композитный индекс DBN-FBD
- `bubble_risk_level` - Уровень риска (Low/Medium/High/Critical)
- `early_warning_index` - Индекс раннего предупреждения
- `market_regime` - Рыночный режим (кластеризация)

### Сигналы предупреждения:
- `total_warnings` - Количество предупреждений
- `total_dangers` - Количество опасных сигналов
- Индивидуальные флаги `_warning` и `_danger` для каждой метрики

---

## Обработка в коде

### Загрузка и подготовка (`1_data_preparation.py`):
```python
def load_data(self, file_path='financial_data_with_fixed_metrics.csv'):
    data = pd.read_csv(file_path, parse_dates=['observation_date'])
    data.set_index('observation_date', inplace=True)
    return data
```

### Расчет метрик (`2_bubble_metrics.py`):
```python
# Веса категорий в композитном индексе
self.metric_categories = {
    'valuation': {'weight': 0.30, 'metrics': ['The_Buffett_Indicator', 'CAPE_or_Earnings_Ratio_P_E10', 'PE_Ratio', 'Dividend_Yield']},
    'dynamics': {'weight': 0.20, 'metrics': ['SPX_exp_deviation', 'VIX', 'VIX_SPX_ratio', 'SPX_volatility_21d']},
    'credit': {'weight': 0.20, 'metrics': ['margin_debt_to_market', 'CREDIT_GAP', 'BAA10YM']},
    'macro': {'weight': 0.15, 'metrics': ['GDP_YoY_USA', 'Unemployment_USA', 'Inflation_Rate', 'T10Y2Y']},
    'sentiment': {'weight': 0.15, 'metrics': ['AAII_sentiment', 'UMCSENT']}
}
```

### Прогнозирование (`3_bubble_forecast.py`):
```python
# Random Forest для прогноза композитного индекса
self.model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```

---

## Валидация на исторических данных

Модель проверена на трех крупных пузырях:
1. **Dot-com (2000):** Индекс DBN-FBD = 0.737
2. **Housing (2007):** Индекс DBN-FBD = 0.700  
3. **COVID (2021):** Индекс DBN-FBD = 0.666

---

*Методология разработана С.М. Гавриковым и Н.И. Лысенок (НИУ ВШЭ) для исследования синтетических финансовых пузырей.*
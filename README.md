# Predicting Next-Day S&P 500 Stock Returns with Linear Regression

Course project for **CSDA5320 (Python)**. The project tests whether information available at the close of a trading day (recent returns, price trend, volatility and trading volume) can explain and predict the next day's return of S&P 500 stocks.

**Author:** [Your Name]  
**Student ID:** [Your Student ID]

## Research Question and Hypotheses

- **H0:** The features have no linear relationship with next-day return (all slope coefficients = 0), and the model predicts no better than the baseline.
- **H1:** At least one feature is related to next-day return, and the model predicts better than the baseline.

## Dataset

[S&P 500 Historical Data](https://www.kaggle.com/datasets/jacksaleeby/s-and-p500-historical-data) from Kaggle (daily open, high, low, close, volume and ticker).

| Item | Value |
|---|---|
| Usable observations | 2,693,941 stock-days |
| Companies | 472 |
| Date range | 2000-01-18 to 2026-02-19 |
| Training set | 1,784,203 rows (2000-01-18 to 2018-04-18) |
| Test set | 909,293 rows (2018-04-20 to 2026-02-19) |

The data are downloaded automatically with `kagglehub` and are not stored in this repository.

## Methods

1. **Cleaning:** lowercase columns, datetime conversion, duplicate and invalid-value removal.
2. **Feature engineering (per ticker):** daily return, return lag 1, return lag 5, daily range, volume change, volume / 5-day average volume, 5-day momentum, 10-day volatility. The target is the next day's return.
3. **EDA:** descriptive statistics, distributions, correlation heatmap, price trends, boxplots, and a market-level time series.
4. **Chronological split:** earliest 70% of dates for training and latest 30% for testing, with one date removed between them. Features are capped at the 1st and 99th training percentiles.
5. **Regression:** simple and multiple linear regression (OLS, `statsmodels`) with a variance inflation factor check.
6. **Inference:** t-tests, F-test, and a Wald test with standard errors clustered by date (stocks traded on the same day are correlated).
7. **Out-of-sample evaluation:** RMSE and R² against a training-mean baseline, direction accuracy against a majority-class baseline, a binomial test, date-level paired t-tests, and a top-half versus bottom-half prediction spread.

## Key Results

| Measure | Result |
|---|---|
| Overall F-test (training) | F = 321.1, p < 0.001 |
| Joint Wald test (clustered by date) | chi-square(8) = 19.93, p = 0.011 |
| R-squared (training) | 0.001438 (about 0.14%) |
| Individual coefficients, clustered p-values | none significant (all p > 0.11) |
| Test RMSE: model vs baseline | 0.022755 vs 0.022758 |
| Out-of-sample R-squared | 0.026% |
| Direction accuracy: model vs always "up" | 52.08% vs 52.22% |
| Binomial test (accuracy > baseline) | p = 0.995 |
| Top-half minus bottom-half spread | 0.00025 per day, p = 0.059 |

**Conclusion:** The features are jointly significant in the training data, so the in-sample null hypothesis is rejected. However, they explain only about 0.14% of the variation in next-day returns, and out of sample the model does not beat the baseline, so the null hypothesis of no predictive value cannot be rejected. The negative coefficients on recent returns and 5-day momentum suggest a faint short-term reversal effect, but it is far too small to be useful. This is consistent with the weak form of the efficient market hypothesis.

## Repository Structure

```
.
├── sp500_regression_project.py   # full analysis (run this first)
├── dashboard.py                  # Streamlit dashboard
├── README.md
├── results/                      # created by the analysis script (CSV tables, key_numbers.txt)
└── figures/                      # created by the analysis script (PNG charts)
```

## How to Run

**1. Install the requirements**

```bash
pip install numpy pandas seaborn matplotlib statsmodels scipy kagglehub streamlit
```

**2. Run the analysis** (downloads the data, runs the models, saves tables and figures)

```bash
python sp500_regression_project.py
```

This can take a few minutes because the dataset has about 2.7 million rows. All printed output is also saved to `results/key_numbers.txt`.

**3. Launch the dashboard** (from the same folder)

```bash
streamlit run dashboard.py
```

The dashboard has tabs for the overview, EDA, regression results, out-of-sample performance, hypothesis tests, and an interactive ticker explorer.

## Limitations

- Survivorship bias is possible because the dataset only contains companies in the S&P 500 index.
- Transaction costs and taxes are ignored.
- Linear regression cannot capture nonlinear relationships, and the residuals are heavy-tailed.
- Only price and volume features are used (no fundamentals, news or sentiment).
- The test period is a single historical period (April 2018 to February 2026).

## Tools

Python, pandas, NumPy, seaborn, matplotlib, statsmodels, SciPy, kagglehub, Streamlit.


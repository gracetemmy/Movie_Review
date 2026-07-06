# Gold Price Nowcasting from Cross-Asset Prices

## Overview

This project builds a regression model that estimates the daily value of the SPDR Gold ETF (GLD) from the same-day prices of four other major assets. It is framed as a **nowcast**: it uses day-*t* asset prices to explain day-*t* gold, estimating a contemporaneous cross-asset relationship rather than forecasting future prices. The central question is whether gold's daily movement can be explained by other assets, and if so, which ones actually matter.

The key result is that gold's daily returns are driven overwhelmingly by silver, that this relationship is linear and stable out-of-sample, and that modeling price levels fails entirely due to non-stationarity. These findings are established with a chronological (leakage-free) evaluation rather than a random split.

## Problem Statement

Given the daily closing prices of the S&P 500 (SPX), an oil ETF (USO), a silver ETF (SLV), and the EUR/USD exchange rate, estimate the same-day gold ETF price (GLD).

This is a nowcast, not a forecast. It does not predict future prices and is not a trading signal. Any reported R-squared reflects contemporaneous co-movement between assets, not predictive power.

## Dataset

| Property | Value |
|---|---|
| Source | Gold Price Data (Kaggle) |
| Samples | ~2,290 trading days |
| Period | 2008 to May 2018 |
| Features | 5 numeric cross-asset prices |

| Feature | Description |
|---|---|
| Date | Trading date |
| SPX | S&P 500 index value |
| USO | United States Oil Fund (ETF) price |
| SLV | iShares Silver Trust ETF price |
| EUR/USD | Euro / US Dollar exchange rate |
| GLD | SPDR Gold Trust ETF price (target) |

Engineered features include 1-day and 5-day trailing transforms for the four predictor assets, plus calendar variables extracted from the date. All rolling features are backward-looking only, and no feature is derived from the target (GLD).

## Methodology

The project follows a phased workflow designed to avoid the common failure modes of financial time-series modeling.

| Phase | Description |
|---|---|
| Framing | Defined the task as a nowcast; selected baselines and metrics before modeling |
| Data validation | Verified row counts, dtypes, duplicates, and calendar gaps |
| EDA | Examined each series, distributions, correlations, and stationarity |
| Feature engineering | Built trailing features leakage-free; handled NaNs and gap-affected rows |
| Validation design | Chronological train/test split with TimeSeriesSplit cross-validation |
| Modeling | Baselines, linear, regularized, and tree models, best-first |
| Evaluation | Predicted-vs-actual and residual diagnostics |
| Interpretation | Coefficient analysis and error analysis |

Two parallel tracks were modeled and compared. The **price-level** track predicts GLD in dollars; the **returns** track predicts GLD's daily return. Comparing the two tests the robustness of the relationship, since price levels are prone to non-stationarity while returns are more stable.

Data was split chronologically (train through 2016, test 2017 to May 2018) with no shuffling, because random splitting on time-ordered data leaks future information into training. Hyperparameter tuning used TimeSeriesSplit cross-validation on the training set only, and feature scaling was fit on training data alone.

## Results

Silver alone explains roughly 69 percent of gold's daily return variance out-of-sample. Adding all other features improves this by less than one percentage point.

Model comparison on the returns track (test R-squared):

| Model | Test R-squared | Notes |
|---|---|---|
| Gradient Boosting (tuned) | 0.696 | Matched linear; no gain from complexity |
| Full Linear Regression | 0.696 | Simple and best-in-class |
| Lasso (6 features) | 0.695 | Same fit, half the features |
| SLV-only baseline | 0.687 | One feature, nearly matches everything |
| Random Forest | 0.662 | Underperformed; no non-linear structure |
| Ridge | 0.612 | Stable but shrunk |

Price level versus returns (SLV-only baseline, test R-squared):

| Track | Test R-squared | Outcome |
|---|---|---|
| Returns | +0.687 | Stable and robust out-of-sample |
| Level | -9.74 | Collapsed due to regime drift |

The price-level model failed out-of-sample because the silver-gold price relationship drifted between the training and test periods: silver traded up to ~47 during training but sat in a 15 to 17 range during the test window. Modeling returns removed this drift and produced a stable relationship. Every model family converges on approximately 0.69, indicating the performance ceiling is set by available signal rather than model choice.

## Limitations

The model is explanatory, not predictive. It uses same-day inputs to explain same-day gold and does not forecast future values.

A single feature (silver) dominates, so the model offers limited insight into gold's broader economic drivers.

Asset relationships are non-stationary and drift over time, as demonstrated by the failure of the price-level track.

The dataset is small (~2,290 rows) and ends in May 2018, so results may not generalize to later periods or different market regimes.

The feature set is limited to four cross-asset prices and omits stronger economic drivers of gold such as real interest rates, inflation expectations, and the dollar index.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data handling | pandas, numpy |
| Modeling | scikit-learn |
| Statistics | statsmodels |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter Notebook |

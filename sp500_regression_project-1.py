"""
CSDA5320 Python Project
Can historical returns, price trend and trading volume predict next-day stock returns?

Methods used (all from the course): pandas, seaborn, time series, simple and multiple linear regression.

H0: The features have no linear relationship with next-day return (all slope coefficients = 0)
    and the model predicts no better than the baseline.
H1: At least one feature is related to next-day return and the model predicts better than the baseline.

"""

""" The figures and the results tables are saved in the figures/ and results/ folders. """

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import kagglehub

sns.set_theme(style="whitegrid")
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

lines = []         


def say(text=""):
    print(text)
    lines.append(str(text))


def savefig(name):
    plt.tight_layout()
    plt.savefig("figures/" + name, dpi=200)
    plt.close()



path = kagglehub.dataset_download("jacksaleeby/s-and-p500-historical-data")
df = pd.read_csv(os.path.join(path, "SP500_Historical_Data.csv"))
print(df.head())
df.info()

# cleaning data
df.columns = df.columns.str.lower()                                   # lowercase column names
df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)  # text -> datetime
df["date"] = df["date"].dt.normalize()

rows_before = len(df)
df = df.dropna(subset=["date", "open", "high", "low", "close", "volume", "ticker"])
df = df.drop_duplicates(subset=["ticker", "date"])
df = df[(df["close"] > 0) & (df["open"] > 0) & (df["volume"] > 0)]    # remove impossible values
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
say(f"Rows before cleaning: {rows_before:,}   after cleaning: {len(df):,}")


# transforming the data
by_ticker = df.groupby("ticker")

df["daily_return"] = by_ticker["close"].pct_change()                       # today's return
df["return_lag_1"] = by_ticker["daily_return"].shift(1)                    # yesterday's return
df["return_lag_5"] = by_ticker["daily_return"].shift(5)                    # return 5 days ago
df["daily_range"] = (df["high"] - df["low"]) / df["open"]                  # intraday range
df["volume_change"] = by_ticker["volume"].pct_change()                     # volume growth
df["volume_ma_5"] = by_ticker["volume"].transform(lambda x: x.rolling(5).mean())
df["volume_ma_ratio"] = df["volume"] / df["volume_ma_5"]                   # volume vs 5-day average
df["momentum_5d"] = by_ticker["close"].pct_change(periods=5)               # 5-day price trend
df["volatility_10d"] = by_ticker["daily_return"].transform(lambda x: x.rolling(10).std())

# Target = TOMORROW's return (continuous). Direction = 1 if tomorrow's return is positive.
df["next_day_return"] = by_ticker["daily_return"].shift(-1)

features = ["daily_return", "return_lag_1", "return_lag_5", "daily_range",
            "volume_change", "volume_ma_ratio", "momentum_5d", "volatility_10d"]
target = "next_day_return"

model_df = df[["ticker", "date"] + features + [target]].copy()
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
model_df["next_day_up"] = (model_df[target] > 0).astype(int)
model_df = model_df.sort_values(["date", "ticker"]).reset_index(drop=True)

say(f"Rows used for modelling: {len(model_df):,}")
say(f"Number of companies: {model_df['ticker'].nunique()}")
say(f"Date range: {model_df['date'].min().date()} to {model_df['date'].max().date()}")

#eda
say("\n--- DESCRIPTIVE STATISTICS ---")
desc = model_df[features + [target]].describe().T
desc["skew"] = model_df[features + [target]].skew()
say(desc.round(4).to_string())
desc.to_csv("results/descriptive_statistics.csv")

share_up = model_df["next_day_up"].mean()
say(f"\nShare of up-days (next_day_return > 0): {share_up:.4f}")

eda = model_df.sample(n=min(200_000, len(model_df)), random_state=42)   # sample = faster plots

# Figure 1: distribution of the target
plt.figure(figsize=(7, 4))
sns.histplot(eda[target].clip(-0.15, 0.15), bins=100)
plt.title("Distribution of Next-Day Return (clipped to +/-15% for display)")
plt.xlabel("Next-day return")
savefig("fig1_target_distribution.png")

# Figure 2: class balance
plt.figure(figsize=(5, 4))
sns.countplot(x="next_day_up", data=model_df)
plt.xticks([0, 1], ["Down / flat", "Up"])
plt.title("Next-Day Direction")
savefig("fig2_direction_counts.png")

# Figure 3: feature distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), features):
    low, high = eda[col].quantile([0.01, 0.99])
    sns.histplot(eda[col].clip(low, high), bins=50, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
savefig("fig3_feature_distributions.png")

# Figure 4: correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(model_df[features + [target]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
savefig("fig4_correlation_heatmap.png")

# Figure 5: price trend for a few well-known companies
example = [t for t in ["AAPL", "MSFT", "AMZN", "JPM"] if t in set(df["ticker"])]
if len(example) == 0:
    example = df["ticker"].unique()[:4].tolist()
plt.figure(figsize=(10, 5))
for t in example:
    d = df[df["ticker"] == t]
    plt.plot(d["date"], d["close"] / d["close"].iloc[0], label=t)
plt.yscale("log")
plt.legend()
plt.title("Closing Price Relative to First Day (log scale)")
savefig("fig5_price_trend.png")

# Figure 6: volume ratio vs next-day return
small = eda.sample(n=min(30_000, len(eda)), random_state=1)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=small, x="volume_ma_ratio", y=target, alpha=0.2)
plt.xlim(0, small["volume_ma_ratio"].quantile(0.99))
plt.ylim(-0.15, 0.15)
plt.title("Volume / 5-Day Average Volume vs Next-Day Return")
savefig("fig6_volume_vs_return.png")

# Figure 7: boxplots of the features for up days and down days
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), features):
    low, high = small[col].quantile([0.01, 0.99])
    sns.boxplot(x=small["next_day_up"], y=small[col].clip(low, high), ax=ax, showfliers=False)
    ax.set_title(col)
    ax.set_xlabel("Next-day up (1) / down (0)")
    ax.set_ylabel("")
savefig("fig7_features_by_direction.png")

# Time series view: equal-weighted "market" return across all companies
market = model_df.groupby("date")[["daily_return", target]].mean()
plt.figure(figsize=(10, 4))
sns.lineplot(data=market, x=market.index, y="daily_return")
plt.title("Average Daily Return Across All Companies (time series)")
savefig("fig8_market_return_timeseries.png")

# Simple regression on the market series: does today's market return predict tomorrow's?
ts_model = sm.OLS(market[target], sm.add_constant(market["daily_return"])).fit(
    cov_type="HAC", cov_kwds={"maxlags": 5})      # robust to autocorrelation in time series
say("\n--- TIME SERIES CHECK: market return tomorrow ~ market return today ---")
say(f"slope = {ts_model.params['daily_return']:.4f},  p-value = {ts_model.pvalues['daily_return']:.4g},  "
    f"R-squared = {ts_model.rsquared:.4f}")


#test train split
all_dates = np.sort(model_df["date"].unique())
cut = int(0.70 * len(all_dates))
train = model_df[model_df["date"] < all_dates[cut - 1]].copy()    # one day dropped as a gap
test = model_df[model_df["date"] >= all_dates[cut]].copy()
say(f"\nTrain: {train['date'].min().date()} to {train['date'].max().date()} ({len(train):,} rows)")
say(f"Test : {test['date'].min().date()} to {test['date'].max().date()} ({len(test):,} rows)")

# Extreme values: cap each feature at the 1st and 99th percentile of the TRAIN data
for col in features:
    low, high = train[col].quantile([0.01, 0.99])
    train[col] = train[col].clip(low, high)
    test[col] = test[col].clip(low, high)

# Multicollinearity check (VIF > 5-10 is a warning sign)
vif_sample = sm.add_constant(train[features].sample(n=min(100_000, len(train)), random_state=1))
vif = pd.DataFrame({"feature": features,
                    "VIF": [variance_inflation_factor(vif_sample.values, i + 1) for i in range(len(features))]})
say("\n--- VARIANCE INFLATION FACTORS ---")
say(vif.round(3).to_string(index=False))
vif.to_csv("results/vif.csv", index=False)

#regression analysis
y_train, y_test = train[target], test[target]
groups = pd.factorize(train["date"])[0]    # stocks on the same day are correlated -> cluster by date

# ---- Model 1: simple linear regression (one predictor: today's return) ----
X1 = sm.add_constant(train[["daily_return"]])
m1 = sm.OLS(y_train, X1).fit()
m1_cl = sm.OLS(y_train, X1).fit(cov_type="cluster", cov_kwds={"groups": groups})
say("\n=== MODEL 1: SIMPLE LINEAR REGRESSION ===")
say(f"slope = {m1.params['daily_return']:.5f}, t-test p-value = {m1.pvalues['daily_return']:.4g}, "
    f"clustered p-value = {m1_cl.pvalues['daily_return']:.4g}, R-squared = {m1.rsquared:.6f}")

# ---- Model 2: multiple linear regression (all 8 features) ----
X2 = sm.add_constant(train[features])
m2 = sm.OLS(y_train, X2).fit()
m2_cl = sm.OLS(y_train, X2).fit(cov_type="cluster", cov_kwds={"groups": groups})
say("\n=== MODEL 2: MULTIPLE LINEAR REGRESSION (standard output) ===")
say(m2.summary().as_text())

coef_table = pd.DataFrame({
    "coefficient": m2.params,
    "std_error": m2.bse,
    "p_value_standard": m2.pvalues,
    "p_value_clustered_by_date": m2_cl.pvalues})
coef_table.to_csv("results/regression_coefficients.csv")
say("\nCoefficient table (standard and date-clustered p-values):")
say(coef_table.round(5).to_string())

# Overall F-test: H0 all slopes = 0
say(f"\nOverall F-test: F = {m2.fvalue:.3f}, p-value = {m2.f_pvalue:.4g}")
say(f"R-squared = {m2.rsquared:.6f}, Adjusted R-squared = {m2.rsquared_adj:.6f}")

# Same joint test, but with date-clustered standard errors (more trustworthy)
wald = m2_cl.wald_test(np.eye(len(m2.params))[1:], scalar=True)
say(f"Joint Wald test with clustered SE: chi-square = {float(wald.statistic):.3f}, p-value = {float(wald.pvalue):.4g}")

# Regression diagnostics: residual plots
resid_sample = pd.DataFrame({"fitted": m2.fittedvalues, "resid": m2.resid}).sample(n=30_000, random_state=1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.scatterplot(data=resid_sample, x="fitted", y="resid", alpha=0.2, ax=axes[0])
axes[0].axhline(0, color="red")
axes[0].set_ylim(-0.15, 0.15)
axes[0].set_title("Residuals vs Fitted Values")
sns.histplot(resid_sample["resid"].clip(-0.15, 0.15), bins=80, ax=axes[1])
axes[1].set_title("Distribution of Residuals")
savefig("fig9_residual_diagnostics.png")

# Coefficient plot (95% confidence intervals, clustered)
ci = m2_cl.conf_int().drop("const")
ci.columns = ["low", "high"]
ci["coef"] = m2_cl.params.drop("const")
plt.figure(figsize=(8, 5))
plt.errorbar(ci["coef"], range(len(ci)), xerr=[ci["coef"] - ci["low"], ci["high"] - ci["coef"]], fmt="o")
plt.yticks(range(len(ci)), ci.index)
plt.axvline(0, color="red", linestyle="--")
plt.title("Regression Coefficients with 95% Confidence Intervals")
savefig("fig10_coefficients.png")


#OUT-OF-SAMPLE TEST (the real test of prediction ability)

test["pred"] = m2.predict(sm.add_constant(test[features]))
baseline_value = y_train.mean()                  # baseline: always predict the average return

rmse_model = np.sqrt(np.mean((y_test - test["pred"]) ** 2))
rmse_base = np.sqrt(np.mean((y_test - baseline_value) ** 2))
r2_oos = 1 - np.sum((y_test - test["pred"]) ** 2) / np.sum((y_test - baseline_value) ** 2)
say("\n=== OUT-OF-SAMPLE RESULTS (test period) ===")
say(f"RMSE model    = {rmse_model:.6f}")
say(f"RMSE baseline = {rmse_base:.6f}")
say(f"Out-of-sample R-squared = {r2_oos:.6f}   (positive = better than baseline)")

#date-level paired t-test on squared errors (each trading day is one observation)
test["err_model"] = (y_test - test["pred"]) ** 2
test["err_base"] = (y_test - baseline_value) ** 2
daily_err = test.groupby("date")[["err_model", "err_base"]].mean()
gain = daily_err["err_base"] - daily_err["err_model"]
t_stat, p_val = stats.ttest_1samp(gain, 0, alternative="greater")
say(f"Paired t-test on daily squared error (H1: model error < baseline error): t = {t_stat:.3f}, p-value = {p_val:.4g}")

#direction accuracy: predicted sign of the return vs the actual sign
test["pred_up"] = (test["pred"] > 0).astype(int)
base_class = int(train["next_day_up"].mean() >= 0.5)        # majority class of the TRAIN period
acc_model = (test["pred_up"] == test["next_day_up"]).mean()
acc_base = (base_class == test["next_day_up"]).mean()
say(f"\nDirection accuracy: model = {acc_model:.4f}, baseline (always predict {base_class}) = {acc_base:.4f}")
say(f"Share of days the model predicts 'up': {test['pred_up'].mean():.4f}")

n_test = len(test)
n_correct = int((test["pred_up"] == test["next_day_up"]).sum())
binom = stats.binomtest(n_correct, n_test, p=acc_base, alternative="greater")
say(f"Binomial test (H1: accuracy > baseline accuracy): p-value = {binom.pvalue:.4g}")

test["correct_model"] = (test["pred_up"] == test["next_day_up"]).astype(int)
test["correct_base"] = (base_class == test["next_day_up"]).astype(int)
daily_acc = test.groupby("date")[["correct_model", "correct_base"]].mean()
acc_gain = daily_acc["correct_model"] - daily_acc["correct_base"]
t2, p2 = stats.ttest_1samp(acc_gain, 0, alternative="greater")
say(f"Date-level paired t-test on accuracy gain: mean gain = {acc_gain.mean():.5f}, t = {t2:.3f}, p-value = {p2:.4g}")

#predicted vs actual: top half vs bottom half of predicted returns each day
test["pred_rank"] = test.groupby("date")["pred"].rank(pct=True)
high = test[test["pred_rank"] > 0.5].groupby("date")[target].mean()
low = test[test["pred_rank"] <= 0.5].groupby("date")[target].mean()
spread = (high - low).dropna()
t3, p3 = stats.ttest_1samp(spread, 0, alternative="greater")
say(f"\nAverage next-day return, top-half minus bottom-half of predictions: {spread.mean():.5f} per day")
say(f"One-sided t-test: t = {t3:.3f}, p-value = {p3:.4g}")

plt.figure(figsize=(6, 4))
sns.barplot(x=["Model", "Baseline"], y=[rmse_model, rmse_base])
plt.ylim(min(rmse_model, rmse_base) * 0.98, max(rmse_model, rmse_base) * 1.02)
plt.title("Test RMSE: Model vs Baseline")
savefig("fig11_rmse_comparison.png")


#summary table of all hypothesis tests

summary = pd.DataFrame({
    "test": ["Simple regression slope (t-test, clustered)",
             "Multiple regression overall F-test",
             "Joint Wald test, clustered by date",
             "Daily squared-error gain > 0 (paired t-test)",
             "Direction accuracy > baseline (binomial)",
             "Direction accuracy gain > 0 (date-level t-test)",
             "Top-half minus bottom-half return spread > 0"],
    "p_value": [m1_cl.pvalues["daily_return"], m2.f_pvalue, float(wald.pvalue),
                p_val, binom.pvalue, p2, p3]})
summary["decision_at_5_percent"] = np.where(summary["p_value"] < 0.05, "Reject H0", "Fail to reject H0")
say("\n=== SUMMARY OF HYPOTHESIS TESTS ===")
say(summary.round(5).to_string(index=False))
summary.to_csv("results/hypothesis_test_summary.csv", index=False)

with open("results/key_numbers.txt", "w") as f:
    f.write("\n".join(lines))
print("\nDone. Figures are in figures/, tables and key_numbers.txt are in results/")

"""
Dashboard for the S&P 500 next-day return project (Streamlit).

Setup:   pip install streamlit
Step 1:  run sp500_regression_project.py first (it creates the results/ and figures/ folders)
Step 2:  streamlit run dashboard.py      (run it from the same folder)
"""

import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="S&P 500 Next-Day Return Project", layout="wide")


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def load_table(name, index_col=None):
    """Read a CSV saved by the main script (returns None if the file is missing)."""
    path = os.path.join("results", name)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=index_col)
    return None


def show_figure(name, caption):
    """Show a PNG saved by the main script."""
    path = os.path.join("figures", name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.warning(f"{name} not found. Run sp500_regression_project.py first.")


@st.cache_data
def load_raw_data():
    """Load the Kaggle dataset (used only by the Ticker Explorer tab)."""
    import kagglehub
    path = kagglehub.dataset_download("jacksaleeby/s-and-p500-historical-data")
    data = pd.read_csv(os.path.join(path, "SP500_Historical_Data.csv"))
    data.columns = data.columns.str.lower()
    data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_convert(None)
    return data.sort_values(["ticker", "date"])


# ----------------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------------
st.title("Can Returns, Trends and Volume Predict Next-Day S&P 500 Stock Returns?")
st.caption("CSDA5320 Python Project | multiple linear regression on 2.69 million stock-day observations")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "EDA", "Regression", "Out-of-Sample", "Hypothesis Tests", "Ticker Explorer"])

# ----------------------------------------------------------------------------
# Tab 1: Overview
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Project summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", "2,693,941")
    c2.metric("Companies", "472")
    c3.metric("Training period", "2000 - Apr 2018")
    c4.metric("Test period", "Apr 2018 - Feb 2026")

    st.markdown("""
**Hypotheses**
- **H0:** the features have no linear relationship with next-day return and the model predicts no better than the baseline.
- **H1:** at least one feature is related to next-day return and the model predicts better than the baseline.

**Features:** daily return, return lag 1, return lag 5, daily range, volume change,
volume / 5-day average volume, 5-day momentum, 10-day volatility.

**Main finding:** the features are jointly significant in the training data (F-test p < 0.001,
clustered Wald p = 0.011) but explain only 0.14% of the variation, and out of sample the model
does **not** beat the baseline. H0 for prediction is not rejected, which is consistent with the
weak form of the efficient market hypothesis.
""")

# ----------------------------------------------------------------------------
# Tab 2: EDA
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Exploratory data analysis")
    desc = load_table("descriptive_statistics.csv", index_col=0)
    if desc is not None:
        st.write("Descriptive statistics")
        st.dataframe(desc.round(4), use_container_width=True)

    vif = load_table("vif.csv")
    if vif is not None:
        st.write("Variance inflation factors (all well below 5, so no multicollinearity problem)")
        st.bar_chart(vif.set_index("feature")["VIF"])

    choice = st.selectbox("Choose a figure", [
        "Target distribution", "Direction counts", "Feature distributions",
        "Correlation heatmap", "Price trend", "Volume vs return",
        "Features by direction", "Market return time series"])
    figures = {
        "Target distribution": "fig1_target_distribution.png",
        "Direction counts": "fig2_direction_counts.png",
        "Feature distributions": "fig3_feature_distributions.png",
        "Correlation heatmap": "fig4_correlation_heatmap.png",
        "Price trend": "fig5_price_trend.png",
        "Volume vs return": "fig6_volume_vs_return.png",
        "Features by direction": "fig7_features_by_direction.png",
        "Market return time series": "fig8_market_return_timeseries.png"}
    show_figure(figures[choice], choice)

# ----------------------------------------------------------------------------
# Tab 3: Regression
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Multiple linear regression (training data)")
    coef = load_table("regression_coefficients.csv", index_col=0)
    if coef is not None:
        coef.index.name = "variable"
        st.dataframe(coef.round(5), use_container_width=True)

        st.write("Coefficient sizes (all tiny)")
        st.bar_chart(coef.drop("const")["coefficient"])

        threshold = st.slider("Significance level", 0.01, 0.10, 0.05, 0.01)
        sig_standard = coef[coef["p_value_standard"] < threshold].index.tolist()
        sig_clustered = coef[coef["p_value_clustered_by_date"] < threshold].index.tolist()
        st.write(f"**Significant with standard errors:** {', '.join(sig_standard) or 'none'}")
        st.write(f"**Significant with date-clustered standard errors:** {', '.join(sig_clustered) or 'none'}")
        st.info("Standard p-values treat all stock-days as independent. Stocks move together on the "
                "same day, so the clustered p-values are the more trustworthy ones.")

    c1, c2, c3 = st.columns(3)
    c1.metric("R-squared", "0.001438")
    c2.metric("Adjusted R-squared", "0.001433")
    c3.metric("F-statistic", "321.1")
    show_figure("fig9_residual_diagnostics.png", "Residual diagnostics")
    show_figure("fig10_coefficients.png", "Coefficients with 95% confidence intervals")

# ----------------------------------------------------------------------------
# Tab 4: Out-of-sample
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Performance on the test period (2018-2026)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model RMSE", "0.022755", delta="-0.000003 vs baseline", delta_color="normal")
    c2.metric("Out-of-sample R-squared", "0.026%")
    c3.metric("Share predicted 'up'", "83.87%")

    c1, c2 = st.columns(2)
    c1.metric("Direction accuracy (model)", "52.08%")
    c2.metric("Direction accuracy (always up)", "52.22%")

    comparison = pd.DataFrame({
        "Measure": ["RMSE", "Direction accuracy"],
        "Model": [0.022755, 0.5208],
        "Baseline": [0.022758, 0.5222]})
    st.table(comparison)
    st.write("Top-half minus bottom-half predicted return spread: 0.00025 per day (p = 0.059)")
    show_figure("fig11_rmse_comparison.png", "Test RMSE: model vs baseline")

# ----------------------------------------------------------------------------
# Tab 5: Hypothesis tests
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("Summary of hypothesis tests")
    tests = load_table("hypothesis_test_summary.csv")
    if tests is not None:
        level = st.slider("Significance level", 0.01, 0.10, 0.05, 0.01, key="level_tests")
        tests["decision"] = tests["p_value"].apply(lambda p: "Reject H0" if p < level else "Fail to reject H0")
        st.dataframe(tests[["test", "p_value", "decision"]].round(5), use_container_width=True)
        st.bar_chart(tests.set_index("test")["p_value"])
        st.caption("Bars below the chosen significance level reject H0.")
    st.markdown("**Conclusion:** the in-sample relationship is statistically detectable but economically tiny; "
                "the model does not predict better than the baseline out of sample.")

# ----------------------------------------------------------------------------
# Tab 6: Ticker explorer (interactive, uses the raw data)
# ----------------------------------------------------------------------------
with tab6:
    st.subheader("Explore one company")
    if st.checkbox("Load the raw dataset (downloads from Kaggle on first use)"):
        raw = load_raw_data()
        ticker = st.selectbox("Ticker", sorted(raw["ticker"].unique()))
        stock = raw[raw["ticker"] == ticker].copy()
        start, end = st.date_input("Date range",
                                   value=(stock["date"].min().date(), stock["date"].max().date()))
        stock = stock[(stock["date"].dt.date >= start) & (stock["date"].dt.date <= end)]
        stock["daily_return"] = stock["close"].pct_change()

        c1, c2, c3 = st.columns(3)
        c1.metric("Trading days", f"{len(stock):,}")
        c2.metric("Average daily return", f"{stock['daily_return'].mean():.4%}")
        c3.metric("Daily volatility", f"{stock['daily_return'].std():.4%}")

        st.write("Closing price")
        st.line_chart(stock.set_index("date")["close"])
        st.write("Trading volume")
        st.bar_chart(stock.set_index("date")["volume"])
        st.write("Distribution of daily returns")
        st.bar_chart(pd.cut(stock["daily_return"].dropna(), bins=40).value_counts().sort_index()
                     .rename(lambda x: str(round(x.mid, 3))))
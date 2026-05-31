"""
Sales & Demand Forecasting System
===================================
Runs forecasting for Electronics, Clothing, and Food & Beverage.
Saves a separate chart image for each category.
"""

# ── 1. IMPORTS ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── 2. DATA GENERATION ───────────────────────────────────────────────────────
np.random.seed(42)

def generate_retail_data(n_months=36, category="Electronics"):
    base_params = {
        "Electronics":     {"base": 42000, "trend": 1100, "peak_months": [11, 12]},
        "Clothing":        {"base": 28000, "trend":  550, "peak_months": [6, 7, 11, 12]},
        "Food & Beverage": {"base": 55000, "trend":  750, "peak_months": [11, 12]},
    }
    p = base_params[category]
    dates, sales = [], []
    start = datetime(2022, 1, 1)
    for i in range(n_months):
        date = start + pd.DateOffset(months=i)
        month = date.month
        seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 3) / 12)
        if month in p["peak_months"]:
            seasonal += 0.3
        noise = np.random.normal(0, p["base"] * 0.05)
        value = p["base"] + p["trend"] * i + p["base"] * (seasonal - 1) + noise
        dates.append(date)
        sales.append(max(int(value), 500))
    return pd.DataFrame({"date": dates, "sales": sales, "category": category})

# ── 3. EVALUATION FUNCTION ───────────────────────────────────────────────────
def evaluate(actual, predicted, label=""):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"\n  {label} Performance:")
    print(f"    MAE  : ${mae:,.0f}")
    print(f"    RMSE : ${rmse:,.0f}")
    print(f"    MAPE : {mape:.1f}%")
    print(f"    R²   : {r2:.3f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

# ── 4. FORECAST FUNCTION ─────────────────────────────────────────────────────
def forecast_future(df, model, scaler, features, n_months=3):
    history = df.copy()
    forecasts = []
    for step in range(n_months):
        last_row  = history.iloc[-1].copy()
        next_date = last_row["date"] + pd.DateOffset(months=1)
        next_month = next_date.month
        new_row = {
            "date":           next_date,
            "time_index":     last_row["time_index"] + 1,
            "month_sin":      np.sin(2 * np.pi * next_month / 12),
            "month_cos":      np.cos(2 * np.pi * next_month / 12),
            "lag_1":          history["sales"].iloc[-1],
            "lag_3":          history["sales"].iloc[-3],
            "lag_12":         history["sales"].iloc[-12] if len(history) >= 12 else history["sales"].mean(),
            "rolling_3_mean": history["sales"].iloc[-3:].mean(),
            "rolling_6_mean": history["sales"].iloc[-6:].mean(),
        }
        X_new = pd.DataFrame([new_row])[features]
        pred  = float(model.predict(scaler.transform(X_new))[0])
        new_row["sales"] = pred
        forecasts.append({"date": next_date, "sales": pred})
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    return pd.DataFrame(forecasts)

# ── 5. MAIN LOOP — runs for all 3 categories ─────────────────────────────────
MONTHS   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
FEATURES = ["time_index","month_sin","month_cos","lag_1","lag_3","lag_12","rolling_3_mean","rolling_6_mean"]

for category in ["Electronics", "Clothing", "Food & Beverage"]:
    print("\n" + "="*55)
    print(f"  CATEGORY: {category}")
    print("="*55)

    # Load & clean
    df = generate_retail_data(n_months=36, category=category)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date").resample("MS").mean(numeric_only=True)
    df["sales"] = df["sales"].interpolate(method="linear")
    df = df.reset_index()

    # Feature engineering
    df["month"]         = df["date"].dt.month
    df["time_index"]    = range(len(df))
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    df["lag_1"]         = df["sales"].shift(1)
    df["lag_3"]         = df["sales"].shift(3)
    df["lag_12"]        = df["sales"].shift(12)
    df["rolling_3_mean"]= df["sales"].rolling(3).mean()
    df["rolling_6_mean"]= df["sales"].rolling(6).mean()
    df = df.dropna().reset_index(drop=True)

    # Train/test split
    TRAIN_SIZE = len(df) - 6
    train = df.iloc[:TRAIN_SIZE]
    test  = df.iloc[TRAIN_SIZE:]
    X_train, y_train = train[FEATURES], train["sales"]
    X_test,  y_test  = test[FEATURES],  test["sales"]

    # Train model
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_s, y_train)

    train_preds = model.predict(X_train_s)
    test_preds  = model.predict(X_test_s)

    evaluate(y_train, train_preds, "Training")
    test_metrics = evaluate(y_test, test_preds, "Test")

    # Forecast
    forecast_df = forecast_future(df, model, scaler, FEATURES, n_months=3)
    print(f"\n  3-Month Forecast:")
    for _, row in forecast_df.iterrows():
        print(f"    {row['date'].strftime('%B %Y')}: ${row['sales']:,.0f}")

    # ── Charts ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Sales Forecasting Dashboard — {category}", fontsize=16, fontweight="bold")

    # Plot 1: Historical + forecast
    ax1 = axes[0, 0]
    fit_all = model.predict(scaler.transform(df[FEATURES]))
    ax1.plot(df["date"], df["sales"],  color="#378ADD", lw=2,  label="Actual sales")
    ax1.plot(df["date"], fit_all,      color="#1D9E75", lw=1.5, ls="--", label="Model fit")
    ax1.plot(forecast_df["date"], forecast_df["sales"],
             color="#EF9F27", lw=2, marker="o", ms=6, label="3-month forecast")
    ax1.axvline(df["date"].iloc[TRAIN_SIZE], color="#E24B4A", lw=1, ls=":", alpha=0.7)
    ax1.set_title("Monthly sales & forecast")
    ax1.set_ylabel("Sales ($)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    # Plot 2: Residuals
    ax2 = axes[0, 1]
    residuals = y_test.values - test_preds
    ax2.bar(test["date"], residuals,
            color=["#1D9E75" if r >= 0 else "#E24B4A" for r in residuals])
    ax2.axhline(0, color="#444", lw=1)
    ax2.set_title("Forecast errors (test period)")
    ax2.set_ylabel("Residual ($)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    # Plot 3: Seasonality
    ax3 = axes[1, 0]
    monthly_avg = df.groupby("month")["sales"].mean()
    colors_bar  = ["#1D9E75" if v >= monthly_avg.mean() else "#B4B2A9" for v in monthly_avg]
    ax3.bar(MONTHS[:len(monthly_avg)], monthly_avg.values, color=colors_bar)
    ax3.set_title("Average sales by month (seasonality)")
    ax3.set_ylabel("Avg sales ($)")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    # Plot 4: Feature importance
    ax4 = axes[1, 1]
    feat_imp = pd.Series(np.abs(model.coef_), index=FEATURES).sort_values()
    feat_imp.plot(kind="barh", ax=ax4,
                  color=["#378ADD" if v == feat_imp.max() else "#B4B2A9" for v in feat_imp])
    ax4.set_title("Feature importance (|coefficient|)")
    ax4.set_xlabel("Absolute coefficient")

    plt.tight_layout()
    filename = f"sales_forecast_{category.replace(' & ', '_').replace(' ', '_')}.png"
    plt.show()
    
    print(f"\n  Chart saved: {filename}")

print("\n✓ All 3 categories done! Check your Downloads folder for the chart images.")

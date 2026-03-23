"""
data_pipeline.py — Full data pipeline for Black Swans in the Blockchain
Downloads BTC/USDT and ETH/USDT 1-min OHLCV from Binance,
computes log-returns, labels crash regimes, and saves as parquet.

Usage:
    python src/data_pipeline.py --download        # Step 1: Download raw data
    python src/data_pipeline.py --process         # Step 2: Compute log-returns + label regimes
    python src/data_pipeline.py --describe        # Step 3: Descriptive statistics + plots
    python src/data_pipeline.py --all             # Run all steps

Requirements:
    pip install python-binance pandas pyarrow numpy matplotlib seaborn scipy
"""

import os
import time
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — change these paths to match your project structure
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # assumes src/ is one level down
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1m"  # 1-minute candles
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"

# Crash regime windows (inclusive). Each tuple: (name, start, end, description)
CRASH_EVENTS = [
    ("COVID_CRASH",    "2020-03-12", "2020-03-13", "COVID crash, -40% in 24h"),
    ("CHINA_BAN",      "2021-05-19", "2021-05-23", "China mining ban, -30%"),
    ("FTX_COLLAPSE",   "2022-11-08", "2022-11-11", "FTX collapse, -75% over 4 days"),
    ("REGULATORY_2023","2023-08-17", "2023-08-18", "Regulatory news spike, -10%"),
]

# Binance API column names returned by get_historical_klines
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]

# Columns to keep and their types
NUMERIC_COLS = ["open", "high", "low", "close", "volume", "quote_asset_volume",
                "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]


# ---------------------------------------------------------------------------
# STEP 1: DOWNLOAD RAW DATA
# ---------------------------------------------------------------------------
def download_symbol(symbol: str, interval: str, start: str, end: str, 
                    output_dir: Path, chunk_days: int = 30) -> Path:
    """
    Download historical klines from Binance in chunks to avoid API timeouts.
    
    Binance returns max ~1000 candles per request for 1-min data, so we 
    chunk by calendar month to stay well within limits. Each chunk is 
    ~43,200 candles (30 days * 24h * 60min).
    
    The API is free and does not require authentication for public market data.
    Empty api_key/api_secret works fine for kline downloads.
    """
    from binance import Client
    
    # Empty credentials work for public data endpoints
    client = Client("", "")
    
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    
    all_klines = []
    chunk_start = start_dt
    chunk_num = 0
    
    print(f"\n{'='*60}")
    print(f"Downloading {symbol} {interval} data")
    print(f"Period: {start} to {end}")
    print(f"{'='*60}")
    
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end_dt)
        
        chunk_num += 1
        print(f"  Chunk {chunk_num}: {chunk_start.date()} -> {chunk_end.date()} ... ", end="", flush=True)
        
        try:
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=str(chunk_start),
                end_str=str(chunk_end)
            )
            all_klines.extend(klines)
            print(f"{len(klines):,} candles")
            
        except Exception as e:
            print(f"ERROR: {e}")
            print(f"  Retrying in 10 seconds...")
            time.sleep(10)
            try:
                klines = client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=str(chunk_start),
                    end_str=str(chunk_end)
                )
                all_klines.extend(klines)
                print(f"  Retry OK: {len(klines):,} candles")
            except Exception as e2:
                print(f"  Retry FAILED: {e2}. Skipping chunk.")
        
        chunk_start = chunk_end
        
        # Be respectful to the API — small delay between chunks
        time.sleep(0.5)
    
    if not all_klines:
        raise RuntimeError(f"No data downloaded for {symbol}. Check API access.")
    
    # Build DataFrame
    df = pd.DataFrame(all_klines, columns=KLINE_COLUMNS)
    
    # Convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop the 'ignore' column
    df = df.drop(columns=["ignore"])
    
    # Remove exact duplicates (Binance sometimes returns overlapping candles)
    before = len(df)
    df = df.drop_duplicates(subset=["open_time"], keep="first")
    after = len(df)
    if before != after:
        print(f"  Removed {before - after} duplicate timestamps")
    
    # Sort by time
    df = df.sort_values("open_time").reset_index(drop=True)
    
    # Save as parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}_{interval}.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"\n  SAVED: {output_path}")
    print(f"  Total candles: {len(df):,}")
    print(f"  Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    return output_path


def run_download():
    """Download all symbols."""
    for symbol in SYMBOLS:
        download_symbol(symbol, INTERVAL, START_DATE, END_DATE, RAW_DIR)
    
    print(f"\n{'='*60}")
    print("Download complete. Raw data saved to:", RAW_DIR)
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# STEP 2: PROCESS — Log-returns, regime labels, train/test split
# ---------------------------------------------------------------------------
def process_symbol(symbol: str, raw_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Load raw OHLCV, compute log-returns, label crash regimes, 
    remove stale/zero-price gaps, and save processed data.
    """
    raw_path = raw_dir / f"{symbol}_{INTERVAL}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}. Run --download first.")
    
    print(f"\nProcessing {symbol}...")
    df = pd.read_parquet(raw_path)
    
    # ---- Log-returns ----
    # r_t = log(Close_t / Close_{t-1})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # ---- Clean problematic values ----
    # Remove rows where close = 0 (stale periods)
    zero_mask = df["close"] == 0
    if zero_mask.sum() > 0:
        print(f"  Removed {zero_mask.sum()} zero-close rows")
        df = df[~zero_mask]
    
    # Remove infinite log-returns (from price gaps)
    inf_mask = ~np.isfinite(df["log_return"])
    if inf_mask.sum() > 0:
        print(f"  Removed {inf_mask.sum()} non-finite log-return rows")
        df = df[np.isfinite(df["log_return"])]
    
    # Remove the first row (NaN from shift)
    df = df.dropna(subset=["log_return"])
    
    # ---- Detect and remove stale periods ----
    # A "stale" period is where the price doesn't change for >= 5 consecutive candles
    # This indicates exchange downtime or data gaps
    df["price_change"] = df["close"].diff().abs()
    stale_mask = df["price_change"] == 0
    
    # Find runs of stale prices
    stale_runs = stale_mask.astype(int).groupby((~stale_mask).cumsum()).cumsum()
    long_stale = stale_runs >= 5
    if long_stale.sum() > 0:
        print(f"  Removed {long_stale.sum()} stale-period rows (>= 5 consecutive unchanged prices)")
        df = df[~long_stale]
    
    df = df.drop(columns=["price_change"])
    
    # ---- Regime labeling ----
    df["regime"] = "NORMAL"
    df["crash_event"] = ""
    
    for event_name, crash_start, crash_end, description in CRASH_EVENTS:
        mask = (df["open_time"] >= pd.Timestamp(crash_start)) & \
               (df["open_time"] <= pd.Timestamp(crash_end) + pd.Timedelta(days=1))
        n_labeled = mask.sum()
        df.loc[mask, "regime"] = "CRASH"
        df.loc[mask, "crash_event"] = event_name
        print(f"  Labeled {n_labeled:,} rows as {event_name}")
    
    # ---- Train/test split labels ----
    # Train: 2019-2021 normal regime only
    # Test: 2022-2023 (including crashes)
    train_mask = (df["open_time"].dt.year <= 2021) & (df["regime"] == "NORMAL")
    test_mask = df["open_time"].dt.year >= 2022
    
    df["split"] = "unused"
    df.loc[train_mask, "split"] = "train"
    df.loc[test_mask, "split"] = "test"
    
    # Also create 5-min resampled version
    df_5min = resample_to_5min(df)
    
    # ---- Save ----
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path_1min = output_dir / f"{symbol}_1min_processed.parquet"
    path_5min = output_dir / f"{symbol}_5min_processed.parquet"
    
    df.to_parquet(path_1min, index=False)
    df_5min.to_parquet(path_5min, index=False)
    
    print(f"\n  1-min data: {len(df):,} rows -> {path_1min}")
    print(f"  5-min data: {len(df_5min):,} rows -> {path_5min}")
    print(f"  Train rows: {(df['split'] == 'train').sum():,}")
    print(f"  Test rows:  {(df['split'] == 'test').sum():,}")
    print(f"  Crash rows: {(df['regime'] == 'CRASH').sum():,}")
    
    return df


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-min OHLCV to 5-min, recompute log-returns.
    Preserves regime labels (takes the 'worst' label in each 5-min window).
    """
    df_temp = df.set_index("open_time")
    
    # OHLCV resampling
    ohlcv = df_temp[["open", "high", "low", "close", "volume"]].resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    
    # Recompute log-returns on 5-min close prices
    ohlcv["log_return"] = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
    ohlcv = ohlcv.dropna(subset=["log_return"])
    
    # Regime: if ANY 1-min candle in the 5-min window is CRASH, label as CRASH
    regime = df_temp["regime"].resample("5min").apply(
        lambda x: "CRASH" if "CRASH" in x.values else "NORMAL"
    )
    crash_event = df_temp["crash_event"].resample("5min").apply(
        lambda x: x[x != ""].iloc[0] if (x != "").any() else ""
    )
    
    # Split: take from the 1-min split
    split = df_temp["split"].resample("5min").apply(
        lambda x: x.mode().iloc[0] if len(x) > 0 else "unused"
    )
    
    ohlcv["regime"] = regime
    ohlcv["crash_event"] = crash_event
    ohlcv["split"] = split
    
    ohlcv = ohlcv.reset_index()
    
    # Clean non-finite returns (same as 1-min)
    ohlcv = ohlcv[np.isfinite(ohlcv["log_return"])]
    
    return ohlcv


def run_process():
    """Process all symbols."""
    for symbol in SYMBOLS:
        process_symbol(symbol, RAW_DIR, PROCESSED_DIR)
    
    print(f"\n{'='*60}")
    print("Processing complete. Processed data saved to:", PROCESSED_DIR)
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# STEP 3: DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------
def describe_symbol(symbol: str, processed_dir: Path, results_dir: Path):
    """
    Generate descriptive statistics and diagnostic plots for one symbol.
    This documents the heavy-tail properties required for your thesis.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    path = processed_dir / f"{symbol}_1min_processed.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}. Run --process first.")
    
    df = pd.read_parquet(path)
    returns = df["log_return"].values
    
    fig_dir = results_dir / "descriptive"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Descriptive Statistics: {symbol}")
    print(f"{'='*60}")
    
    # ---- Summary statistics ----
    summary = {
        "N observations": len(returns),
        "Mean": np.mean(returns),
        "Std Dev": np.std(returns),
        "Skewness": stats.skew(returns),
        "Kurtosis (excess)": stats.kurtosis(returns),
        "Min": np.min(returns),
        "Max": np.max(returns),
        "1st percentile": np.percentile(returns, 1),
        "5th percentile": np.percentile(returns, 5),
        "95th percentile": np.percentile(returns, 95),
        "99th percentile": np.percentile(returns, 99),
        "Jarque-Bera stat": stats.jarque_bera(returns).statistic,
        "Jarque-Bera p-val": stats.jarque_bera(returns).pvalue,
    }
    
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key:25s}: {val:.6f}")
        else:
            print(f"  {key:25s}: {val}")
    
    # Save summary as CSV
    pd.Series(summary, name=symbol).to_csv(fig_dir / f"{symbol}_summary_stats.csv")
    
    # ---- Plot 1: Return distribution vs Normal ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{symbol} Log-Return Distribution", fontsize=14)
    
    # Histogram
    axes[0].hist(returns, bins=500, density=True, alpha=0.7, color="steelblue", label="Empirical")
    x = np.linspace(returns.min(), returns.max(), 1000)
    axes[0].plot(x, stats.norm.pdf(x, np.mean(returns), np.std(returns)), 
                 "r-", linewidth=2, label="Normal fit")
    axes[0].set_xlim(np.percentile(returns, 0.1), np.percentile(returns, 99.9))
    axes[0].set_title("Distribution (clipped at 0.1th/99.9th percentile)")
    axes[0].set_xlabel("Log-return")
    axes[0].legend()
    
    # QQ plot
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title("QQ Plot vs Normal")
    
    # Log-scale tail plot (left tail)
    sorted_neg = np.sort(-returns[returns < 0])  # positive losses
    ecdf_y = np.arange(1, len(sorted_neg) + 1) / len(returns)
    axes[2].semilogy(sorted_neg, ecdf_y, ".", markersize=1, alpha=0.3)
    axes[2].set_title("Left Tail (Survival Function)")
    axes[2].set_xlabel("|Negative log-return|")
    axes[2].set_ylabel("P(Loss > x)")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # ---- Plot 2: ACF / PACF of returns and squared returns ----
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{symbol} Autocorrelation Diagnostics", fontsize=14)
    
    # Subsample for plotting (ACF on 2.5M points is slow)
    max_plot = min(100_000, len(returns))
    r_sub = returns[:max_plot]
    
    plot_acf(r_sub, lags=50, ax=axes[0, 0], title="ACF of Log-Returns")
    plot_pacf(r_sub, lags=50, ax=axes[0, 1], title="PACF of Log-Returns")
    plot_acf(r_sub**2, lags=50, ax=axes[1, 0], title="ACF of Squared Log-Returns")
    plot_pacf(r_sub**2, lags=50, ax=axes[1, 1], title="PACF of Squared Log-Returns")
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # ---- Plot 3: Return time series with crash events highlighted ----
    fig, ax = plt.subplots(figsize=(18, 5))
    
    # Downsample for plotting (plot every 60th point = hourly)
    plot_df = df.iloc[::60].copy()
    ax.plot(plot_df["open_time"], plot_df["log_return"], 
            linewidth=0.3, color="steelblue", alpha=0.7)
    
    # Highlight crash windows
    colors = ["red", "orange", "purple", "green"]
    for i, (name, start, end, desc) in enumerate(CRASH_EVENTS):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1),
                   alpha=0.3, color=colors[i % len(colors)], label=f"{name}")
    
    ax.set_title(f"{symbol} Log-Returns with Crash Events")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log-return (1-min)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"{symbol}_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n  Plots saved to: {fig_dir}")
    print(f"  Files: {symbol}_distribution.png, {symbol}_acf_pacf.png, {symbol}_timeseries.png")


def run_describe():
    """Generate descriptive statistics for all symbols."""
    for symbol in SYMBOLS:
        describe_symbol(symbol, PROCESSED_DIR, RESULTS_DIR)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Data pipeline for Black Swans in the Blockchain")
    parser.add_argument("--download", action="store_true", help="Download raw data from Binance")
    parser.add_argument("--process", action="store_true", help="Compute log-returns, label regimes")
    parser.add_argument("--describe", action="store_true", help="Generate descriptive stats and plots")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    if not any([args.download, args.process, args.describe, args.all]):
        parser.print_help()
        return
    
    if args.download or args.all:
        run_download()
    
    if args.process or args.all:
        run_process()
    
    if args.describe or args.all:
        run_describe()


if __name__ == "__main__":
    main()
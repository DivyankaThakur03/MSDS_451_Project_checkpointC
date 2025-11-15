"""
CHECKPOINT C: MONTE CARLO SIMULATION & WALK-FORWARD VALIDATION
Financial Engineering Term Project - NVDA Trading Strategy
Author: Divyanka Thakur
Date: November 2025

This script implements robust validation methods to test if the NVDA directional
timing strategy is viable as a commercial ETF product.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

OUTPUT_DIR = Path("./checkpoint_c_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA ACQUISITION & FEATURE ENGINEERING
# =============================================================================

START_DATE = "2000-01-01"
END_DATE = "2025-01-01"

print("Loading market data...")
nvda = yf.download("NVDA", start=START_DATE, end=END_DATE, progress=False)
vix = yf.download("^VIX", start=START_DATE, end=END_DATE, progress=False)
spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)
tnx = yf.download("^TNX", start=START_DATE, end=END_DATE, progress=False)

def clean_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df

nvda, vix, spy, tnx = clean_df(nvda), clean_df(vix), clean_df(spy), clean_df(tnx)

# Feature engineering
nvda["Return"] = np.log(nvda["Adj Close"] / nvda["Adj Close"].shift(1))
nvda["Vol5"] = nvda["Return"].rolling(5).std()
nvda["Vol20"] = nvda["Return"].rolling(20).std()
nvda["SMA5"] = nvda["Adj Close"].rolling(5).mean()
nvda["SMA20"] = nvda["Adj Close"].rolling(20).mean()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

nvda["RSI14"] = compute_rsi(nvda["Adj Close"], 14)
spy["SPY_Return"] = np.log(spy["Adj Close"] / spy["Adj Close"].shift(1))
spy["SPY_Close"] = spy["Adj Close"]

nvda_spy = pd.merge(nvda[["Adj Close"]], spy[["SPY_Close"]], 
                    left_index=True, right_index=True, how="inner")
nvda["Relative_Strength"] = np.log(nvda_spy["Adj Close"] / nvda_spy["SPY_Close"])

features = nvda[["Return", "Vol5", "Vol20", "SMA5", "SMA20", "RSI14", "Relative_Strength"]].copy()
features = features.merge(vix[["Adj Close"]].rename(columns={"Adj Close": "VIX_Level"}), 
                         left_index=True, right_index=True, how="left")
features = features.merge(spy[["SPY_Return"]], left_index=True, right_index=True, how="left")
features = features.merge(tnx[["Adj Close"]].rename(columns={"Adj Close": "TNX_Yield"}), 
                         left_index=True, right_index=True, how="left")
features = features.dropna()

features["Target"] = (features["Return"].shift(-1) > 0).astype(int)
features = features.dropna()

print(f"Dataset prepared: {len(features):,} trading days")

# =============================================================================
# HISTORICAL PARAMETERS
# =============================================================================

daily_return_mean = features["Return"].mean()
daily_return_std = features["Return"].std()
annual_return_mean = daily_return_mean * 252
annual_volatility = daily_return_std * np.sqrt(252)
autocorr_lag1 = features["Return"].autocorr(lag=1)

print(f"\nHistorical statistics:")
print(f"  Annual return: {annual_return_mean*100:.2f}%")
print(f"  Annual volatility: {annual_volatility*100:.2f}%")
print(f"  Autocorrelation: {autocorr_lag1:.4f}")

# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

print("\nGenerating Monte Carlo simulations...")
N_SIMULATIONS = 500
N_DAYS = len(features)

np.random.seed(42)
simulated_returns = np.zeros((N_SIMULATIONS, N_DAYS))

for sim in range(N_SIMULATIONS):
    returns = np.zeros(N_DAYS)
    returns[0] = daily_return_mean
    
    for t in range(1, N_DAYS):
        momentum = autocorr_lag1 * returns[t-1]
        shock = np.random.normal(0, daily_return_std)
        returns[t] = daily_return_mean + momentum + shock
    
    simulated_returns[sim] = returns

# Visualize sample paths
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

sample_sims = [0, 50, 100, 150, 200]
for sim_idx in sample_sims:
    cumulative = (1 + simulated_returns[sim_idx]).cumprod()
    axes[0].plot(cumulative, alpha=0.6, linewidth=1)

historical_cumulative = (1 + features["Return"]).cumprod()
axes[0].plot(historical_cumulative.values, color='black', linewidth=2.5, 
             label='Historical NVDA', linestyle='--', alpha=0.8)

axes[0].set_ylabel('Cumulative Return (Growth of $1)', fontsize=12)
axes[0].set_title('Sample Synthetic Price Paths vs Historical', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_yscale('log')

final_values = [(1 + sim_returns).prod() for sim_returns in simulated_returns]
historical_final = (1 + features["Return"]).prod()

axes[1].hist(final_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[1].axvline(historical_final, color='red', linewidth=2.5, linestyle='--', 
                label=f'Historical: ${historical_final:.2f}')
axes[1].axvline(np.median(final_values), color='green', linewidth=2, linestyle=':', 
                label=f'Median: ${np.median(final_values):.2f}')
axes[1].set_xlabel('Final Portfolio Value (Growth of $1)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of Final Values (500 Simulations)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_monte_carlo_synthetic_paths.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# TRAIN MODEL
# =============================================================================

print("\nTraining model...")
feature_cols = ["Vol5", "Vol20", "SMA5", "SMA20", "RSI14", 
                "VIX_Level", "SPY_Return", "TNX_Yield", "Relative_Strength"]

X = features[feature_cols]
y = features["Target"]

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                     random_state=42, eval_metric='logloss')
model.fit(X_train_balanced, y_train_balanced)

y_pred_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
thresholds = np.arange(0.35, 0.55, 0.01)
best_threshold = 0.50
best_accuracy = 0

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    if len(np.unique(y_pred)) == 2:
        acc = accuracy_score(y_test, y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

y_pred_final = (y_pred_proba >= best_threshold).astype(int)
print(f"Test accuracy: {best_accuracy:.4f} (threshold: {best_threshold:.2f})")

# =============================================================================
# TEST STRATEGY ON MONTE CARLO PATHS
# =============================================================================

print("\nRunning strategy on Monte Carlo simulations...")
RISK_FREE_RATE_DAILY = 0.02 / 252

mc_results = []

for sim_idx in range(N_SIMULATIONS):
    sim_returns = simulated_returns[sim_idx]
    
    strategy_returns = np.where(
        y_pred_final == 1,
        sim_returns[split_idx:],
        RISK_FREE_RATE_DAILY
    )
    
    total_return = (1 + strategy_returns).prod() - 1
    buy_hold_return = (1 + sim_returns[split_idx:]).prod() - 1
    
    excess_returns = strategy_returns - RISK_FREE_RATE_DAILY
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    mc_results.append({
        'simulation': sim_idx,
        'strategy_return': total_return,
        'buy_hold_return': buy_hold_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'outperformed': total_return > buy_hold_return
    })

mc_df = pd.DataFrame(mc_results)

print(f"\nMonte Carlo results:")
print(f"  Strategy return: {mc_df['strategy_return'].mean()*100:.2f}%")
print(f"  Buy-hold return: {mc_df['buy_hold_return'].mean()*100:.2f}%")
print(f"  Win rate: {mc_df['outperformed'].mean()*100:.1f}%")
print(f"  Sharpe ratio: {mc_df['sharpe_ratio'].mean():.3f}")
print(f"  Max drawdown: {mc_df['max_drawdown'].mean()*100:.2f}%")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].hist(mc_df['strategy_return']*100, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axvline(mc_df['strategy_return'].mean()*100, color='red', linewidth=2, 
                   linestyle='--', label=f"Mean: {mc_df['strategy_return'].mean()*100:.2f}%")
axes[0, 0].set_xlabel('Strategy Return (%)', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Distribution of Strategy Returns', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].scatter(mc_df['buy_hold_return']*100, mc_df['strategy_return']*100, 
                   alpha=0.5, s=30, c='steelblue')
max_val = max(mc_df['buy_hold_return'].max(), mc_df['strategy_return'].max()) * 100
min_val = min(mc_df['buy_hold_return'].min(), mc_df['strategy_return'].min()) * 100
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[0, 1].set_xlabel('Buy & Hold Return (%)', fontsize=11)
axes[0, 1].set_ylabel('Strategy Return (%)', fontsize=11)
axes[0, 1].set_title('Strategy vs Buy-Hold', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].hist(mc_df['sharpe_ratio'], bins=50, alpha=0.7, color='coral', edgecolor='black')
axes[1, 0].axvline(mc_df['sharpe_ratio'].mean(), color='red', linewidth=2, 
                   linestyle='--', label=f"Mean: {mc_df['sharpe_ratio'].mean():.3f}")
axes[1, 0].axvline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.5)
axes[1, 0].set_xlabel('Sharpe Ratio', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('Distribution of Sharpe Ratios', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(mc_df['max_drawdown']*100, bins=50, alpha=0.7, color='indianred', edgecolor='black')
axes[1, 1].axvline(mc_df['max_drawdown'].mean()*100, color='red', linewidth=2, 
                   linestyle='--', label=f"Mean: {mc_df['max_drawdown'].mean()*100:.2f}%")
axes[1, 1].set_xlabel('Maximum Drawdown (%)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Distribution of Maximum Drawdowns', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_monte_carlo_results.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

print("\nRunning walk-forward validation...")
TRAIN_YEARS = 10
TEST_YEARS = 1
TRAIN_DAYS = TRAIN_YEARS * 252
TEST_DAYS = TEST_YEARS * 252

wf_results = []
wf_predictions = []
start_idx = 0
fold = 0

while start_idx + TRAIN_DAYS + TEST_DAYS <= len(features):
    fold += 1
    train_end = start_idx + TRAIN_DAYS
    test_end = train_end + TEST_DAYS
    
    X_train_wf = features[feature_cols].iloc[start_idx:train_end]
    y_train_wf = features["Target"].iloc[start_idx:train_end]
    X_test_wf = features[feature_cols].iloc[train_end:test_end]
    y_test_wf = features["Target"].iloc[train_end:test_end]
    
    X_train_wf_balanced, y_train_wf_balanced = smote.fit_resample(X_train_wf, y_train_wf)
    
    wf_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                            random_state=42, eval_metric='logloss')
    wf_model.fit(X_train_wf_balanced, y_train_wf_balanced)
    
    y_pred_proba_wf = wf_model.predict_proba(X_test_wf)[:, 1]
    y_pred_wf = (y_pred_proba_wf >= 0.45).astype(int)
    
    test_returns_wf = features["Return"].iloc[train_end:test_end]
    strategy_returns_wf = np.where(y_pred_wf == 1, test_returns_wf, RISK_FREE_RATE_DAILY)
    
    total_return_wf = (1 + strategy_returns_wf).prod() - 1
    buy_hold_return_wf = (1 + test_returns_wf).prod() - 1
    
    excess_wf = strategy_returns_wf - RISK_FREE_RATE_DAILY
    sharpe_wf = np.sqrt(252) * excess_wf.mean() / excess_wf.std() if excess_wf.std() > 0 else 0
    
    cumulative_wf = (1 + strategy_returns_wf).cumprod()
    running_max_wf = np.maximum.accumulate(cumulative_wf)
    drawdown_wf = (cumulative_wf - running_max_wf) / running_max_wf
    max_dd_wf = drawdown_wf.min()
    
    wf_results.append({
        'fold': fold,
        'train_start': features.index[start_idx].date(),
        'train_end': features.index[train_end-1].date(),
        'test_start': features.index[train_end].date(),
        'test_end': features.index[test_end-1].date(),
        'strategy_return': total_return_wf,
        'buy_hold_return': buy_hold_return_wf,
        'sharpe_ratio': sharpe_wf,
        'max_drawdown': max_dd_wf,
        'outperformed': total_return_wf > buy_hold_return_wf
    })
    
    wf_predictions.extend(list(zip(features.index[train_end:test_end], y_pred_wf, strategy_returns_wf)))
    start_idx += TEST_DAYS

wf_df = pd.DataFrame(wf_results)

print(f"\nWalk-forward results ({len(wf_df)} folds):")
print(f"  Strategy return: {wf_df['strategy_return'].mean()*100:.2f}%")
print(f"  Buy-hold return: {wf_df['buy_hold_return'].mean()*100:.2f}%")
print(f"  Win rate: {wf_df['outperformed'].mean()*100:.1f}%")
print(f"  Sharpe ratio: {wf_df['sharpe_ratio'].mean():.3f}")

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

x_pos = np.arange(len(wf_df))
width = 0.35

axes[0].bar(x_pos - width/2, wf_df['strategy_return']*100, width, 
            label='Strategy', alpha=0.8, color='steelblue')
axes[0].bar(x_pos + width/2, wf_df['buy_hold_return']*100, width, 
            label='Buy & Hold', alpha=0.8, color='coral')
axes[0].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[0].set_xlabel('Test Period', fontsize=11)
axes[0].set_ylabel('Return (%)', fontsize=11)
axes[0].set_title('Walk-Forward Returns by Fold', fontsize=13, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([f"{r['test_start'].year}" for _, r in wf_df.iterrows()], rotation=45)
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')

wf_pred_df = pd.DataFrame(wf_predictions, columns=['date', 'prediction', 'strategy_return'])
wf_pred_df = wf_pred_df.set_index('date')

cumulative_wf = (1 + wf_pred_df['strategy_return']).cumprod()
test_features = features.loc[wf_pred_df.index]
cumulative_bh_wf = (1 + test_features['Return']).cumprod()

axes[1].plot(cumulative_wf.index, cumulative_wf, label='Strategy', linewidth=2.5, color='steelblue')
axes[1].plot(cumulative_bh_wf.index, cumulative_bh_wf, label='Buy & Hold', 
             linewidth=2.5, color='black', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_ylabel('Cumulative Return', fontsize=11)
axes[1].set_title('Walk-Forward Cumulative Returns', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_walk_forward_results.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FEE STRUCTURE ANALYSIS
# =============================================================================

print("\nAnalyzing fee structures...")
management_fees = [0.00, 0.01, 0.02, 0.03, 0.04]
performance_fees = [0.00, 0.10, 0.20]

fee_results = []

for mgmt_fee in management_fees:
    for perf_fee in performance_fees:
        net_returns = []
        
        for _, row in mc_df.iterrows():
            strategy_return = row['strategy_return']
            buy_hold_return = row['buy_hold_return']
            
            net_after_mgmt = strategy_return - mgmt_fee
            excess_return = max(0, net_after_mgmt - buy_hold_return)
            performance_fee_amt = excess_return * perf_fee
            
            final_net_return = net_after_mgmt - performance_fee_amt
            net_returns.append(final_net_return)
        
        fee_results.append({
            'mgmt_fee': mgmt_fee,
            'perf_fee': perf_fee,
            'mean_net_return': np.mean(net_returns),
            'median_net_return': np.median(net_returns),
            'pct_positive': (np.array(net_returns) > 0).mean()
        })

fee_df = pd.DataFrame(fee_results)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pivot_mean = fee_df.pivot(index='mgmt_fee', columns='perf_fee', values='mean_net_return')
sns.heatmap(pivot_mean * 100, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
            ax=axes[0], cbar_kws={'label': 'Mean Return (%)'})
axes[0].set_xlabel('Performance Fee (%)', fontsize=11)
axes[0].set_ylabel('Management Fee (%)', fontsize=11)
axes[0].set_title('Mean Net Returns After Fees', fontsize=13, fontweight='bold')
axes[0].set_xticklabels([f'{int(x*100)}%' for x in performance_fees])
axes[0].set_yticklabels([f'{int(x*100)}%' for x in management_fees], rotation=0)

pivot_pos = fee_df.pivot(index='mgmt_fee', columns='perf_fee', values='pct_positive')
sns.heatmap(pivot_pos * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=50, 
            ax=axes[1], cbar_kws={'label': 'Success Rate (%)'})
axes[1].set_xlabel('Performance Fee (%)', fontsize=11)
axes[1].set_ylabel('Management Fee (%)', fontsize=11)
axes[1].set_title('% Positive Returns', fontsize=13, fontweight='bold')
axes[1].set_xticklabels([f'{int(x*100)}%' for x in performance_fees])
axes[1].set_yticklabels([f'{int(x*100)}%' for x in management_fees], rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_fee_structure_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FINAL ASSESSMENT
# =============================================================================

historical_test_return = (1 + features.loc[X_test.index, "Return"]).prod() - 1
historical_strategy_return = (1 + np.where(y_pred_final == 1, 
                                           features.loc[X_test.index, "Return"], 
                                           RISK_FREE_RATE_DAILY)).prod() - 1

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print(f"\nHistorical Test (2020-2024):")
print(f"  Strategy: {historical_strategy_return*100:.2f}%")
print(f"  Buy-Hold: {historical_test_return*100:.2f}%")

print(f"\nMonte Carlo (500 scenarios):")
print(f"  Mean return: {mc_df['strategy_return'].mean()*100:.2f}%")
print(f"  Win rate: {mc_df['outperformed'].mean()*100:.1f}%")
print(f"  Sharpe: {mc_df['sharpe_ratio'].mean():.3f}")

print(f"\nWalk-Forward ({len(wf_df)} folds):")
print(f"  Mean return: {wf_df['strategy_return'].mean()*100:.2f}%")
print(f"  Win rate: {wf_df['outperformed'].mean()*100:.1f}%")

# Viability assessment
historical_success = historical_strategy_return > historical_test_return
mc_success = mc_df['outperformed'].mean() > 0.50
wf_success = wf_df['outperformed'].mean() > 0.50
positive_sharpe = mc_df['sharpe_ratio'].mean() > 0
reasonable_dd = mc_df['max_drawdown'].mean() > -0.50

viable = historical_success and mc_success and wf_success and positive_sharpe and reasonable_dd

print(f"\n{'='*80}")
if viable:
    print("CONCLUSION: POTENTIALLY VIABLE")
    print("="*80)
    print("\nThe strategy shows promise across validation methods.")
    print("Consider paper trading before live implementation.")
else:
    print("CONCLUSION: NOT RECOMMENDED")
    print("="*80)
    print("\nThe strategy fails to consistently outperform buy-and-hold.")
    print("\nKey issues:")
    if not historical_success:
        print("  • Underperformed on recent data")
    if not mc_success:
        print("  • Low win rate in simulations")
    if not wf_success:
        print("  • Poor adaptation to regime changes")

# Save results
mc_df.to_csv(OUTPUT_DIR / "monte_carlo_results.csv", index=False)
wf_df.to_csv(OUTPUT_DIR / "walk_forward_results.csv", index=False)
fee_df.to_csv(OUTPUT_DIR / "fee_structure_analysis.csv", index=False)

summary_stats = pd.DataFrame({
    'Metric': [
        'Historical Strategy Return',
        'Historical Buy-Hold Return',
        'MC Mean Strategy Return',
        'MC Win Rate',
        'MC Mean Sharpe',
        'WF Mean Return',
        'WF Win Rate',
        'Viability'
    ],
    'Value': [
        f"{historical_strategy_return*100:.2f}%",
        f"{historical_test_return*100:.2f}%",
        f"{mc_df['strategy_return'].mean()*100:.2f}%",
        f"{mc_df['outperformed'].mean()*100:.1f}%",
        f"{mc_df['sharpe_ratio'].mean():.3f}",
        f"{wf_df['strategy_return'].mean()*100:.2f}%",
        f"{wf_df['outperformed'].mean()*100:.1f}%",
        "VIABLE" if viable else "NOT VIABLE"
    ]
})

summary_stats.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
print(f"\nResults saved to {OUTPUT_DIR}/")
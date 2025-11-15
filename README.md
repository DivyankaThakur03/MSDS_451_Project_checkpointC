# MSDS 451 Project: Checkpoint C
## Final Validation & Business Viability Assessment

Building on Checkpoints A & B, this project rigorously validates the NVDA directional timing strategy using Monte Carlo simulation and walk-forward testing to determine commercial viability as an active ETF product.

---

## Key Question
**Should we launch this NVDA timing strategy as an active ETF?**

**Answer: NO** - Strategy consistently underperforms buy-and-hold across all validation methods.

---

## What's New in Checkpoint C

- **Monte Carlo Simulation** - Tested strategy on 500 synthetic market scenarios  
- **Walk-Forward Validation** - Rolling 10-year training windows (14 folds, 2010-2024)  
- **Fee Structure Analysis** - Impact of management (0-4%) and performance (0-20%) fees  
- **Business Decision** - Clear viability assessment with 5 objective criteria  

---

## Key Results Summary

| Validation Method | Strategy | Buy-Hold | Win Rate |
|-------------------|----------|----------|----------|
| **Historical Test (2020-2024)** | +231% | +999% | 0% |
| **Monte Carlo (500 sims)** | +107% | +292% | 38.8% |
| **Walk-Forward (14 folds)** | +5% | +49% | 14.3% |

**Conclusion:** Strategy fails all viability criteria. Not recommended for commercialization.

---

## Repository Contents
```
/checkpoint_c_figures/
  ├── 01_monte_carlo_synthetic_paths.png
  ├── 02_monte_carlo_results.png
  ├── 03_walk_forward_results.png
  ├── 04_fee_structure_analysis.png
  ├── monte_carlo_results.csv
  ├── walk_forward_results.csv
  ├── fee_structure_analysis.csv
  └── summary_statistics.csv
checkpointc.py
Checkpointc_report.pdf
README.md
```

---

## How to Run
```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost yfinance imbalanced-learn matplotlib seaborn

# Run validation script
python checkpointc.py
```

**Runtime:** ~10-15 minutes  
**Output:** 4 figures + 4 CSV files saved to `./checkpoint_c_figures/`

---

## Why Did the Strategy Fail?

1. **Regime Adaptation Failure** - Model couldn't anticipate 2023-2024 AI boom
2. **Binary Switching Too Extreme** - 100% in/out amplifies cost of missing rallies
3. **Volatility Misinterpretation** - High volatility during surge interpreted as bearish
4. **Opportunity Cost** - Missing critical up days destroyed annual performance

---

## Key Lessons

- Monte Carlo + Walk-Forward validation are essential
- Negative results are valuable research (prevent capital misallocation)
- Market timing is extremely difficult for high-growth assets
- Binary strategies amplify prediction errors
- Regime changes break models trained on historical data

---

**Author:** Divyanka Thakur  
**Date:** November 2025  

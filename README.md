# SOH-LoadNorm: Discharge-Only Proxy SOH Estimation

Code for reproducing figures in "Discharge-Only Proxy SOH Estimation via Load-Normalized Voltage Dispersion with Post-Hoc Reliability Indicators under VIN-Grouped Validation."

## Data

The dataset is from [Liu et al. (2025)](https://www.nature.com/articles/s41467-025-56485-7), publicly available at http://ivstskl.changan.com.cn/?p=2697.

## Figures

- `fig1.py` — Pipeline overview
- `fig2.py` — Label validity assessment
- `fig3.py` — Main prediction performance
- `fig4.py` — LoadNorm mechanism evidence
- `fig5.py` — Generalization limits
- `fig6.py` — Reliability framework

## Requirements

```
numpy pandas matplotlib scipy scikit-learn catboost
```

## Citation

If you use this code, please cite our paper.

# SOH-LoadNorm: Discharge-Only Proxy SOH Estimation

Code for reproducing figures in "Discharge-Only Proxy SOH Estimation via Load-Normalized Voltage Dispersion with Post-Hoc Reliability Indicators under VIN-Grouped Validation."

## Data

The dataset is from [Liu et al. (2025)](https://www.nature.com/articles/s41467-025-56485-7), made publicly available by IVST (State Key Laboratory of Intelligent Vehicle Safety Technology) under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

**This repository does not contain the dataset.** Please download the data directly from the original source: http://ivstskl.changan.com.cn/?p=2697. The dataset license prohibits redistribution to third parties (see [Terms of Use](http://ivstskl.changan.com.cn/?p=2758)). Data paths in the figure scripts must be updated to point to your local copy.

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

## Code and Data Availability

### GitHub Repository (code and sample data)

The implementation, training scripts, and sample data are available on GitHub:

- Sample data:  
  https://github.com/JaneGondwegithub/swin-convlstm-lulc-prediction/tree/main/data/sample  

- Training scripts:  
  https://github.com/JaneGondwegithub/swin-convlstm-lulc-prediction/tree/main/training  

These are provided for demonstration and reproducibility of the workflow.

---

### Zenodo (full data, models, and outputs)

Due to size constraints, full datasets and trained models are hosted on Zenodo:

- Dataset (LULC maps and drivers):  
  https://doi.org/10.5281/zenodo.19940989  

- Trained models and 2024 validation outputs:  
  https://doi.org/10.5281/zenodo.19927477  

- Future projections (2029–2034):  
  https://doi.org/10.5281/zenodo.19927974  

---

### Reproducibility Instructions

To reproduce results:

1. Download full dataset and models from Zenodo  
2. Place:
   - datasets → `data/processed/`  
   - model weights → `results/`  
3. Use scripts in `training/`, `validation/`, and `inference/`  
4. Sample data in `data/sample/` can be used for quick testing

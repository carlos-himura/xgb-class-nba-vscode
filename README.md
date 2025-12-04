#NBA Classification Challenge – Local XGBoost Modeling (VS Code)

This repository contains the local development version of an NBA binary classification project.
All experimentation was done in VS Code, including EDA, feature engineering, modeling, Hyperopt tuning, and MLflow tracking.

The goal of this challenge was to build an XGBoost model capable of predicting whether a player would perform:

- 1 → Above Average

- 0 → Below Average

based on historical NBA game statistics.

| Notebook                              | Description                                                                                                                                                                                                       |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1_eda_challenge_2.ipynb**           | Conducts the **exploratory data analysis (EDA)** for the challenge. Includes distribution checks, correlations, missing values inspection, and initial insights to understand the dataset.                        |
| **2_feature_eng_challenge_2.ipynb**   | Performs **feature engineering**, transformations, and selection. Features were filtered by retaining those required to reach **95% of total importance**, calculated via `xgb_clf.feature_importances_`.         |
| **3_training_xgb_challenge_2.ipynb**  | Trains XGBoost models **with and without Hyperopt**, and also trains models **with and without feature engineering** to compare performance. Includes **MLflow tracking** for parameters, metrics, and artifacts. |
| **4_training_xgb2_challenge_2.ipynb** | Final training notebook using **feature engineering + Hyperopt**, producing the **best model**. Also includes **MLflow tracking** for full experiment reproducibility.                                            |

# Datasets

Files inside the Datasets/ folder include:

- training_data.csv – main training dataset
- ds_feature_eng.csv – dataset after feature engineering
- ds_preprocessed.csv – cleaned dataset version
- blind_test_data.csv – hold-out dataset for final evaluation

# Technologies & Tools

- Python
- XGBoost
- Pandas / NumPy
- Scikit-Learn
- Hyperopt (Bayesian optimization)
- MLflow (experiment tracking)
- Matplotlib / Seaborn

# Hyperparameter Optimization (Hyperopt)

Hyperopt was used to optimize:
- max_depth
- eta
- min_child_weight
- subsample
- gamma
- colsample_bytree

Search strategy:
- Random + TPE (Bayesian optimization)

Hyperopt results were reused to train the best model in Notebook 4.

# Class Balancing

To handle class imbalance, XGBoost’s scale_pos_weight was used:
scale_pos_weight = negative_samples / positive_samples

This helped the model focus more on minority class predictions.

# Summary:
This repo documents the local development workflow that preceded the final SageMaker deployment version.
It includes full experimentation, feature selection, Hyperopt tuning, MLflow tracking, and final model selection.

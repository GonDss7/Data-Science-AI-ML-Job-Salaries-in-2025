Data Science, AI & ML Job Salaries (EDA + ML, 2020–2025)
Goal. Explore and model global salaries for Data Science / ML / AI roles using the Kaggle dataset “Salaries for Data Science Jobs” (2020–2025).
What’s inside. Data cleaning, exploratory analysis, outlier handling, categorical encoding, and baseline regression models (Linear Regression, Random Forest, XGBoost).

Notebook: Ds.ipynb
Data: salaries.csv (Kaggle) → uses salary_in_usd as the target

1) Dataset
Each row is a reported salary with context:

work_year, experience_level (EN/MI/SE/EX), employment_type (FT/PT/CT/FL)

job_title, employee_residence, company_location, company_size (S/M/L)

remote_ratio (0 / 50 / 100)

salary (original currency) and salary_in_usd (normalized target)

Why salary_in_usd? It standardizes currencies and makes global comparisons meaningful.

2) Environment & Libraries
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

bash
Copiar
Editar
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
3) Methodology (step-by-step)
3.1 Load & Inspect
Load salaries.csv, check dtypes, basic info and missing values.

Convert numeric fields and show duplicate rows for sanity checks.

3.2 Basic Cleaning
Drop exact duplicates.

Drop salary and salary_currency (we model salary_in_usd only).

3.3 Exploratory Data Analysis (EDA)
Univariate: histogram + boxplot of salary_in_usd.

Categorical views: count plots and salary distributions by:

experience_level

employment_type

These reveal skewness (high-end outliers) and expected pay gaps across levels.

3.4 Outlier Handling (IQR rule)
To avoid extreme values dominating the model, the notebook trims outliers using the Interquartile Range:

keep if 
𝑄
1
−
1.5
 
𝐼
𝑄
𝑅
≤
𝑦
≤
𝑄
3
+
1.5
 
𝐼
𝑄
𝑅
keep if Q1−1.5IQR≤y≤Q3+1.5IQR
This produces df_no_outliers and a cleaner salary distribution.

3.5 Feature Preparation
Features: all columns except salary_in_usd

Encoding: One-Hot Encoding for categorical variables (get_dummies(..., drop_first=True))

Split: 80/20 train–test

3.6 Baseline Models
Linear Regression (OLS)

RandomForestRegressor (100 trees)

XGBRegressor (100 estimators; default learning rate and depth)

3.7 Evaluation
Report MSE, RMSE, and R² on the test set.

On my baseline run with one-hot encoding (after outlier trimming):
• Linear Regression: R² ≈ 0.33
• Random Forest: R² ≈ 0.33
• XGBoost (100 trees, default): similar without tuning

Interpretation: with one-hot encoding and default parameters, the models capture a modest portion of variance. High cardinality features (e.g., job_title, locations) and remaining noise limit performance without additional feature engineering or tuning.

4) Key Findings (EDA)
Experience matters: higher experience_level shifts the salary distribution upward.

Remote ratio varies by role/company, but salary differences exist; worth slicing by company_location.

Geography matters: location effects are large; normalizing to USD is essential, but country/region still drives differences.

Outliers: trimming with IQR stabilizes the distribution and helps models generalize.

5) Why baseline ML underperforms & how to improve
What hurts performance

High-cardinality categoricals (job_title, countries) explode columns with pure One-Hot, producing sparse features and diluting signal.

Default hyperparameters on ensembles rarely fit this kind of data well.

Mixed effects (year × location × seniority) aren’t captured explicitly.

What to try next

Encoding alternatives for high cardinality

Label / Ordinal encoding for tree-based models (RF, XGBoost, LightGBM).

Target / Leave-One-Out encoding (with cross-validated leakage control).

Frequency/Count encoding for job_title / company_location.

Modeling

Tune RandomForest (n_estimators, max_depth, min_samples_leaf).

Try XGBoost/LightGBM/CatBoost with proper tuning.

Segment models by region or experience level.

Validation

Use K-Fold CV to get stable estimates.

Track MAE (business-friendly) alongside R²/RMSE.

Explainability

Feature importance (tree-based).

Partial dependence / SHAP for top drivers.

6) Reproducibility
Put salaries.csv in the project root.

Open Ds.ipynb (Colab or Jupyter) and run sequentially.

Optional: export cleaned data and visuals to feed a BI dashboard (Tableau/Power BI).

7) Project Structure
bash
Copiar
Editar
.
├── Ds.ipynb               # analysis notebook (EDA + ML)
├── salaries.csv           # raw dataset (not included in repo)
├── /figures               # export charts (optional)
└── README.md              # this file
8) Conclusions
The dataset is rich and realistic; EDA confirms strong effects from experience and geography.

Outlier trimming is necessary for stable modeling.

With pure one-hot encoding and default models, R² ~ 0.33 is a reasonable baseline.

Better encoding for high-cardinality features (and hyperparameter tuning) is the lever to substantially increase predictive power.

Next iterations will compare Label/Target encoding + XGBoost/LightGBM/CatBoost and introduce cross-validated model selection.

9) Acknowledgements
Dataset: Salaries for Data Science Jobs (Kaggle).
Thanks to the open-source community behind pandas, scikit-learn, xgboost, matplotlib, and seaborn.

License
This project is for educational and research purposes. Use the dataset according to the original Kaggle license.

If you want, I can tailor this to your exact metric prints (copy me your final MSE/RMSE/R²) and add a mini “Results” table + image placeholders for your charts.

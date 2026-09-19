"""
03_modelo_regresion.py
Entrenamiento, evaluación y diagnóstico de la regresión lineal.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan

INPUT = Path("../data/Saber_11_2020-1.csv")
OUT = Path("../resultados")
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "PUNT_GLOBAL"
FEATURES = [
    "FAMI_ESTRATOVIVIENDA",
    "FAMI_PERSONASHOGAR",
    "ESTU_GENERO",
    "COLE_NATURALEZA",
]

df = pd.read_csv(INPUT, low_memory=False)
model_df = df[FEATURES + [TARGET]].dropna().copy()

X = model_df[FEATURES]
y = model_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), FEATURES)]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = pd.DataFrame({
    "Métrica": ["MAE", "MSE", "RMSE", "R2"],
    "Valor": [
        mean_absolute_error(y_test, y_pred),
        mean_squared_error(y_test, y_pred),
        np.sqrt(mean_squared_error(y_test, y_pred)),
        r2_score(y_test, y_pred),
    ]
})
metrics.to_csv(OUT / "metricas_modelo.csv", index=False)

# OLS para diagnóstico
X_dummies = pd.get_dummies(model_df[FEATURES], drop_first=True, dtype=float)
X_ols = sm.add_constant(X_dummies)
ols = sm.OLS(y, X_ols).fit()

bp = het_breuschpagan(ols.resid, X_ols)
bp_results = pd.DataFrame({
    "prueba": ["LM Statistic", "LM p-value", "F Statistic", "F p-value"],
    "valor": bp
})
bp_results.to_csv(OUT / "breusch_pagan.csv", index=False)

coef = pd.DataFrame({
    "coeficiente": ols.params,
    "p_value": ols.pvalues,
})
coef.to_csv(OUT / "coeficientes_ols.csv")

print(metrics)
print("\nR2 OLS:", ols.rsquared)
print("\nBreusch-Pagan:")
print(bp_results)

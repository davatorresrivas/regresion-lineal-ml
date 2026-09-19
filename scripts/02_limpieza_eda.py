"""
02_limpieza_eda.py
Limpieza y análisis exploratorio de las variables seleccionadas.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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
selected = df[FEATURES + [TARGET]].copy()
model_df = selected.dropna().copy()

print("Original:", selected.shape)
print("Después de casos completos:", model_df.shape)

model_df[TARGET].describe().to_csv(OUT / "estadisticos_punt_global.csv")

for variable in FEATURES:
    summary = model_df.groupby(variable)[TARGET].agg(["count", "mean", "median", "std"])
    summary.to_csv(OUT / f"eda_{variable}.csv")

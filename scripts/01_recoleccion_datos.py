"""
01_recoleccion_datos.py
Descarga el dataset Saber 11° 2020-1 desde la API de Datos Abiertos Colombia.
"""

from pathlib import Path
import pandas as pd

API_URL = "https://www.datos.gov.co/resource/a8xr-en99.json?$limit=50000"
OUTPUT = Path("../data/Saber_11_2020-1.csv")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_json(API_URL)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"Datos descargados: {df.shape}")
print(f"Archivo guardado en: {OUTPUT}")

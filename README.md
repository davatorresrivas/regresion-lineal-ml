# Regresión Lineal – Saber 11° 2020-1

Proyecto académico para construir y evaluar un modelo de Regresión Lineal utilizando datos abiertos del ICFES.

## Objetivo

Estimar el `punt_global` utilizando cuatro variables categóricas:

- `fami_estratovivienda`
- `fami_personas hogar`
- `estu_genero`
- `cole_naturaleza`

El modelo se utiliza con fines predictivos y descriptivos; no se interpretan los resultados como relaciones causales.

## Fuente de datos

Datos Abiertos Colombia – ICFES, conjunto **Saber 11° 2020-1**.

Dataset ID: `a8xr-en99`

Página oficial:
https://www.datos.gov.co/en/en/Education/Saber-11-2020-1/a8xr-en99

API:
https://www.datos.gov.co/resource/a8xr-en99.json

## Estructura

- `notebooks/regresion_lineal_saber11.ipynb`: análisis completo.
- `scripts/01_recoleccion_datos.py`: descarga desde API.
- `scripts/02_limpieza_eda.py`: limpieza y EDA.
- `scripts/03_modelo_regresion.py`: modelo, métricas y diagnóstico.
- `requirements.txt`: dependencias.
- `data/`: datos locales, si se utilizan.
- `resultados/`: archivos generados por los scripts.

## Ejecución

```bash
pip install -r requirements.txt
python scripts/01_recoleccion_datos.py
python scripts/02_limpieza_eda.py
python scripts/03_modelo_regresion.py
```

También se puede abrir el notebook con Jupyter o VS Code.

## Reproducibilidad

La partición entrenamiento/prueba utiliza `random_state=42`.

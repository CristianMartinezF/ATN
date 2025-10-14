# Ajuste de Curvas (Streamlit)

Modelos: Lineal, Exponencial, Geométrica, Hiperbólico, Polinómica (≤4), Media Móvil, Exponencial Asintótico, Logístico, Logarítmico y Potencial.

## Pasos rápidos

```powershell
# 1) Crear y activar venv (Windows PowerShell)
py -3 -m venv .venv
. .venv\Scripts\Activate.ps1

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Ejecutar la app
streamlit run app.py
```

Abrirá `http://localhost:8501`. Carga `data/ejemplo.csv` o tu propio CSV (con columnas `x,y`).

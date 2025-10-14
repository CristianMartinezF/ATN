
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# =============== Utilidades =================

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return {"R": np.nan, "R2": np.nan}
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # correlación de Pearson (R)
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        R = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        R = np.nan
  
    return {"R": R, "R2": r2}

def format_coeff(val):
    try:
        return f"{val:.6g}"
    except Exception:
        return str(val)

def add_model_row(rows, name, eq, yhat, priority=False):
    m = compute_metrics(rows['_y'], yhat)
    rows['resultados'].append({
        "Modelo": name,
        "Ecuación": eq,
        "R": m["R"],
        "R²": m["R2"],
        "_yhat": yhat,
        "_prio": priority
    })

# =============== Modelos lineales / transformaciones =================

def fit_linear(x, y):
    c = np.polyfit(x, y, 1)  # [b, a] para a + b x
    a = c[1]; b = c[0]
    yhat = a + b * x
    eq = f"y = {format_coeff(a)} + {format_coeff(b)}·x"
    return yhat, eq

def fit_logarithmic(x, y):
    mask = x > 0
    if np.sum(mask) < 2: 
        return None, "x <= 0; no se puede logarítmico"
    X = np.log(x[mask]); Y = y[mask]
    b, a = np.polyfit(X, Y, 1)  # Y = a + b ln x
    yhat_full = np.full_like(y, np.nan, dtype=float)
    yhat_full[mask] = a + b * np.log(x[mask])
    eq = f"y = {format_coeff(a)} + {format_coeff(b)}·ln(x)"
    return yhat_full, eq

def fit_potential(x, y):
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 2:
        return None, "x,y > 0 requeridos para potencial"
    X = np.log(x[mask]); Y = np.log(y[mask])
    b, ln_a = np.polyfit(X, Y, 1)
    a = np.exp(ln_a)
    yhat = a * (x ** b)
    eq = f"y = {format_coeff(a)}·x^{format_coeff(b)}"
    return yhat, eq

def fit_exponential(x, y):
    # y = a e^{b x}  => ln y = ln a + b x
    mask = y > 0
    if np.sum(mask) < 2:
        return None, "y > 0 requerido para exponencial"
    Y = np.log(y[mask]); X = x[mask]
    b, ln_a = np.polyfit(X, Y, 1)
    a = np.exp(ln_a)
    yhat = a * np.exp(b * x)
    eq = f"y = {format_coeff(a)}·e^({format_coeff(b)}·x)"
    return yhat, eq

def fit_geometric(x, y):
    # Geométrica: y = a·b^x  => ln y = ln a + x ln b
    mask = y > 0
    if np.sum(mask) < 2:
        return None, "y > 0 requerido para geométrica"
    Y = np.log(y[mask]); X = x[mask]
    ln_b, ln_a = np.polyfit(X, Y, 1)
    a = np.exp(ln_a)
    b = np.exp(ln_b)
    yhat = a * (b ** x)
    eq = f"y = {format_coeff(a)}·({format_coeff(b)})^x"
    return yhat, eq

def fit_hyperbolic(x, y):
    # y = a + b/x  => Y = a + b*U con U=1/x (x!=0)
    mask = x != 0
    if np.sum(mask) < 2:
        return None, "x ≠ 0 requerido para hiperbólico"
    U = 1 / x[mask]; Y = y[mask]
    b, a = np.polyfit(U, Y, 1)
    yhat = a + b / x
    eq = f"y = {format_coeff(a)} + {format_coeff(b)}/x"
    return yhat, eq

def fit_polynomial(x, y, degree=2):
    degree = int(degree)
    degree = max(1, min(4, degree))
    coefs = np.polyfit(x, y, degree)  # p[0] x^n + ... + p[n]
    p = np.poly1d(coefs)
    yhat = p(x)
    # construir ecuación legible
    terms = []
    for i, c in enumerate(coefs):
        pow_ = degree - i
        if pow_ == 0:
            terms.append(f"{format_coeff(c)}")
        elif pow_ == 1:
            terms.append(f"{format_coeff(c)}·x")
        else:
            terms.append(f"{format_coeff(c)}·x^{pow_}")
    eq = "y = " + " + ".join(terms)
    return yhat, eq

def moving_average(x, y, k=3):
    k = int(k)
    k = max(2, k)
    # Suavizado centrado; rellenamos bordes con NaN para mantener longitud
    w = np.ones(k) / k
    y_valid = np.convolve(y, w, mode="valid")
    pad = (len(y) - len(y_valid)) // 2
    yhat = np.full_like(y, np.nan, dtype=float)
    yhat[pad:pad+len(y_valid)] = y_valid
    # ecuación descriptiva
    eq = f"Media móvil (k={k})"
    return yhat, eq

# =============== Modelos no lineales (curve_fit) =================

def model_exp_asym(x, a, b, c):
    # y = c - a * exp(-b x)  (asintota superior c)
    return c - a * np.exp(-b * x)

def fit_exp_asymptotic(x, y):
    # Guess: c ~ max(y), a ~ c - y0, b ~ 0.1
    c0 = np.nanmax(y)
    a0 = max(c0 - y[0], 1e-6)
    b0 = 0.1
    try:
        popt, _ = curve_fit(model_exp_asym, x, y, p0=[a0, b0, c0], maxfev=20000)
        a, b, c = popt
        yhat = model_exp_asym(x, a, b, c)
        eq = f"y = {format_coeff(c)} - {format_coeff(a)}·e^(-{format_coeff(b)}·x)"
        return yhat, eq
    except Exception as e:
        return None, f"No converge: {e}"

def model_logistic(x, L, A, k):
    # y = L / (1 + A·e^{-k x})
    return L / (1.0 + A * np.exp(-k * x))

def fit_logistic(x, y):
    # Guess: L ~ max(y), A ~ (L/y0)-1 si y0>0, k ~ 0.1
    L0 = np.nanmax(y)
    y0 = y[0] if y[0] > 0 else max(np.nanmean(y), 1e-3)
    A0 = max(L0 / y0 - 1.0, 0.1)
    k0 = 0.1
    try:
        popt, _ = curve_fit(model_logistic, x, y, p0=[L0, A0, k0], maxfev=20000)
        L, A, k = popt
        yhat = model_logistic(x, L, A, k)
        eq = f"y = {format_coeff(L)} / (1 + {format_coeff(A)}·e^(-{format_coeff(k)}·x))"
        return yhat, eq
    except Exception as e:
        return None, f"No converge: {e}"

# =============== UI =================

st.set_page_config(page_title="Ajuste de Curvas", layout="wide")

st.title("Ajuste de Curvas (Proyecto)")
st.caption("Modelos: Lineal, Exponencial, Geométrica, Hiperbólico, Polinómica (≤4), Media Móvil, Exponencial Asintótico, Logístico, Logarítmico y Potencial.")

with st.sidebar:
    st.header("Navegación")
    pagina = st.radio("Ir a:", ["Aplicación", "Acerca de"], index=0)

if pagina == "Acerca de":
    st.subheader("Acerca de")
    st.markdown("""
**Versión:** 1.0  
**Aplicación:** ATN — Ajuste de Curvas (Streamlit)  
**Autores:** Cristian Martinez y Karolay Mizzar              
**Asignatura:** Análisis de Técnicas Numéricas               
**Universidad:** CECAR          
**Repositorio:** *[https://github.com/CristianMartinezF/ATN.git]*

Esta herramienta permite evaluar 10 modelos de ajuste de curvas y reporta **R** y **R²** según la rúbrica del curso.  
Carga tus datos `x,y`, selecciona modelos y compara visualmente las curvas (opción de **superponer**).
    """)
    st.stop()  # No ejecutar el resto si estamos en "Acerca de"





with st.sidebar:
    st.header("1) Datos")
    up = st.file_uploader("Sube un CSV con columnas x,y", type=["csv"])
    if st.button("Cargar ejemplo"):
        up = "data/ejemplo.csv"
    degree = st.slider("Grado polinómico (1–4)", 1, 4, 2)
    k_ma = st.slider("k media móvil (≥2)", 2, 20, 3)
    st.header("2) Modelos a evaluar")
    modelos = st.multiselect(
        "Selecciona uno o varios",
        [
            "Lineal", "Exponencial", "Geométrica", "Hiperbólico",
            "Polinómica", "Media Móvil", "Exponencial Asintótico",
            "Logístico", "Logarítmico", "Potencial"
        ],
        default=["Lineal","Polinómica","Exponencial"]
    )
    overlay = st.checkbox("Superponer todas las curvas en una sola gráfica", value=True)

# Cargar datos
if up is None:
    st.warning("Sube un CSV o usa el ejemplo desde la barra lateral.")
    st.stop()

if isinstance(up, str):
    df = pd.read_csv(up)
else:
    df = pd.read_csv(up)

if not set(['x','y']).issubset(df.columns):
    st.error("El CSV debe tener columnas 'x' y 'y'.")
    st.stop()

df = df[['x','y']].dropna()
df = df.sort_values('x')
x = df['x'].to_numpy(dtype=float)
y = df['y'].to_numpy(dtype=float)

st.subheader("Datos")
st.write(df.head())

# Resultados contenedor
rows = {"_y": y, "resultados": []}

# Ajustes seleccionados
if "Lineal" in modelos:
    yhat, eq = fit_linear(x, y)
    add_model_row(rows, "Lineal", eq, yhat, priority=True)

if "Logarítmico" in modelos:
    yhat, eq = fit_logarithmic(x, y)
    if yhat is not None: add_model_row(rows, "Logarítmico", eq, yhat)

if "Potencial" in modelos:
    yhat, eq = fit_potential(x, y)
    if yhat is not None: add_model_row(rows, "Potencial", eq, yhat)

if "Exponencial" in modelos:
    yhat, eq = fit_exponential(x, y)
    if yhat is not None: add_model_row(rows, "Exponencial (a·e^{bx})", eq, yhat)

if "Geométrica" in modelos:
    yhat, eq = fit_geometric(x, y)
    if yhat is not None: add_model_row(rows, "Geométrica (a·b^x)", eq, yhat)

if "Hiperbólico" in modelos:
    yhat, eq = fit_hyperbolic(x, y)
    if yhat is not None: add_model_row(rows, "Hiperbólico (a + b/x)", eq, yhat)

if "Polinómica" in modelos:
    yhat, eq = fit_polynomial(x, y, degree=degree)
    add_model_row(rows, f"Polinómica (grado {degree})", eq, yhat)

if "Media Móvil" in modelos:
    yhat, eq = moving_average(x, y, k=k_ma)
    add_model_row(rows, "Media Móvil", eq, yhat)

if "Exponencial Asintótico" in modelos:
    yhat, eq = fit_exp_asymptotic(x, y)
    if yhat is not None: add_model_row(rows, "Exponencial Asintótico", eq, yhat)

if "Logístico" in modelos:
    yhat, eq = fit_logistic(x, y)
    if yhat is not None: add_model_row(rows, "Logístico", eq, yhat)

# Tabla de métricas
if len(rows['resultados']) == 0:
    st.error("Ningún modelo pudo ajustarse con estos datos/selecciones.")
    st.stop()

tabla = pd.DataFrame([
    {
        "Modelo": r["Modelo"],
        "Ecuación": r["Ecuación"],
        "R": r["R"],
        "R²": r["R²"]
    } for r in rows["resultados"]
])

tabla = tabla.sort_values(by=["R²"], ascending=[False])
st.subheader("Métricas (R, R²)")
st.dataframe(tabla, use_container_width=True)

# Mostrar ecuaciones limpias
with st.expander("Ecuaciones de los modelos"):
    for r in rows["resultados"]:
        st.write(f"**{r['Modelo']}**: {r['Ecuación']}")

# Gráficas
st.subheader("Gráficas")
fig = plt.figure()
plt.scatter(x, y, label="Datos", s=20)

# colores por defecto; Streamlit renderiza bien
if overlay:
    for r in rows["resultados"]:
        yhat = r["_yhat"]
        if yhat is None or np.all(np.isnan(yhat)): 
            continue
        plt.plot(x, yhat, label=r["Modelo"])
    plt.xlabel("x"); plt.ylabel("y"); plt.title("Comparación de modelos"); plt.grid(True); plt.legend()
    st.pyplot(fig)
else:
    st.pyplot(fig)
    # gráficas individuales
    for r in rows["resultados"]:
        yhat = r["_yhat"]
        if yhat is None or np.all(np.isnan(yhat)): 
            continue
        fig_i = plt.figure()
        plt.scatter(x, y, label="Datos", s=20)
        plt.plot(x, yhat, label=r["Modelo"])

        plt.xlabel("x"); plt.ylabel("y"); plt.title(r["Modelo"]); plt.grid(True); plt.legend()
        st.pyplot(fig_i)

st.success("Listo. Ajustes calculados y graficados.")

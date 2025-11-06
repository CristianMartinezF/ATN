# Manual de Usuario — ATN (Ajuste de Curvas)

**Versión:** 1.0  
**Aplicación:** ATN — Ajuste de Curvas (Streamlit)  
**Autores:** Cristian Martinez, Karolay Mizzar y Alejandro Chavez  
**Asignatura:** Análisis de Técnicas Numéricas  
**Universidad:** CECAR 
**Repositorio:** *[https://github.com/CristianMartinezF/ATN.git]*

---

## 1. Requisitos

- **Windows 10/11** (recomendado) o Linux/Mac con Python 3.11+
- **Python 3.11 o superior**
- Permisos para crear y activar entornos virtuales

> Verifica Python con:  
> `py --version` (Windows) o `python3 --version` (Linux/Mac)

---

## 2. Instalación (Windows, PowerShell)

1. **Clonar o descargar** el repositorio ATN:
   ```powershell
   cd C:\Users\USUARIO
   git clone https://github.com/CristianMartinezF/ATN.git
   cd ATN
   ```
   > Si no usas Git, descarga el ZIP del repositorio y descomprímelo.

2. **Crear y activar** el entorno virtual:
   ```powershell
   py -3 -m venv .venv
   . .venv\Scripts\Activate.ps1
   ```
   > Si ves error de ejecución, corre una sola vez:  
   > `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

3. **Instalar dependencias**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación**:
   ```powershell
   streamlit run app.py
   ```
   Se abrirá en el navegador en `http://localhost:8501`.

---

## 3. Estructura de archivos

```
ATN/
  app.py                # Aplicación Streamlit
  requirements.txt      # Dependencias
  .gitignore
  data/
    ejemplo.csv         # Dataset de ejemplo (x,y)
  docs/
  MANUAL_USUARIO.md     # Este manual
  README.md             # Resumen del proyecto
```

---

## 4. Uso de la aplicación

1. **Carga de datos**
   - En la barra lateral, sube un archivo **CSV** con dos columnas: `x,y` (con minúsculas).
   - También puedes cargar el ejemplo (`data/ejemplo.csv`).

2. **Selección de modelos**
   - Marca uno o varios de los **10** modelos:  
     Lineal, Exponencial, Geométrica, Hiperbólico, Polinómica (≤4), Media Móvil, Exponencial Asintótico, Logístico, Logarítmico y Potencial.
   - Ajusta **grado** (polinómica) y **k** (media móvil) si aplica.
   - Activa **“Superponer todas las curvas”** para verlas en una sola gráfica.

3. **Resultados**
   - Tabla de **métricas** con **R** y **R²** (según la rúbrica).  
   - Despliegue de **ecuaciones** por modelo.  
   - **Gráficas**: nube de puntos + curvas ajustadas (con leyenda).

4. **Interpretación básica**
   - **R²** cerca de **1** ⇒ mejor ajuste (proporción de varianza explicada).  
   - **R** es la correlación de Pearson entre valores observados y predichos; cercano a ±1 indica relación fuerte (el signo indica dirección).  
   - Selecciona el modelo con **mayor R²** y **forma coherente** con los datos (p. ej., saturación → logístico).

---

## 5. Límites y validaciones por modelo

- **Logarítmico**: requiere **x > 0**.  
- **Potencial**: requiere **x > 0, y > 0**.  
- **Exponencial/Geométrica**: requieren **y > 0**.  
- **Hiperbólico**: requiere **x ≠ 0**.  
- **Logístico/Exponencial Asintótico**: ajustes no lineales; pueden **no converger** con ciertos datos (la app lo indica).  
- **Media Móvil**: suaviza; no extrapola fuera del rango.

---

## 6. Solución de problemas

- **No abre el navegador**: entra manualmente a `http://localhost:8501`.  
- **Error al activar venv**:  
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` y repite la activación.  
- **pip falla**:  
  `py -m pip install --upgrade pip` y vuelve a instalar.  
- **No converge (modelos no lineales)**: prueba otros datos o desmarca el modelo.

---


## 8. Créditos y licencia

- Código para fines académicos en la asignatura **Análisis de Técnicas Numéricas**.  
- Licencia recomendada: **MIT**.

---

## 9. Contacto

- Autores: *Cristian Martinez; Karolay Mizzar*; Alejandro Chavez   Cristian.martinezf@cecar.edu.co; Karolay.mizzar@cecar.edu.co; Alejandro.chavez@cecar.edu.co
- Docente: *Carlos Cohen*

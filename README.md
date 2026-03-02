<div align="center">

# 🧠 PSS-14 Stress Predictor

**Sistema inteligente de evaluacion y prediccion de estres percibido**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6600)](https://xgboost.readthedocs.io)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore-FFCA28?logo=firebase)](https://firebase.google.com)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render)](https://render.com)

[Demo en Vivo](https://estres-pss14.onrender.com) · [Panel Admin](https://estres-pss14.onrender.com/admin)

</div>

---

## Descripcion

Sistema web completo para la evaluacion del estres percibido basado en la escala **PSS-14** (Perceived Stress Scale de Cohen, Kamarck & Mermelstein, 1983). Combina un cuestionario clinicamente validado con un modelo de **Machine Learning** (XGBoost) que alcanza un **MAE de 0.51** y un **F1-Score de 96.3%**, junto con analisis personalizados generados por **GPT-4**.

Desarrollado como proyecto de **Capstone** para demostrar la aplicacion de inteligencia artificial en el ambito de la salud mental.

---

## Caracteristicas principales

### Evaluacion Clinica
- Cuestionario PSS-14 completo con 14 items validados
- Scoring automatico con inversion de items (4, 5, 6, 7, 9, 10, 13)
- Clasificacion en tres niveles: **Bajo**, **Moderado** y **Alto** estres
- Subescalas: Indefension Percibida y Autoeficacia Percibida

### Machine Learning
- **Modelo ganador**: XGBoost (seleccionado automaticamente entre 4 modelos)
- **Benchmarking**: RandomForest, GradientBoosting, XGBoost, LightGBM
- **Optimizacion**: HalvingGridSearchCV con validacion cruzada estratificada (5-fold)
- **Feature Engineering**: 10+ features derivadas (coherence score, cuadraticos, ratios)
- **Interpretabilidad**: SHAP values para explicar cada prediccion
- **Metricas**: MAE 0.51 | Precision 96.8% | Recall 96.3% | F1 96.3%

### Analisis con GPT-4
- Interpretacion personalizada del perfil de estres
- Recomendaciones basadas en evidencia segun nivel y patron de respuestas
- Analisis contextualizado por edad y profesion
- Evaluacion de coherencia interna de las respuestas

### Panel de Administracion
- Login seguro con bcrypt
- Dashboard con estadisticas en tiempo real
- Metricas del modelo ML (MAE, Precision, Recall, F1)
- Feature Importance y SHAP analysis
- Tabla de tests recientes
- Reentrenamiento del modelo desde el panel

---

## Stack Tecnologico

### Backend
| Tecnologia | Uso |
|------------|-----|
| **Python 3.11** | Lenguaje principal |
| **Flask** | Framework web + API REST |
| **XGBoost** | Modelo predictivo principal |
| **LightGBM** | Modelo candidato en benchmarking |
| **scikit-learn** | Pipeline ML, preprocesamiento, evaluacion |
| **SHAP** | Interpretabilidad del modelo |
| **OpenAI GPT-4** | Analisis personalizado de resultados |
| **Firebase Firestore** | Base de datos en produccion |
| **SQLite** | Base de datos local (desarrollo) |
| **bcrypt** | Hashing de contrasenas (admin) |
| **Gunicorn** | Servidor WSGI para produccion |

### Frontend
| Tecnologia | Uso |
|------------|-----|
| **React 19** | Framework UI |
| **Vite** | Build tool |
| **Tailwind CSS** | Estilos |
| **Recharts** | Graficos de resultados |
| **Lucide React** | Iconos |
| **Axios** | HTTP client |
| **React Router** | Navegacion SPA |

### Infraestructura
| Servicio | Uso |
|----------|-----|
| **Render** | Hosting del backend + frontend |
| **Firebase** | Base de datos Firestore |
| **GitHub** | Control de versiones |

---

## Estructura del proyecto

```
Estres/
├── backend/
│   ├── app.py                  # Servidor Flask + rutas API
│   ├── ml_extension.py         # Motor ML (StressPredictor)
│   ├── gpt_integration.py      # Integracion con GPT-4
│   ├── pss_scoring.py          # Scoring PSS-14 validado
│   ├── database.py             # Capa de datos SQLite
│   ├── firebase_config.py      # Capa de datos Firebase
│   ├── admin.py                # Panel de administracion (API)
│   ├── csv_handler.py          # Exportacion de datos
│   ├── models.py               # Modelos de datos
│   ├── requirements.txt        # Dependencias Python
│   └── variable.env.example    # Template de variables de entorno
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Router principal
│   │   ├── components/
│   │   │   ├── StressForm.jsx       # Formulario PSS-14
│   │   │   ├── ResultsDashboard.jsx # Dashboard de resultados
│   │   │   └── AdminPanel.jsx       # Panel de administracion
│   │   └── services/
│   │       └── api.js          # Cliente API
│   ├── package.json
│   └── vite.config.js
├── modelos/
│   ├── stress_predictor.joblib # Modelo entrenado
│   ├── preprocessor.joblib     # Preprocesador
│   └── feature_importance.csv  # Importancia de features
├── datos/
│   └── respuestas.csv          # Datos de respuestas
├── render.yaml                 # Configuracion de deploy
└── .gitignore
```

---

## Instalacion local

### Requisitos previos
- Python 3.11+
- Node.js 18+
- Cuenta de OpenAI (API key)
- *(Opcional)* Proyecto en Firebase

### 1. Clonar el repositorio
```bash
git clone https://github.com/zeytx/Estres-PSS14.git
cd Estres-PSS14
```

### 2. Configurar el backend
```bash
cd backend
pip install -r requirements.txt

# Crear archivo de variables de entorno
cp variable.env.example variable.env
# Editar variable.env con tus credenciales:
#   OPENAI_API_KEY=tu-api-key
#   ADMIN_USERNAME=tu-usuario
#   ADMIN_PASSWORD=tu-contrasena
```

### 3. Configurar el frontend
```bash
cd ../frontend
npm install
npm run build
```

### 4. Ejecutar
```bash
cd ../backend
python app.py
```

La aplicacion estara disponible en `http://localhost:5000`

---

## Modelo de Machine Learning

### Pipeline de entrenamiento

```
Datos (SQLite/Firebase)
    ↓
Limpieza y validacion
    ↓
Feature Engineering (16 features base + 10 derivadas)
    ↓
Benchmarking (RF, GB, XGBoost, LightGBM)
    ↓
HalvingGridSearchCV (optimizacion de hiperparametros)
    ↓
Evaluacion (StratifiedKFold 5-fold)
    ↓
Modelo ganador → stress_predictor.joblib
```

### Metricas del modelo actual

| Metrica | Valor |
|---------|-------|
| MAE (Test) | **0.51** |
| MAE CV (5-fold) | 0.71 ± 0.12 |
| Precision | **96.8%** |
| Recall | **96.3%** |
| F1 Score | **96.3%** |
| Modelo | XGBRegressor |
| Muestras | 135 |

### Top 5 Features (importancia)

| Feature | Importancia | Descripcion |
|---------|------------|-------------|
| perceived_self_efficacy | 44.1% | Subescala de autoeficacia |
| efficacy_squared | 22.7% | Efecto no lineal de autoeficacia |
| perceived_helplessness | 12.5% | Subescala de indefension |
| helplessness_squared | 8.4% | Efecto no lineal de indefension |
| coherence_score | 5.2% | Coherencia entre subescalas |

---

## Deploy en produccion

El proyecto esta configurado para deploy automatico en **Render** con Firebase Firestore como base de datos.

### Variables de entorno requeridas

| Variable | Descripcion |
|----------|-------------|
| `OPENAI_API_KEY` | API key de OpenAI |
| `ADMIN_USERNAME` | Usuario del panel admin |
| `ADMIN_PASSWORD` | Contrasena del panel admin |
| `FIREBASE_CREDENTIALS_JSON` | JSON de credenciales de Firebase (una linea) |
| `PYTHON_VERSION` | `3.11.6` |

---

## API Endpoints

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| `POST` | `/api/submit-test` | Enviar respuestas del test |
| `GET` | `/api/test-results/:id` | Obtener resultados (requiere token) |
| `POST` | `/api/predict-stress` | Prediccion ML directa |
| `GET` | `/api/get-analysis/:id` | Analisis GPT de un test |
| `POST` | `/api/admin/login` | Login de administrador |
| `GET` | `/api/admin/stats` | Estadisticas generales |
| `GET` | `/api/admin/model-info` | Info del modelo ML |
| `POST` | `/api/admin/retrain` | Reentrenar modelo |

---

## Autor

**Alvaro Nunez Jesus**

- GitHub: [@zeytx](https://github.com/zeytx)

Proyecto desarrollado como parte del curso de Capstone — 2025/2026.

---

## Licencia

Todos los derechos reservados. Este codigo es propiedad intelectual de Alvaro Nunez Jesus y no esta disponible para uso, copia, modificacion o distribucion sin autorizacion explicita por escrito.

© 2025-2026 Alvaro Nunez Jesus

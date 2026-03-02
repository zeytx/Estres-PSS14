import hashlib
import os
import re
import sys
import json
import warnings
import pandas as pd
import numpy as np
import joblib
import sqlite3
import logging
from typing import Dict, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 - needed for HalvingGridSearchCV
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error

# Modelos avanzados
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
from pss_scoring import (
    INVERTED_INDICES, DIRECT_INDICES,
    process_responses as pss_process_responses
)

# Asegurarse de que el directorio de logs exista
os.makedirs('../logs', exist_ok=True)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/stress_predictor.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TTLCache:
    """Caché con TTL (Time To Live) y tamaño máximo con evicción LRU"""
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self._cache = OrderedDict()  # key -> (value, timestamp)

    def get(self, key):
        if key in self._cache:
            value, ts = self._cache[key]
            if datetime.now() - ts < self.ttl:
                # Mover al final (más reciente)
                self._cache.move_to_end(key)
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, datetime.now())
        # Evicción LRU si excede el tamaño
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def __contains__(self, key):
        return self.get(key) is not None

    def __len__(self):
        return len(self._cache)

    def to_dict(self):
        """Serializa para joblib (sin entradas expiradas)"""
        now = datetime.now()
        return {k: (v, ts) for k, (v, ts) in self._cache.items() if now - ts < self.ttl}

    @classmethod
    def from_dict(cls, data, max_size=1000, ttl_hours=24):
        """Reconstruye desde dict serializado"""
        cache = cls(max_size=max_size, ttl_hours=ttl_hours)
        now = datetime.now()
        ttl = timedelta(hours=ttl_hours)
        for k, (v, ts) in data.items():
            if now - ts < ttl:
                cache._cache[k] = (v, ts)
        return cache


class PredictionStrategy(Enum):
    ML_MODEL = 1
    SIMPLE_SUM = 2
    CACHED = 3
    FALLBACK = 4

class FallbackRegressor(BaseEstimator, RegressorMixin):
    """Regresor de fallback que usa la media o mediana"""
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.value = None
        
    def fit(self, X, y):
        if self.strategy == 'mean':
            self.value = np.mean(y)
        else:  # median
            self.value = np.median(y)
        return self
        
    def predict(self, X):
        return np.full(X.shape[0], self.value)

class StressPredictor:
    # Cache de clase para evitar recargas
    _model_cache = None
    _preprocessor_cache = None
    _prediction_cache_memory = None
    
    def __init__(self, db_path: str = '../datos/pss_database.db'):
        """Inicializa el predictor con configuración robusta"""
        self.db_path = db_path
        self.model_dir = '../modelos'
        self.history_dir = os.path.join(self.model_dir, 'historico')
        self.registry_path = os.path.join(self.model_dir, 'model_registry.json')
        self.model_path = os.path.join(self.model_dir, 'stress_predictor.joblib')
        self.preprocessor_path = os.path.join(self.model_dir, 'preprocessor.joblib')
        self.cache_path = os.path.join(self.model_dir, 'prediction_cache.joblib')
        self.min_training_samples = 10  # Mínimo absoluto para entrenar
        self.optimal_training_samples = 100  # Ideal para buen rendimiento
        self.max_history_versions = 5  # Máximo de versiones históricas
        self.last_training_time = None
        self.model = None
        self.preprocessor = None
        self.prediction_cache = TTLCache(max_size=1000, ttl_hours=24)
        self._load_artifacts()
        self._setup_fallback()
        self.stress_thresholds = {
            'low': 28,    # <= 28: Bajo
            'moderate': 42 # <= 42: Moderado
            # > 42: Alto
        }
        
    def _load_artifacts(self) -> None:
        """Carga modelo, preprocesador y caché de forma segura y eficiente"""
        
        # 1. Intentar cargar de memoria (Clase)
        if StressPredictor._model_cache is not None and StressPredictor._preprocessor_cache is not None:
             self.model = StressPredictor._model_cache
             self.preprocessor = StressPredictor._preprocessor_cache
             if StressPredictor._prediction_cache_memory is not None:
                 self.prediction_cache = StressPredictor._prediction_cache_memory
             return

        try:
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(self.history_dir, exist_ok=True)

            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                StressPredictor._model_cache = self.model
                logger.info("Modelo cargado desde archivo")
            
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                StressPredictor._preprocessor_cache = self.preprocessor
                logger.info("Preprocesador cargado desde archivo")
            
            if os.path.exists(self.cache_path):
                try:
                    cache_data = joblib.load(self.cache_path)
                    if isinstance(cache_data, dict) and not isinstance(cache_data, OrderedDict):
                        # Migrar de dict antiguo a TTLCache
                        self.prediction_cache = TTLCache(max_size=1000, ttl_hours=24)
                        for k, v in cache_data.items():
                            if isinstance(v, tuple) and len(v) == 2:
                                self.prediction_cache._cache[k] = v
                            else:
                                self.prediction_cache.set(k, v)
                    else:
                        self.prediction_cache = TTLCache.from_dict(cache_data, max_size=1000, ttl_hours=24)
                    StressPredictor._prediction_cache_memory = self.prediction_cache
                    logger.info(f"Caché cargada con {len(self.prediction_cache)} entradas")
                except Exception as ce:
                    logger.warning(f"Error cargando caché, reiniciando: {str(ce)}")
                    self.prediction_cache = TTLCache(max_size=1000, ttl_hours=24)

        except Exception as e:
            logger.error(f"Error cargando artefactos: {str(e)}")
            self._setup_fallback()
    
    def _save_artifacts(self) -> None:
        """Guarda artefactos de forma segura con versionado histórico"""
        try:
            os.makedirs(self.history_dir, exist_ok=True)

            joblib.dump(self.model, self.model_path)
            joblib.dump(self.preprocessor, self.preprocessor_path)

            # Guardar caché serializada
            cache_data = self.prediction_cache.to_dict() if isinstance(self.prediction_cache, TTLCache) else self.prediction_cache
            joblib.dump(cache_data, self.cache_path)

            # Actualizar cache en memoria
            StressPredictor._model_cache = self.model
            StressPredictor._preprocessor_cache = self.preprocessor
            StressPredictor._prediction_cache_memory = self.prediction_cache
        except Exception as e:
            logger.error(f"Error guardando artefactos: {str(e)}")
    
    def _save_model_version(self, metrics: Dict, version_hash: str) -> None:
        """Guarda una versión histórica del modelo con sus métricas"""
        try:
            os.makedirs(self.history_dir, exist_ok=True)

            # Guardar copia del modelo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_model_path = os.path.join(self.history_dir, f'model_{timestamp}.joblib')
            joblib.dump(self.model, version_model_path)

            # Actualizar registro de modelos
            registry = []
            if os.path.exists(self.registry_path):
                try:
                    with open(self.registry_path, 'r') as f:
                        registry = json.load(f)
                except (json.JSONDecodeError, IOError):
                    registry = []

            registry.append({
                'version': version_hash,
                'timestamp': timestamp,
                'model_type': type(self.model).__name__,
                'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in metrics.items()},
                'model_file': f'model_{timestamp}.joblib'
            })

            # Mantener solo las últimas N versiones
            if len(registry) > self.max_history_versions:
                old_entries = registry[:-self.max_history_versions]
                for entry in old_entries:
                    old_path = os.path.join(self.history_dir, entry.get('model_file', ''))
                    if os.path.exists(old_path):
                        os.remove(old_path)
                registry = registry[-self.max_history_versions:]

            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)

            logger.info(f"Versión del modelo guardada: {version_hash} ({len(registry)} versiones en registro)")

        except Exception as e:
            logger.error(f"Error guardando versión histórica: {str(e)}")

    def _setup_fallback(self) -> None:
        """Configura sistemas de fallback"""
        self.fallback_model = FallbackRegressor(strategy='median')
        self.fallback_ready = False
        self.fallback_values = {'mean': 28, 'median': 27}  # Valores por defecto
    
    def _validate_inputs(self, age: int, profession: str, responses: list) -> bool:
        """Valida las entradas antes de procesar"""
        if not isinstance(age, int) or age < 10 or age > 120:
            logger.warning(f"Edad inválida: {age}")
            return False
            
        if not self._is_valid_profession(profession):
            logger.warning(f"Profesión inválida: {profession}")
            return False
            
        if not isinstance(responses, list) or len(responses) != 14:
            logger.warning(f"Respuestas inválidas: {responses}")
            return False
            
        if any(not isinstance(r, int) or r < 0 or r > 4 for r in responses):
            logger.warning(f"Valores de respuesta inválidos: {responses}")
            return False
            
        return True
    
    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara datos para entrenamiento con validación robusta"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Añadir este log para saber cuántos registros hay en total
            count_query = "SELECT COUNT(*) FROM tests"
            total_tests = conn.execute(count_query).fetchone()[0]
            logger.info(f"Total de tests en la base de datos: {total_tests}")
            
            # Consulta con manejo de errores
            tests_query = """
                SELECT test_id, age, profession, total_score 
                FROM tests 
                WHERE age BETWEEN 10 AND 120 
                AND total_score BETWEEN 0 AND 56
            """

            
            tests_df = pd.read_sql(tests_query, conn)
            

            def clean_profession(prof):
                if not isinstance(prof, str):
                    return 'unknown'
                
                # Casos especiales de concatenación
                prof = prof.replace('IngenieroIngeniero', 'Ingeniero')
                prof = prof.replace('ComunicaciónComunicación', 'Comunicación')
                
                # Separar por mayúsculas (camelCase)
                prof = ' '.join(re.findall('[A-Z][^A-Z]*', prof))
                
                # Tomar solo la primera profesión si hay múltiples
                prof = prof.split(',')[0].split(';')[0].split('/')[0]
                
                # Normalización básica
                prof = prof.strip().title()
                
                return prof[:30] if len(prof) >= 2 else 'unknown'

            # Aplicar limpieza de profesiones
            logger.info("Aplicando limpieza de profesiones...")
            tests_df['profession'] = tests_df['profession'].apply(clean_profession)

            tests_df = tests_df[tests_df['profession'] != 'unknown']

            responses_query = """
                SELECT test_id, question_number, processed_value 
                FROM responses 
                WHERE processed_value BETWEEN 0 AND 4
                ORDER BY test_id, question_number
            """
            responses_df = pd.read_sql(responses_query, conn)

            # Añadir estos logs para ver cuántos datos quedan después de los filtros
            logger.info(f"Tests después del filtro: {len(tests_df)}")
            logger.info(f"Respuestas después del filtro: {len(responses_df)}")
            logger.info(f"Respuestas test_ids únicos: {responses_df['test_id'].nunique()}")

            conn.close()
            
            if tests_df.empty or responses_df.empty:
                logger.warning("Datos vacíos en la base de datos")
                return None, None
            
            # Verificar integridad de respuestas por test
            response_counts = responses_df.groupby('test_id').size()
            complete_tests = response_counts[response_counts == 14].index

            logger.info(f"Tests con 14 respuestas completas: {len(complete_tests)}")

            # Pivotear respuestas con validación
            try:
                responses_pivot = responses_df.pivot(
                    index='test_id',
                    columns='question_number',
                    values='processed_value'
                ).add_prefix('q_').fillna(0) # Mantener 0 como valor faltante
                
                # Verificar que tenemos todas las preguntas
                if responses_pivot.shape[1] != 14:
                    logger.warning(f"Faltan preguntas, solo tenemos {responses_pivot.shape[1]}")
                    return None, None
                    
            except Exception as e:
                logger.error(f"Error procesando respuestas: {str(e)}")
                return None, None
            
            # Unir datos con validación de integridad
            df = pd.merge(tests_df, responses_pivot, on='test_id', how='inner')
            if df.empty:
                logger.warning("Merge resultó en dataframe vacío")
                return None, None
                
            # Limpieza final
            df = df.dropna(subset=['age', 'profession'])
            if len(df) < self.min_training_samples:
                logger.warning(f"Datos insuficientes después de limpieza: {len(df)}")
                return None, None

            feature_cols = ['age', 'profession'] + [f'q_{i}' for i in range(1, 15)]    
            X = df.drop(['test_id', 'total_score'], axis=1)
            y = df['total_score']
            
            # === Feature Engineering PSS-14 (centralizado) ===
            X = self._apply_feature_engineering(X)

            logger.info(f"Feature engineering completado. Features totales: {X.shape[1]}")
            logger.info(f"Datos finales - X: {X.shape}, y: {y.shape}")

            profession_counts = X['profession'].value_counts()
            logger.info(f"Top 5 profesiones en datos finales:")
            for prof, count in profession_counts.head().items():
                logger.info(f"  - {prof}: {count} casos")


            # Calcular valores de fallback con manejo seguro de la moda
            try:
                from scipy.stats import mode
                mode_result = mode(y, keepdims=True)
                
                self.fallback_values = {
                    'mean': float(y.mean()),
                    'median': float(y.median()),
                    'mode': float(mode_result.mode[0]) if mode_result.count[0] > 1 else float(y.median())
                }
                self.fallback_ready = True
                logger.info(f"Valores de fallback calculados: {self.fallback_values}")
            except Exception as e:
                logger.error(f"Error calculando valores de fallback: {str(e)}")
                self.fallback_values = {
                    'mean': 28.0,
                    'median': 27.0,
                    'mode': 28.0
                }
                self.fallback_ready = False
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparando datos: {str(e)}")
            return None, None
    
    def _create_optimized_pipeline(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """Crea y optimiza el pipeline usando HalvingGridSearchCV con modelos avanzados"""

        # 1. Definir Preprocesador
        numeric_features = ['age']
        categorical_features = ['profession']
        question_features = [f'q_{i+1}' for i in range(14)]
        # Features derivadas del PSS-14
        engineered_features = self._get_engineered_feature_names()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',
                min_frequency=1,
                sparse_output=False,
            ))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), ['age']),  
                
                ('cat', categorical_transformer, ['profession']), 
                
                ('q', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', RobustScaler(quantile_range=(5, 95)))
                ]), question_features),

                ('eng', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), [f for f in engineered_features if f in X.columns])
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        # 2. Definir modelos y parámetros para búsqueda
        models_params = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True),
                'params': {
                    'regressor__n_estimators': [200, 300, 500],
                    'regressor__max_depth': [8, 12, 18, None],
                    'regressor__min_samples_split': [2, 5, 10],
                    'regressor__min_samples_leaf': [1, 2, 4],
                    'regressor__max_features': ['sqrt', 'log2', 0.5]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(
                    random_state=42,
                    validation_fraction=0.1,
                    n_iter_no_change=15,  # Early stopping
                    tol=1e-4
                ),
                'params': {
                    'regressor__n_estimators': [200, 400],
                    'regressor__learning_rate': [0.01, 0.05, 0.1],
                    'regressor__max_depth': [3, 5, 7],
                    'regressor__min_samples_split': [2, 5],
                    'regressor__subsample': [0.8, 0.9, 1.0]
                }
            }
        }

        # Agregar XGBoost si está disponible
        if HAS_XGBOOST:
            models_params['XGBoost'] = {
                'model': XGBRegressor(
                    random_state=42, n_jobs=-1,
                    eval_metric='mae',
                    early_stopping_rounds=20,
                    verbosity=0
                ),
                'params': {
                    'regressor__n_estimators': [200, 400, 600],
                    'regressor__learning_rate': [0.01, 0.05, 0.1],
                    'regressor__max_depth': [4, 6, 8],
                    'regressor__min_child_weight': [1, 3, 5],
                    'regressor__subsample': [0.8, 0.9],
                    'regressor__colsample_bytree': [0.7, 0.9, 1.0],
                    'regressor__reg_alpha': [0, 0.1],
                    'regressor__reg_lambda': [1, 2]
                }
            }
            logger.info("XGBoost disponible y agregado al benchmark")

        # Agregar LightGBM si está disponible
        if HAS_LIGHTGBM:
            models_params['LightGBM'] = {
                'model': LGBMRegressor(
                    random_state=42, n_jobs=-1,
                    verbose=-1,
                    force_col_wise=True
                ),
                'params': {
                    'regressor__n_estimators': [200, 400, 600],
                    'regressor__learning_rate': [0.01, 0.05, 0.1],
                    'regressor__max_depth': [4, 6, 8, -1],
                    'regressor__num_leaves': [31, 50, 80],
                    'regressor__min_child_samples': [5, 10, 20],
                    'regressor__subsample': [0.8, 0.9, 1.0],
                    'regressor__colsample_bytree': [0.7, 0.9, 1.0],
                    'regressor__reg_alpha': [0, 0.1],
                    'regressor__reg_lambda': [0, 1]
                }
            }
            logger.info("LightGBM disponible y agregado al benchmark")

        best_score = float('-inf')
        best_pipeline = None
        best_model_name = ""

        # Determinar si usar HalvingGridSearchCV (más eficiente para espacios grandes)
        use_halving = len(X) >= 50  # Necesita suficientes datos para halvear

        # 3. Iterar sobre modelos y encontrar el mejor
        # Silenciar warnings de feature names durante GridSearch (especialmente LightGBM)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

            for name, mp in models_params.items():
                logger.info(f"Evaluando modelo: {name}...")

                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', mp['model'])
                ])

                # Para XGBoost, ajustar early_stopping_rounds (no soportado en GridSearch directo)
                search_params = mp['params'].copy()
                if name == 'XGBoost':
                    # Remover early_stopping ya que no funciona con GridSearchCV
                    pipeline.named_steps['regressor'].set_params(early_stopping_rounds=None)

                try:
                    if use_halving and len(search_params) > 3:
                        grid_search = HalvingGridSearchCV(
                            pipeline,
                            search_params,
                            cv=5,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1,
                            verbose=1,
                            factor=3,
                            min_resources='smallest'
                        )
                    else:
                        grid_search = GridSearchCV(
                            pipeline,
                            search_params,
                            cv=5,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1,
                            verbose=1
                        )

                    grid_search.fit(X, y)
                    score = grid_search.best_score_
                    logger.info(f"Mejor score (MAE negativo) para {name}: {score:.4f}")
                    logger.info(f"Mejores parámetros: {grid_search.best_params_}")

                    if score > best_score:
                        best_score = score
                        best_pipeline = grid_search.best_estimator_
                        best_model_name = name

                except Exception as e:
                    logger.error(f"Error optimizando {name}: {str(e)}")

        if best_pipeline is None:
             logger.warning("Fallo en optimización, usando modelo por defecto RF")
             model = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5, 
                min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
             )
             best_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
             ])
             best_model_name = "RandomForestDefault"

        logger.info(f"Modelo ganador: {best_model_name} con CV MAE: {-best_score:.4f}")
        return best_pipeline
    
    def _evaluate_model(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evalúa el modelo con StratifiedKFold y múltiples métricas"""
        try:
            # Discretizar y para stratificación por nivel de estrés
            y_strata = pd.cut(y,
                            bins=[-1, self.stress_thresholds['low'], self.stress_thresholds['moderate'], 57],
                            labels=["Bajo", "Moderado", "Alto"])

            # StratifiedKFold garantiza proporción de cada nivel en cada fold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            mae_scores = []
            for train_idx, val_idx in skf.split(X, y_strata):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                pipeline.fit(X_train_fold, y_train_fold)
                y_pred_fold = pipeline.predict(X_val_fold)
                mae_scores.append(mean_absolute_error(y_val_fold, y_pred_fold))

            avg_mae = np.mean(mae_scores)
            std_mae = np.std(mae_scores)

            # Entrenamiento final y evaluación en conjunto de prueba estratificado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y_strata
            )
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            test_mae = mean_absolute_error(y_test, y_pred)

            # Métricas de clasificación (discretizando)
            y_test_class = pd.cut(y_test,
                                bins=[-1, self.stress_thresholds['low'], self.stress_thresholds['moderate'], 57],
                                labels=["Bajo", "Moderado", "Alto"])
            
            y_pred_class = pd.cut(y_pred, 
                                bins=[-1, self.stress_thresholds['low'], self.stress_thresholds['moderate'], 57],
                                labels=["Bajo", "Moderado", "Alto"])

            # Manejar posibles NaN en las clases
            y_test_class = y_test_class.fillna('Moderado')
            y_pred_class = y_pred_class.fillna('Moderado')

            precision = precision_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
            recall = recall_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
            f1 = f1_score(y_test_class, y_pred_class, average='weighted', zero_division=0)

            logger.info(f"MAE Validación Cruzada: {avg_mae:.2f} (±{std_mae:.2f}), Test: {test_mae:.2f}")
            logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            return {
                'mae': test_mae,
                'mae_cv_mean': avg_mae,
                'mae_cv_std': std_mae,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
        except Exception as e:
            logger.error(f"Error evaluando modelo: {str(e)}")
            return {'mae': float('inf'), 'precision': 0, 'recall': 0, 'f1': 0}

    def train_model(self, force_retrain: bool = False) -> Dict[str, any]:
        """Entrena un nuevo modelo con validación robusta y tracking completo"""

        # EJECUTAR LIMPIEZA DE DATOS PRIMERO
        logger.info("Ejecutando limpieza de datos de profesiones...")
        self.clean_profession_data()


        # Metadata base para logs y resultados
        training_metadata = {
            'timestamp': datetime.now().isoformat(),
            'samples': 0,
            'status': None,
            'metrics': {}
        }

        # 1. Verificar necesidad de reentrenamiento
        if (not force_retrain and self.model is not None and 
            self.last_training_time and 
            (datetime.now() - self.last_training_time) < timedelta(days=7)):
            
            training_metadata.update({
                'status': 'skipped',
                'reason': 'model_is_recent'
            })
            logger.info("ENTRENAMIENTO SKIPPED | Modelo reciente", extra={'metadata': training_metadata})
            return training_metadata

        # 2. Preparar datos
        X, y = self._prepare_data()
        training_metadata['samples'] = len(X) if X is not None else 0

        if X is None or len(X) < self.min_training_samples:
            training_metadata.update({
                'status': 'failed',
                'reason': 'insufficient_data',
                'min_samples_required': self.min_training_samples
            })
            logger.warning("ENTRENAMIENTO FALLIDO | Datos insuficientes", extra={'metadata': training_metadata})
            return training_metadata
        # Validación adicional de tipos de datos
        try:
            X['age'] = pd.to_numeric(X['age'])
            X['profession'] = X['profession'].astype(str)
            for q in [f'q_{i+1}' for i in range(14)]:
                X[q] = pd.to_numeric(X[q])
            logger.info("Validación de tipos de datos exitosa")
        except Exception as e:
            logger.error(f"Error en validación de tipos: {str(e)}")
            training_metadata.update({
                'status': 'failed',
                'reason': 'invalid_data_types',
                'error': str(e)
            })
            return training_metadata

        try:
            # 3. Crear y evaluar pipeline
            pipeline = self._create_optimized_pipeline(X, y)
            metrics = self._evaluate_model(pipeline, X, y)
            training_metadata['metrics'].update(metrics)

            # 4. # Validar rendimiento (MAE < 4.5 puntos ≈ 8% error en escala 0-56, F1 ≥ 0.55)
            if metrics['mae'] >= 4.5 or metrics['f1'] < 0.55:
                training_metadata.update({
                    'status': 'failed',
                    'reason': 'poor_performance'
                })
                logger.warning(f"ENTRENAMIENTO FALLIDO | MAE: {metrics['mae']:.2f}, F1: {metrics['f1']:.2f}")
                return training_metadata

            # 5. Entrenamiento final
            pipeline.fit(X, y)
            self.model = pipeline.named_steps['regressor']
            self.preprocessor = pipeline.named_steps['preprocessor']
            self.last_training_time = datetime.now()

            # 6. Feature importance (con manejo seguro)
            if hasattr(self.model, 'feature_importances_'):
                try:
                    importances = self.model.feature_importances_
                    features = self.preprocessor.get_feature_names_out()
                    top_features = dict(zip(features, importances))
                    training_metadata['top_features'] = dict(
                        sorted(top_features.items(), key=lambda x: -x[1])[:5]
                    )
                    
                    # MANTENER LOGGING DETALLADO EXISTENTE
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    logger.info("Top 10 características más importantes:")
                    for idx, row in importance_df.head(10).iterrows():
                        logger.info(f"{row['Feature']}: {row['Importance']:.4f}")
                    
                    # Guardar en archivo
                    importance_df.to_csv(
                        os.path.join(self.model_dir, 'feature_importance.csv'), 
                        index=False
                    )
                    
                except Exception as e:
                    logger.error(f"Error calculando feature importance: {str(e)}")

            # 6b. SHAP values para interpretabilidad avanzada
            if HAS_SHAP and hasattr(self.model, 'predict'):
                try:
                    X_sample = X.sample(min(100, len(X)), random_state=42)
                    X_transformed = self.preprocessor.transform(X_sample)

                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(X_transformed)

                    # Guardar resumen SHAP
                    feature_names = self.preprocessor.get_feature_names_out()
                    shap_importance = np.abs(shap_values).mean(axis=0)
                    shap_df = pd.DataFrame({
                        'Feature': feature_names[:len(shap_importance)],
                        'SHAP_Importance': shap_importance[:len(feature_names)]
                    }).sort_values('SHAP_Importance', ascending=False)

                    shap_df.to_csv(
                        os.path.join(self.model_dir, 'shap_importance.csv'),
                        index=False
                    )

                    training_metadata['shap_top_features'] = dict(
                        zip(shap_df['Feature'].head(5), shap_df['SHAP_Importance'].head(5))
                    )

                    logger.info("Top 5 SHAP features:")
                    for _, row in shap_df.head(5).iterrows():
                        logger.info(f"  SHAP {row['Feature']}: {row['SHAP_Importance']:.4f}")

                except Exception as e:
                    logger.warning(f"Error calculando SHAP values: {str(e)}")

            # 7. Agregar métricas adicionales del modelo
            if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
                training_metadata['metrics']['oob_score'] = float(self.model.oob_score_)
            
            training_metadata['metrics']['n_estimators'] = getattr(self.model, 'n_estimators', None)

            # 8. Guardar y retornar resultados
            self._save_artifacts()
            model_version = self._generate_version_hash(X, y)
            self._save_model_version(metrics, model_version)

            training_metadata.update({
                'status': 'success',
                'model_version': model_version,
                'model_type': type(self.model).__name__
            })

            # LOGGING MEJORADO CON METADATA
            logger.info(f"ENTRENAMIENTO COMPLETADO | "
                       f"MAE: {metrics['mae']:.2f} | "
                       f"Precision: {metrics['precision']:.3f} | "
                       f"Recall: {metrics['recall']:.3f} | "
                       f"F1: {metrics['f1']:.3f} | "
                       f"Muestras: {len(X)} | "
                       f"Versión: {model_version} | "
                       f"OOB: {training_metadata['metrics'].get('oob_score', 'N/A')}", 
                       extra={'metadata': training_metadata})
            
            return training_metadata

        except Exception as e:
            training_metadata.update({
                'status': 'failed',
                'reason': 'training_error',
                'error': str(e)
            })
            logger.error(f"ENTRENAMIENTO ERROR | {str(e)}", extra={'metadata': training_metadata})
            return training_metadata
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica feature engineering centralizado PSS-14 a un DataFrame.
        Usado tanto en entrenamiento como en predicción para garantizar consistencia.

        Features generadas:
        - perceived_helplessness: Suma de ítems directos (indefensión)
        - perceived_self_efficacy: Suma de ítems invertidos (autoeficacia)
        - helplessness_efficacy_ratio: Ratio entre dimensiones
        - response_variance: Variabilidad interna de respuestas
        - extreme_response_count: Cantidad de respuestas 0 o 4
        - coherence_score: Discrepancia entre dimensiones directa/invertida
        - helplessness_squared: Efecto no-lineal de indefensión
        - efficacy_squared: Efecto no-lineal de autoeficacia
        - age_x_helplessness: Interacción edad × indefensión
        - age_x_efficacy: Interacción edad × autoeficacia
        - max_direct_item: Máximo en ítems directos (pico de estrés)
        - min_inverted_item: Mínimo en ítems invertidos (mayor déficit)
        - high_stress_item_count: Cantidad de ítems con puntuación ≥3 (directos) o ≤1 (invertidos procesados)
        """
        helplessness_cols = ['q_1', 'q_2', 'q_3', 'q_8', 'q_11', 'q_12', 'q_14']
        efficacy_cols = ['q_4', 'q_5', 'q_6', 'q_7', 'q_9', 'q_10', 'q_13']
        q_cols = [f'q_{i}' for i in range(1, 15)]

        # --- Dimensiones PSS-14 base ---
        df['perceived_helplessness'] = df[helplessness_cols].sum(axis=1)
        df['perceived_self_efficacy'] = df[efficacy_cols].sum(axis=1)
        df['helplessness_efficacy_ratio'] = df['perceived_helplessness'] / (df['perceived_self_efficacy'] + 1)

        # --- Consistencia y extremismo ---
        df['response_variance'] = df[q_cols].var(axis=1)
        df['extreme_response_count'] = df[q_cols].apply(
            lambda row: sum(1 for v in row if v in [0, 4]), axis=1
        )

        # --- Coherencia entre dimensiones ---
        avg_direct = df[helplessness_cols].mean(axis=1)
        avg_inverted = df[efficacy_cols].mean(axis=1)
        df['coherence_score'] = abs(avg_direct - (4 - avg_inverted))

        # --- Features avanzadas: no-linealidad ---
        df['helplessness_squared'] = df['perceived_helplessness'] ** 2
        df['efficacy_squared'] = df['perceived_self_efficacy'] ** 2

        # --- Interacciones edad × dimensiones ---
        if 'age' in df.columns:
            df['age_x_helplessness'] = df['age'] * df['perceived_helplessness']
            df['age_x_efficacy'] = df['age'] * df['perceived_self_efficacy']

        # --- Picos de estrés por ítem ---
        df['max_direct_item'] = df[helplessness_cols].max(axis=1)
        df['min_inverted_item'] = df[efficacy_cols].min(axis=1)

        # --- Conteo de ítems en zona de alto estrés ---
        def count_high_stress_items(row):
            count = 0
            for col in helplessness_cols:
                if row[col] >= 3:
                    count += 1
            for col in efficacy_cols:
                if row[col] <= 1:  # Tras inversión, valor bajo = alto estrés
                    count += 1
            return count

        df['high_stress_item_count'] = df.apply(count_high_stress_items, axis=1)

        return df

    def _get_engineered_feature_names(self) -> list:
        """Retorna los nombres de las features de ingeniería (debe coincidir con _apply_feature_engineering)"""
        return [
            'perceived_helplessness', 'perceived_self_efficacy',
            'helplessness_efficacy_ratio', 'response_variance',
            'extreme_response_count', 'coherence_score',
            'helplessness_squared', 'efficacy_squared',
            'age_x_helplessness', 'age_x_efficacy',
            'max_direct_item', 'min_inverted_item',
            'high_stress_item_count'
        ]

    def _get_cache_key(self, age: int, profession: str, responses: list) -> str:
        """Genera una clave única y determinista para el caché"""
        key_data = f"{age}|{profession}|{'_'.join(map(str, responses))}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _determine_stress_level(self, score: float) -> str:
        """Determina el nivel de estrés con umbrales configurables"""
        score = round(score)
        
        # Umbrales configurables
        low_threshold = 28
        moderate_threshold = 42
        
        if score <= low_threshold:
            return "Bajo"
        elif score <= moderate_threshold:
            return "Moderado"
        else:
            return "Alto"
    
    def predict_stress(self, age: int, profession: str, responses: list) -> Dict:
        """Predice el nivel de estrés con múltiples estrategias de fallback"""
        start_time = datetime.now()
        
        # Validación de entrada
        if not self._validate_inputs(age, profession, responses):
            logger.warning("Entradas inválidas, usando fallback")
            return self._fallback_prediction(responses, PredictionStrategy.FALLBACK)
        
        # Procesar respuestas usando módulo centralizado
        processed_responses = pss_process_responses(responses)
        total_score = sum(processed_responses)
        
        # Verificar caché primero
        cache_key = self._get_cache_key(age, profession, responses)
        cached_result = self.prediction_cache.get(cache_key) if isinstance(self.prediction_cache, TTLCache) else self.prediction_cache.get(cache_key)
        if cached_result is not None:
            logger.info("Predicción encontrada en caché")
            result = cached_result
            result['strategy'] = PredictionStrategy.CACHED.name
            result['response_time_ms'] = (datetime.now() - start_time).total_seconds() * 1000
            return result
        
        # Intentar con el modelo ML si está disponible
        if self.model is not None and self.preprocessor is not None:
            try:
                # Crear DataFrame con features base
                input_data = pd.DataFrame({
                    'age': [age],
                    'profession': [profession],
                    **{f'q_{i+1}': [resp] for i, resp in enumerate(processed_responses)}
                })
                
                # Feature Engineering PSS-14 (centralizado — misma lógica que entrenamiento)
                input_data = self._apply_feature_engineering(input_data)

                X_processed = self.preprocessor.transform(input_data)
                predicted_score = self.model.predict(X_processed)[0]
                predicted_score = np.clip(predicted_score, 0, 56)

                # Calcular confianza con datos reales
                confidence = self._calculate_confidence(predicted_score, processed_responses, age, profession)

                result = {
                    'predicted_score': float(predicted_score),
                    'predicted_stress_level': self._determine_stress_level(predicted_score),
                    'confidence': confidence,  #Nueva métrica
                    'confidence_interpretation': self._interpret_confidence(confidence),
                    'strategy': PredictionStrategy.ML_MODEL.name,
                    'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
                }
                
                # Almacenar en caché
                if isinstance(self.prediction_cache, TTLCache):
                    self.prediction_cache.set(cache_key, result)
                else:
                    self.prediction_cache[cache_key] = result

                # Guardar caché a disco periódicamente (no en cada predicción)
                cache_size = len(self.prediction_cache) if isinstance(self.prediction_cache, TTLCache) else len(self.prediction_cache)
                if cache_size % 25 == 0:
                    self._save_artifacts()
                self._log_prediction_analytics(result, start_time, age, profession, responses)
                return result
                
            except Exception as e:
                logger.error(f"Error en predicción ML: {str(e)}")
                # Intentar reentrenar si el error parece ser de modelo
                if "feature mismatch" in str(e).lower():
                    self.train_model(force_retrain=True)
        
        #FALLBACK (cuando ML falla)
        result = self._fallback_prediction(processed_responses, PredictionStrategy.SIMPLE_SUM)
        
        #LOGGING PARA FALLBACK
        self._log_prediction_analytics(result, start_time, age, profession, responses)
        
        return result

    def _fallback_prediction(self, responses: list, strategy: PredictionStrategy) -> Dict:
        """Predicción de fallback mejorada con heurística basada en patrones"""
        total_score = sum(responses)
        
        #primer intento
        try:
            # Análisis de patrones de respuesta
            predicted_score = self._calculate_heuristic_score(responses, total_score)
            fallback_type = 'enhanced_heuristic'
            fallback_confidence = self._calculate_heuristic_confidence(responses)
            
        except Exception as e:
            logger.warning(f"Error en heurística, usando fallback simple: {str(e)}")
            predicted_score = total_score
            fallback_type = 'simple_sum'
            fallback_confidence = 0.3 if strategy == PredictionStrategy.FALLBACK else 0.5
        
        # Usar estadísticas de fallback si están disponibles (caso específico)
        if strategy == PredictionStrategy.FALLBACK and self.fallback_ready:
            # Si la heurística falló, usar estadísticas de fallback
            if fallback_type == 'simple_sum':
                predicted_score = self.fallback_values['median']
                stress_level = self._determine_stress_level(predicted_score)
                fallback_type = 'median'
                fallback_confidence = 0.4
                
                return {
                    'predicted_score': float(predicted_score),
                    'predicted_stress_level': stress_level,
                    'confidence': fallback_confidence,
                    'confidence_interpretation': self._interpret_confidence(fallback_confidence),
                    'strategy': strategy.name,
                    'fallback_type': fallback_type,
                    'response_time_ms': 0
                }
            
            # Si la heurística funcionó pero queremos priorizar estadísticas históricas
            elif fallback_confidence < 0.5:  # Solo si la heurística tiene baja confianza
                predicted_score = self.fallback_values['median']
                stress_level = self._determine_stress_level(predicted_score)
                fallback_type = 'median_preferred'
                fallback_confidence = 0.45  # Ligeramente mayor que heurística débil
                
                return {
                    'predicted_score': float(predicted_score),
                    'predicted_stress_level': stress_level,
                    'confidence': fallback_confidence,
                    'confidence_interpretation': self._interpret_confidence(fallback_confidence),
                    'strategy': strategy.name,
                    'fallback_type': fallback_type,
                    'response_time_ms': 0
                }
        
        #RETURN PRINCIPAL: Para todos los demás casos (heurística exitosa o simple_sum)
        stress_level = self._determine_stress_level(predicted_score)
        
        return {
            'predicted_score': float(predicted_score),
            'predicted_stress_level': stress_level,
            'confidence': fallback_confidence,
            'confidence_interpretation': self._interpret_confidence(fallback_confidence),
            'strategy': strategy.name,
            'fallback_type': fallback_type,
            'response_time_ms': 0  # Fallback es instantáneo
        }
    
    def _calculate_confidence(self, predicted_score: float, responses: list, age: int = 25, profession: str = 'unknown') -> float:
        """
        Calcula la confianza de la predicción basada en múltiples factores.
        Retorna un valor entre 0.0 y 1.0

        Args:
            predicted_score: Score predicho por el modelo
            responses: Lista de 14 respuestas procesadas
            age: Edad real del usuario (corrige bug anterior que usaba responses[0])
            profession: Profesión real del usuario (corrige bug anterior que usaba 'unknown')
        """
        try:
            confidence_factors = []
            
            # Factor 1: Distancia del score a los extremos (más confianza en valores centrales)
            score_normalized = predicted_score / 56.0
            distance_from_extremes = 1.0 - abs(score_normalized - 0.5) * 2
            confidence_factors.append(distance_from_extremes * 0.3)  # 30% del peso
            
            # Factor 2: Consistencia de las respuestas (menos variabilidad = más confianza)
            response_variance = np.var(responses)
            max_variance = np.var([0, 1, 2, 3, 4])  # Máxima variabilidad posible
            consistency_score = 1.0 - (response_variance / max_variance)
            confidence_factors.append(consistency_score * 0.25)  # 25% del peso
            
            # Factor 3: OOB Score del modelo (si está disponible)
            if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
                oob_confidence = max(0, self.model.oob_score_)  # R² del OOB
                confidence_factors.append(oob_confidence * 0.3)  # 30% del peso
            else:
                confidence_factors.append(0.7)  # Valor conservador por defecto
            
            # Factor 4: Consenso de árboles con datos REALES del usuario
            if hasattr(self.model, 'estimators_'):
                try:
                    # Construir input con datos reales (NO responses[0] como age)
                    input_for_trees = pd.DataFrame({
                        'age': [age],
                        'profession': [profession],
                        **{f'q_{i+1}': [responses[i] if i < len(responses) else 2]
                           for i in range(14)}
                    })

                    # Feature engineering centralizado
                    input_for_trees = self._apply_feature_engineering(input_for_trees)

                    X_trees = self.preprocessor.transform(input_for_trees)

                    # Obtener predicciones individuales de cada árbol
                    tree_predictions = [
                        tree.predict(X_trees)[0]
                        for tree in self.model.estimators_[:min(50, len(self.model.estimators_))]
                    ]
                    
                    # Calcular consenso (qué tan cerca están las predicciones)
                    predictions_std = np.std(tree_predictions)
                    max_std = 28.0  # Estimación de máxima desviación estándar esperada
                    consensus_score = 1.0 - min(predictions_std / max_std, 1.0)
                    confidence_factors.append(consensus_score * 0.15)  # 15% del peso
                    
                except Exception:
                    confidence_factors.append(0.6)  # Valor por defecto si falla
            else:
                confidence_factors.append(0.6)  # Valor por defecto para otros modelos
            
            # Combinar todos los factores
            final_confidence = sum(confidence_factors)
            
            # Normalizar al rango [0.1, 0.95] para evitar extremos
            final_confidence = max(0.1, min(0.95, final_confidence))
            
            return round(final_confidence, 3)
            
        except Exception as e:
            logger.warning(f"Error calculando confianza: {str(e)}")
            return 0.5  # Confianza media por defecto
        
    def _interpret_confidence(self, confidence: float) -> str:
        """Proporciona una interpretación textual de la confianza"""
        if confidence >= 0.8:
            return "Muy alta - Predicción muy confiable"
        elif confidence >= 0.6:
            return "Alta - Predicción confiable"
        elif confidence >= 0.4:
            return "Media - Predicción moderadamente confiable"
        elif confidence >= 0.2:
            return "Baja - Usar con precaución"
        else:
            return "Muy baja - Resultado poco confiable"
        

    def _calculate_heuristic_score(self, responses: list, base_score: int) -> float:
        """Calcula score usando heurística basada en patrones de respuesta PSS-14"""
        
        # Usar constantes centralizadas de pss_scoring
        # Separar respuestas por tipo
        inverted_responses = [responses[i] for i in INVERTED_INDICES if i < len(responses)]
        normal_responses = [responses[i] for i in DIRECT_INDICES if i < len(responses)]

        # Calcular métricas clave
        avg_inverted = np.mean(inverted_responses) if inverted_responses else 2.0
        avg_normal = np.mean(normal_responses) if normal_responses else 2.0
        variance = np.var(responses)
        
        # Patrón 1: Respuestas consistentemente altas en preguntas invertidas
        if avg_inverted >= 2.5 and variance < 1.5:
            # Persona que responde alto en "me siento en control" = menos estrés real
            # Pero el cálculo PSS-14 lo invierte, así que ajustamos
            adjustment_factor = 0.85  # Reducir el score porque hay menos estrés real
            predicted_score = base_score * adjustment_factor
            logger.info(f"Patrón detectado: Respuestas altas consistentes en preguntas invertidas (factor: {adjustment_factor})")
            
        # Patrón 2: Respuestas muy variables (indecisión o confusión)
        elif variance > 2.5:
            # Alta variabilidad puede indicar respuestas menos confiables
            adjustment_factor = 1.1  # Pequeño aumento por incertidumbre
            predicted_score = base_score * adjustment_factor
            logger.info(f"Patrón detectado: Alta variabilidad en respuestas (factor: {adjustment_factor})")
            
        # Patrón 3: Respuestas consistentemente bajas en preguntas normales
        elif avg_normal <= 1.5 and variance < 1.0:
            # Persona que reporta poco estrés de manera consistente
            adjustment_factor = 0.9
            predicted_score = base_score * adjustment_factor
            logger.info(f"Patrón detectado: Respuestas bajas consistentes (factor: {adjustment_factor})")
            
        # Patrón 4: Respuestas extremas (muchos 0s y 4s)
        elif self._has_extreme_pattern(responses):
            # Respuestas polarizadas pueden indicar respuesta emocional intensa
            adjustment_factor = 1.15
            predicted_score = base_score * adjustment_factor
            logger.info(f"Patrón detectado: Respuestas extremas (factor: {adjustment_factor})")
            
        # Patrón 5: Discrepancia entre preguntas normales e invertidas
        elif abs(avg_normal - (4 - avg_inverted)) > 1.5:
            # Inconsistencia entre tipos de preguntas
            adjustment_factor = 1.05  # Ligero aumento por inconsistencia
            predicted_score = base_score * adjustment_factor
            logger.info(f"Patrón detectado: Discrepancia entre tipos de preguntas (factor: {adjustment_factor})")
            
        else:
            # Sin patrón específico detectado, usar score base
            predicted_score = float(base_score)
            logger.info("Sin patrón específico detectado, usando score base")
        
        # Asegurar que esté en el rango válido
        predicted_score = max(0, min(56, predicted_score))
        
        return predicted_score

    def _has_extreme_pattern(self, responses: list) -> bool:
        """Detecta si hay un patrón de respuestas extremas (muchos 0s y 4s)"""
        extreme_count = sum(1 for r in responses if r in [0, 4])
        extreme_ratio = extreme_count / len(responses)
        return extreme_ratio > 0.6  # Más del 60% son respuestas extremas

    def _calculate_heuristic_confidence(self, responses: list) -> float:
        """Calcula confianza específica para predicciones heurísticas"""
        try:
            confidence_factors = []
            
            # Factor 1: Consistencia interna (menos varianza = más confianza)
            variance = np.var(responses)
            max_variance = np.var([0, 4] * 7)  # Máxima varianza posible
            consistency = 1.0 - min(variance / max_variance, 1.0)
            confidence_factors.append(consistency * 0.4)  # 40% del peso
            
            # Factor 2: Ausencia de patrones extremos
            extreme_ratio = sum(1 for r in responses if r in [0, 4]) / len(responses)
            non_extreme_confidence = 1.0 - extreme_ratio
            confidence_factors.append(non_extreme_confidence * 0.3)  # 30% del peso
            
            # Factor 3: Coherencia entre preguntas normales e invertidas
            avg_inverted = np.mean([responses[i] for i in INVERTED_INDICES if i < len(responses)])
            avg_normal = np.mean([responses[i] for i in DIRECT_INDICES if i < len(responses)])

            # Coherencia esperada: normal alto debería corresponder con invertido bajo
            expected_coherence = abs(avg_normal - (4 - avg_inverted))
            coherence_confidence = max(0, 1.0 - (expected_coherence / 2.0))
            confidence_factors.append(coherence_confidence * 0.3)  # 30% del peso
            
            # Combinar factores
            final_confidence = sum(confidence_factors)
            
            # Para heurística, la confianza máxima es moderada (0.7)
            final_confidence = max(0.2, min(0.7, final_confidence))
            
            return round(final_confidence, 3)
            
        except Exception as e:
            logger.warning(f"Error calculando confianza heurística: {str(e)}")
            return 0.4  # Confianza moderada por defecto
        

    def _log_prediction_analytics(self, result: Dict, start_time: datetime, age: int, profession: str, responses: list) -> None:
        """Registra análisis detallado de la predicción en logs"""
        try:
            # Información básica de la predicción
            strategy = result.get('strategy', 'UNKNOWN')
            fallback_type = result.get('fallback_type', 'N/A')
            confidence = result.get('confidence', 0)
            predicted_score = result.get('predicted_score', 0)
            stress_level = result.get('predicted_stress_level', 'Unknown')
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log principal de la predicción
            logger.info(f"PREDICCIÓN COMPLETADA | "
                    f"Estrategia: {strategy} | "
                    f"Fallback: {fallback_type} | "
                    f"Score: {predicted_score:.1f} | "
                    f"Nivel: {stress_level} | "
                    f"Confianza: {confidence:.3f} | "
                    f"Tiempo: {response_time:.1f}ms")
            
            # Análisis de entrada
            response_variance = np.var(responses)
            extreme_responses = sum(1 for r in responses if r in [0, 4])
            extreme_ratio = extreme_responses / len(responses)
            
            logger.info(f"ANÁLISIS ENTRADA | "
                    f"Edad: {age} | "
                    f"Profesión: {profession[:20]}... | "
                    f"Varianza respuestas: {response_variance:.2f} | "
                    f"Respuestas extremas: {extreme_responses}/14 ({extreme_ratio:.1%})")
            
            # Análisis específico por estrategia
            if strategy == 'ML_MODEL':
                self._log_ml_analytics(result, responses)
            elif fallback_type == 'enhanced_heuristic':
                self._log_heuristic_analytics(responses)
            elif fallback_type in ['median', 'median_preferred']:
                logger.info(f"FALLBACK ESTADÍSTICO | "
                        f"Valor usado: {predicted_score:.1f} | "
                        f"Tipo: {fallback_type}")
            
            # Mantener estadísticas en memoria para análisis periódico
            self._update_prediction_stats(result, response_time)
            
        except Exception as e:
            logger.warning(f"Error en logging de análisis: {str(e)}")

    def _log_ml_analytics(self, result: Dict, responses: list) -> None:
        """Logging específico para predicciones ML"""
        try:
            # Información del modelo
            if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
                oob_score = self.model.oob_score_
                logger.info(f"ML MODELO | OOB Score: {oob_score:.3f} | "
                        f"N_estimators: {getattr(self.model, 'n_estimators', 'N/A')}")
            
            # Análisis de importancia de características (solo las top 5)
            if hasattr(self.model, 'feature_importances_') and hasattr(self.preprocessor, 'get_feature_names_out'):
                try:
                    importances = self.model.feature_importances_
                    feature_names = self.preprocessor.get_feature_names_out()
                    
                    # Top 5 características
                    importance_pairs = list(zip(feature_names, importances))
                    top_features = sorted(importance_pairs, key=lambda x: x[1], reverse=True)[:5]
                    
                    feature_log = " | ".join([f"{name}: {imp:.3f}" for name, imp in top_features])
                    logger.info(f"ML TOP FEATURES | {feature_log}")
                    
                except Exception as e:
                    logger.debug(f"No se pudo analizar importancia: {str(e)}")

            # Añadir métricas de clasificación si están disponibles
            if hasattr(self.model, 'evaluation_metrics_'):
                metrics = self.model.evaluation_metrics_
                logger.info(f"ML MÉTRICAS | "
                        f"Precision: {metrics.get('precision', 'N/A'):.3f} | "
                        f"Recall: {metrics.get('recall', 'N/A'):.3f} | "
                        f"F1: {metrics.get('f1', 'N/A'):.3f}")
                    
        except Exception as e:
            logger.warning(f"Error en logging ML: {str(e)}")

    def _log_heuristic_analytics(self, responses: list) -> None:
        """Logging específico para predicciones heurísticas"""
        try:
            # Análisis de patrones detectados usando constantes centralizadas
            inverted_responses = [responses[i] for i in INVERTED_INDICES if i < len(responses)]
            normal_responses = [responses[i] for i in DIRECT_INDICES if i < len(responses)]

            avg_inverted = np.mean(inverted_responses) if inverted_responses else 0
            avg_normal = np.mean(normal_responses) if normal_responses else 0
            variance = np.var(responses)
            
            logger.info(f"HEURÍSTICA ANÁLISIS | "
                    f"Avg Normal: {avg_normal:.2f} | "
                    f"Avg Invertidas: {avg_inverted:.2f} | "
                    f"Varianza: {variance:.2f} | "
                    f"Discrepancia: {abs(avg_normal - (4 - avg_inverted)):.2f}")
            
        except Exception as e:
            logger.warning(f"Error en logging heurístico: {str(e)}")

    def _update_prediction_stats(self, result: Dict, response_time: float) -> None:
        """Actualiza estadísticas en memoria para logging periódico"""
        try:
            # Inicializar si no existe
            if not hasattr(self, 'session_stats'):
                self.session_stats = {
                    'predictions_count': 0,
                    'strategies_used': {},
                    'fallback_types_used': {},
                    'confidence_scores': [],
                    'response_times': [],
                    'last_reset': datetime.now()
                }
            
            # Actualizar contadores
            self.session_stats['predictions_count'] += 1
            
            strategy = result.get('strategy', 'UNKNOWN')
            self.session_stats['strategies_used'][strategy] = self.session_stats['strategies_used'].get(strategy, 0) + 1
            
            fallback_type = result.get('fallback_type', 'N/A')
            if fallback_type != 'N/A':
                self.session_stats['fallback_types_used'][fallback_type] = self.session_stats['fallback_types_used'].get(fallback_type, 0) + 1
            
            confidence = result.get('confidence', 0)
            self.session_stats['confidence_scores'].append(confidence)
            self.session_stats['response_times'].append(response_time)
            
            # Log estadísticas cada 10 predicciones
            if self.session_stats['predictions_count'] % 10 == 0:
                self._log_session_summary()
                
        except Exception as e:
            logger.warning(f"Error actualizando estadísticas: {str(e)}")

    def _log_session_summary(self) -> None:
        """Log resumen de estadísticas de la sesión"""
        try:
            stats = self.session_stats
            
            # Estadísticas básicas
            total = stats['predictions_count']
            avg_confidence = np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0
            avg_response_time = np.mean(stats['response_times']) if stats['response_times'] else 0
            
            logger.info(f"RESUMEN SESIÓN | "
                    f"Total predicciones: {total} | "
                    f"Confianza promedio: {avg_confidence:.3f} | "
                    f"Tiempo promedio: {avg_response_time:.1f}ms")
            
            # Distribución de estrategias
            strategy_log = " | ".join([f"{k}: {v}" for k, v in stats['strategies_used'].items()])
            logger.info(f"ESTRATEGIAS USADAS | {strategy_log}")
            
            # Distribución de fallbacks (si hay)
            if stats['fallback_types_used']:
                fallback_log = " | ".join([f"{k}: {v}" for k, v in stats['fallback_types_used'].items()])
                logger.info(f"FALLBACKS USADOS | {fallback_log}")
            
            # Distribución de confianza
            confidence_scores = stats['confidence_scores']
            if confidence_scores:
                high_conf = sum(1 for c in confidence_scores if c >= 0.7)
                med_conf = sum(1 for c in confidence_scores if 0.4 <= c < 0.7)
                low_conf = sum(1 for c in confidence_scores if c < 0.4)
                
                logger.info(f"DISTRIBUCIÓN CONFIANZA | "
                        f"Alta (≥0.7): {high_conf} | "
                        f"Media (0.4-0.7): {med_conf} | "
                        f"Baja (<0.4): {low_conf}")
            
        except Exception as e:
            logger.warning(f"Error en resumen de sesión: {str(e)}")


    def _generate_version_hash(self, X, y) -> str:
        """Genera un hash único para versionado del modelo"""
        # Solo usar columnas numéricas para el hash
        numeric_X = X.select_dtypes(include=[np.number])
        x_mean = numeric_X.mean().mean() if not numeric_X.empty else 0
        
        data_hash = hashlib.sha256(
            f"{len(X)}{x_mean}{y.mean()}".encode()
        ).hexdigest()[:8]
        time_hash = datetime.now().strftime("%Y%m%d-%H%M")
        return f"{time_hash}-{data_hash}"

    def _clean_profession_for_model(self, profession):
        """Versión simplificada ya que las profesiones vienen validadas"""
        if not isinstance(profession, str):
            return 'unknown'
        return profession.strip().title()[:30]


    def clean_profession_data(self) -> None:
        """Método vacío ya que no necesitamos limpiar profesiones"""
        logger.info("Limpieza de profesiones desactivada (profesiones predefinidas)")
        return


    def _is_valid_profession(self, profession: str) -> bool:
        """Valida que la profesión sea un string válido (Permisivo para 'Otros')"""
        if not isinstance(profession, str) or len(profession) < 2:
            return False
        
        # Aceptamos cualquier profesión válida para permitir "Otros" y entradas libres
        # El preprocesador OneHotEncoder con handle_unknown='ignore' se encargará del resto
        return True
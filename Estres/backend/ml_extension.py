import os
import pandas as pd
import numpy as np
import joblib
import sqlite3
import logging
from typing import Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from scipy.stats import mode

# Asegurarse de que el directorio de logs exista
os.makedirs('../logs', exist_ok=True)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/stress_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    def __init__(self, db_path: str = '../datos/pss_database.db'):
        """Inicializa el predictor con configuración robusta"""
        self.db_path = db_path
        self.model_dir = '../modelos'
        self.model_path = os.path.join(self.model_dir, 'stress_predictor.joblib')
        self.preprocessor_path = os.path.join(self.model_dir, 'preprocessor.joblib')
        self.cache_path = os.path.join(self.model_dir, 'prediction_cache.joblib')
        self.min_training_samples = 10  # Mínimo absoluto para entrenar
        self.optimal_training_samples = 100  # Ideal para buen rendimiento
        self.last_training_time = None
        self.model = None
        self.preprocessor = None
        self.prediction_cache = {}
        self._load_artifacts()
        self._setup_fallback()
        
    def _load_artifacts(self) -> None:
        """Carga modelo, preprocesador y caché de forma segura"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Modelo cargado desde archivo")
            
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                logger.info("Preprocesador cargado desde archivo")
            
            if os.path.exists(self.cache_path):
                self.prediction_cache = joblib.load(self.cache_path)
                logger.info(f"Caché cargada con {len(self.prediction_cache)} entradas")
                
        except Exception as e:
            logger.error(f"Error cargando artefactos: {str(e)}")
            self._setup_fallback()
    
    def _save_artifacts(self) -> None:
        """Guarda artefactos de forma segura"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.preprocessor, self.preprocessor_path)
            joblib.dump(self.prediction_cache, self.cache_path)
        except Exception as e:
            logger.error(f"Error guardando artefactos: {str(e)}")
    
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
            
        if not isinstance(profession, str) or len(profession) > 100:
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
            responses_query = """
                SELECT test_id, question_number, processed_value 
                FROM responses 
                WHERE processed_value BETWEEN 0 AND 4
            """
            
            tests_df = pd.read_sql(tests_query, conn)
            responses_df = pd.read_sql(responses_query, conn)

            # Añadir estos logs para ver cuántos datos quedan después de los filtros
            logger.info(f"Tests después del filtro: {len(tests_df)}")
            logger.info(f"Respuestas después del filtro: {len(responses_df)}")
            logger.info(f"Respuestas test_ids únicos: {responses_df['test_id'].nunique()}")
            conn.close()
            
            if tests_df.empty or responses_df.empty:
                logger.warning("Datos vacíos en la base de datos")
                return None, None
            
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
                
            X = df.drop(['test_id', 'total_score'], axis=1)
            y = df['total_score']
            
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
    
    def _create_pipeline(self) -> Pipeline:
        """Crea pipeline de preprocesamiento y modelo con validación"""
        numeric_features = ['age']
        categorical_features = ['profession']
        question_features = [f'q_{i+1}' for i in range(14)]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='other')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('q', 'passthrough', question_features)
            ])
        
        # Modelo con parámetros robustos
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
    
    def _evaluate_model(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
        """Evalúa el modelo con múltiples métricas"""
        try:
            # Validación cruzada
            cv_scores = cross_val_score(
                pipeline, X, y, 
                cv=5, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            mae_scores = -cv_scores
            avg_mae = np.mean(mae_scores)
            
            # Entrenamiento final y evaluación en conjunto de prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            test_mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"MAE Validación Cruzada: {avg_mae:.2f}, Test: {test_mae:.2f}")
            
            return test_mae
            
        except Exception as e:
            logger.error(f"Error evaluando modelo: {str(e)}")
            return float('inf')
    
    def train_model(self, force_retrain: bool = False) -> Dict[str, float]:
        """Entrena un nuevo modelo con validación robusta"""
        # Verificar si necesitamos reentrenar
        if (not force_retrain and self.model is not None and 
            self.last_training_time and 
            (datetime.now() - self.last_training_time) < timedelta(days=7)):
            logger.info("Modelo actual es reciente, omitiendo reentrenamiento")
            return {'status': 'skipped', 'reason': 'model_is_recent'}
        
        X, y = self._prepare_data()
        
        if X is None or len(X) < self.min_training_samples:
            logger.warning("Datos insuficientes para entrenamiento")
            return {'status': 'failed', 'reason': 'insufficient_data'}
        
        try:
            pipeline = self._create_pipeline()
            mae_score = self._evaluate_model(pipeline, X, y)
            
            # Entrenar con todos los datos si el rendimiento es aceptable
            if mae_score < 5.0:  # Umbral de MAE aceptable
                pipeline.fit(X, y)
                self.model = pipeline.named_steps['regressor']
                self.preprocessor = pipeline.named_steps['preprocessor']
                self.last_training_time = datetime.now()
                self._save_artifacts()
                
                logger.info(f"Modelo entrenado con MAE: {mae_score:.2f}")
                return {
                    'status': 'success',
                    'mae': mae_score,
                    'training_samples': len(X)
                }
            else:
                logger.warning(f"Rendimiento del modelo inaceptable (MAE: {mae_score:.2f})")
                return {
                    'status': 'failed',
                    'reason': 'poor_performance',
                    'mae': mae_score
                }
                
        except Exception as e:
            logger.error(f"Error en entrenamiento: {str(e)}")
            return {'status': 'failed', 'reason': 'training_error'}
    
    def _get_cache_key(self, age: int, profession: str, responses: list) -> str:
        """Genera una clave única para el caché"""
        response_hash = hash(tuple(responses))
        return f"{age}_{profession}_{response_hash}"
    
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
        
        # Procesar respuestas
        inverted_indices = [3, 4, 6, 7, 10, 13]
        processed_responses = [
            0 if resp == 0 and i in inverted_indices else 
            (4 - resp if i in inverted_indices else resp) 
            for i, resp in enumerate(responses)
        ]
        total_score = sum(processed_responses)
        
        # Verificar caché primero
        cache_key = self._get_cache_key(age, profession, responses)
        if cache_key in self.prediction_cache:
            logger.info("Predicción encontrada en caché")
            result = self.prediction_cache[cache_key]
            result['strategy'] = PredictionStrategy.CACHED.name
            result['response_time_ms'] = (datetime.now() - start_time).total_seconds() * 1000
            return result
        
        # Intentar con el modelo ML si está disponible
        if self.model is not None and self.preprocessor is not None:
            try:
                input_data = pd.DataFrame({
                    'age': [age],
                    'profession': [profession],
                    **{f'q_{i+1}': [resp] for i, resp in enumerate(processed_responses)}
                })
                
                X_processed = self.preprocessor.transform(input_data)
                predicted_score = self.model.predict(X_processed)[0]
                predicted_score = np.clip(predicted_score, 0, 56)
                
                result = {
                    'predicted_score': float(predicted_score),
                    'predicted_stress_level': self._determine_stress_level(predicted_score),
                    'strategy': PredictionStrategy.ML_MODEL.name,
                    'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
                }
                
                # Almacenar en caché
                self.prediction_cache[cache_key] = result
                self._save_artifacts()
                
                return result
                
            except Exception as e:
                logger.error(f"Error en predicción ML: {str(e)}")
                # Intentar reentrenar si el error parece ser de modelo
                if "feature mismatch" in str(e).lower():
                    self.train_model(force_retrain=True)
        
        # Fallback a suma simple si ML falla
        return self._fallback_prediction(processed_responses, PredictionStrategy.SIMPLE_SUM)
    
    def _fallback_prediction(self, responses: list, strategy: PredictionStrategy) -> Dict:
        """Predicción de fallback usando diferentes estrategias"""
        total_score = sum(responses)
        
        # Usar estadísticas de fallback si están disponibles
        if strategy == PredictionStrategy.FALLBACK and self.fallback_ready:
            predicted_score = self.fallback_values['median']
            stress_level = self._determine_stress_level(predicted_score)
            
            return {
                'predicted_score': float(predicted_score),
                'predicted_stress_level': stress_level,
                'strategy': strategy.name,
                'fallback_type': 'median'
            }
        
        # Suma simple como último recurso
        stress_level = self._determine_stress_level(total_score)
        
        return {
            'predicted_score': float(total_score),
            'predicted_stress_level': stress_level,
            'strategy': strategy.name,
            'fallback_type': 'simple_sum'
        }
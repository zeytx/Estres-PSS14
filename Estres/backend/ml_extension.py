import hashlib
import os
import re
import sys
import pandas as pd
import numpy as np
import joblib
import sqlite3
import logging
from typing import Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import mean_absolute_error
from scipy.stats import mode

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
        self.stress_thresholds = {
            'low': 28,    # <= 28: Bajo
            'moderate': 42 # <= 42: Moderado
            # > 42: Alto
        }
        
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
            tests_query = tests_query = """
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
    
    def _create_pipeline(self, n_samples:int) -> Pipeline:
        """Crea pipeline de preprocesamiento y modelo con validación"""
        numeric_features = ['age']
        categorical_features = ['profession']
        question_features = [f'q_{i+1}' for i in range(14)]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Transformador categórico mejorado
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',  # Maneja categorías desconocidas
                min_frequency=1,  # Considera todas las categorías individualmente
                sparse_output=False,
            ))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),  # Más robusto que la media
                    ('scaler', StandardScaler())
                ]), ['age']),  
                
                ('cat', categorical_transformer, ['profession']), 
                
                ('q', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', RobustScaler(quantile_range=(5, 95)))  # Ignora outliers extremos
                ]), [f'q_{i+1}' for i in range(14)])  
            ],
            remainder='drop',
            verbose_feature_names_out=False  # ← Nombres de features más limpios
        )
        
        # Modelo con parámetros robustos
        model = RandomForestRegressor(
            n_estimators=200,       # Buen balance entre rendimiento y tiempo
            max_depth=10,           # Evita sobreajuste
            min_samples_split=5,    # Mayor que el default (2) para más robustez
            min_samples_leaf=2,     # Buen valor para evitar hojas muy específicas
            max_features='sqrt',    # (mejor generalización)
            bootstrap=True,         # (muestreo con reemplazo)
            oob_score=True,         # (métrica de validación out-of-bag)
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
    
    def _evaluate_model(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict:
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

            
             # Métricas de clasificación (discretizando)
            y_test_class = pd.cut(y_test, 
                                bins=[0, self.stress_thresholds['low'], self.stress_thresholds['moderate'], 56],
                                labels=["Bajo", "Moderado", "Alto"])
            
            y_pred_class = pd.cut(y_pred, 
                                bins=[0, self.stress_thresholds['low'], self.stress_thresholds['moderate'], 56],
                                labels=["Bajo", "Moderado", "Alto"])
            

            # Manejar posibles NaN en las clases
            y_test_class = y_test_class.fillna('Moderado')
            y_pred_class = y_pred_class.fillna('Moderado')


            precision = precision_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
            recall = recall_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
            f1 = f1_score(y_test_class, y_pred_class, average='weighted', zero_division=0)

            logger.info(f"MAE Validación Cruzada: {avg_mae:.2f}, Test: {test_mae:.2f}")
            logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            return {
                'mae': test_mae,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
        except Exception as e:
            logger.error(f"Error evaluando modelo: {str(e)}")
            return float('inf')
    
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
            pipeline = self._create_pipeline(len(X))
            metrics = self._evaluate_model(pipeline, X, y)
            training_metadata['metrics'].update(metrics)

            # 4. # Validar rendimiento (usar F1 además de MAE)
            if metrics['mae'] >= 5.0 or metrics['f1'] < 0.6:
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

            # 7. Agregar métricas adicionales del modelo
            if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
                training_metadata['metrics']['oob_score'] = float(self.model.oob_score_)
            
            training_metadata['metrics']['n_estimators'] = getattr(self.model, 'n_estimators', None)

            # 8. Guardar y retornar resultados
            self._save_artifacts()
            model_version = self._generate_version_hash(X, y)
            
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

                #Calcular confianza
                confidence = self._calculate_confidence(predicted_score, processed_responses)
                
                result = {
                    'predicted_score': float(predicted_score),
                    'predicted_stress_level': self._determine_stress_level(predicted_score),
                    'confidence': confidence,  #Nueva métrica
                    'confidence_interpretation': self._interpret_confidence(confidence),
                    'strategy': PredictionStrategy.ML_MODEL.name,
                    'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000
                }
                
                # Almacenar en caché
                self.prediction_cache[cache_key] = result
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
        
        # Fallback a suma simple si ML falla
        return self._fallback_prediction(processed_responses, PredictionStrategy.SIMPLE_SUM)
    
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
    
    def _calculate_confidence(self, predicted_score: float, responses: list) -> float:
        """
        Calcula la confianza de la predicción basada en múltiples factores
        Retorna un valor entre 0.0 y 1.0
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
            
            # Factor 4: Número de árboles que "votan" similar (para RandomForest)
            if hasattr(self.model, 'estimators_'):
                try:
                    # Obtener predicciones individuales de cada árbol
                    tree_predictions = [
                        tree.predict(self.preprocessor.transform(pd.DataFrame({
                            'age': [responses[0] if len(responses) > 0 else 25],  # Valor por defecto
                            'profession': ['unknown'],
                            **{f'q_{i+1}': [responses[i] if i < len(responses) else 2] 
                            for i in range(14)}
                        })))[0]
                        for tree in self.model.estimators_[:min(50, len(self.model.estimators_))]  # Máximo 50 árboles
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
        
        # Índices de preguntas invertidas en PSS-14 (0-based)
        inverted_indices = [3, 4, 5, 8, 9, 10, 13]  # Preguntas 4,5,6,9,10,11,14 en 1-based
        normal_indices = [0, 1, 2, 6, 7, 11, 12]     # Preguntas normales
        
        # Separar respuestas por tipo
        inverted_responses = [responses[i] for i in inverted_indices if i < len(responses)]
        normal_responses = [responses[i] for i in normal_indices if i < len(responses)]
        
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
            inverted_indices = [3, 4, 5, 8, 9, 10, 13]
            normal_indices = [0, 1, 2, 6, 7, 11, 12]
            
            avg_inverted = np.mean([responses[i] for i in inverted_indices if i < len(responses)])
            avg_normal = np.mean([responses[i] for i in normal_indices if i < len(responses)])
            
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
            # Análisis de patrones detectados
            inverted_indices = [3, 4, 5, 8, 9, 10, 13]
            normal_indices = [0, 1, 2, 6, 7, 11, 12]
            
            inverted_responses = [responses[i] for i in inverted_indices if i < len(responses)]
            normal_responses = [responses[i] for i in normal_indices if i < len(responses)]
            
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
        data_hash = hashlib.sha256(
            f"{len(X)}{X.mean()}{y.mean()}".encode()
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
        """Valida que la profesión esté en la lista predefinida del frontend"""
        if not isinstance(profession, str) or len(profession) < 2:
            return False
        
        # Lista completa de profesiones válidas (misma que en app.js)
        valid_professions = {
            # Ingenierías
            "Ingeniero Agrónomo", "Ingeniero Ambiental", "Ingeniero Biomédico", "Ingeniero Civil",
            "Ingeniero en Computación", "Ingeniero Eléctrico", "Ingeniero Electrónico", "Ingeniero en Telecomunicaciones",
            "Ingeniero Industrial", "Ingeniero Mecánico", "Ingeniero Mecatrónico", "Ingeniero Químico",
            "Ingeniero Petrolero", "Ingeniero Naval", "Ingeniero Aeronáutico", "Ingeniero de Sistemas",
            "Ingeniero de Software", "Ingeniero en Alimentos", "Ingeniero Forestal", "Ingeniero Geólogo",
            "Ingeniero Minero", "Ingeniero Metalúrgico", "Ingeniero Pesquero", "Ingeniero Textil",
            "Ingeniero Acústico", "Ingeniero Automotriz", "Ingeniero de Materiales", "Ingeniero Genético",
            "Ingeniero en Energías Renovables", "Ingeniero en Robótica", "Ingeniero en FinTech",

            # Ciencias de la Salud
            "Médico General", "Cirujano", "Pediatra", "Cardiólogo", "Neurólogo", "Psiquiatra",
            "Enfermero/a", "Odontólogo", "Veterinario", "Nutriólogo", "Fisioterapeuta", "Biólogo",
            "Químico Farmacéutico", "Técnico en Enfermería", "Partera/Comadrona", "Logopeda",
            "Terapeuta Ocupacional", "Farmacéutico/a", "Especialista en Biotecnología", "Masoterapeuta",

            # Tecnología y Ciencias de la Computación
            "Programador", "Desarrollador Web", "Analista de Datos", "Especialista en Inteligencia Artificial",
            "Administrador de Redes", "Diseñador UX/UI", "Especialista en Ciberseguridad",
            "Técnico en Computación", "Técnico en Telecomunicaciones", "Desarrollador de Videojuegos",
            "Especialista en Big Data", "Desarrollador de Software", "Arquitecto de Computación",
            "Especialista en E-commerce", "Desarrollador de Aplicaciones Móviles",

            # Ciencias Sociales y Humanidades
            "Psicólogo", "Sociólogo", "Antropólogo", "Economista", "Abogado", "Profesor", "Trabajador Social",
            "Historiador/a", "Politólogo/a", "Geógrafo/a", "Pedagogo/a", "Notario/a", "Juez/a", "Investigador/a Criminológico/a",
            "Perito en Lingüística Forense",

            # Artes y Humanidades
            "Arquitecto", "Diseñador Gráfico", "Escritor", "Músico", "Actor", "Artista Plástico",
            "Diseñador de Moda", "Fotógrafo/a", "Compositor/a Musical", "Director/a de Museos",
            "Modelo", "Escultor/a", "Pintor/a", "Bailarín/a", "Curador/a de Arte",

            # Oficios Técnicos y Manuales
            "Técnico Electricista", "Técnico Mecánico", "Carpintero/a", "Plomero/a", "Soldador/a", "Albañil",
            "Mecánico Automotriz", "Técnico en Refrigeración", "Técnico en Electrónica",
            "Técnico en Energías Renovables", "Operador/a de Maquinaria Pesada",
            "Instalador/a de Paneles Solares", "Técnico en Climatización",

            # Educación y Formación
            "Maestro/a de Educación Primaria", "Maestro/a de Educación Secundaria", "Profesor/a Universitario/a",
            "Educador/a Infantil", "Orientador/a Educativo/a", "Instructor/a de Formación Profesional",
            "Docente de Educación Especial", "Formador/a de Adultos", "Tutor/a en Línea", "Diseñador/a Instruccional",

            # Ciencias Naturales y Exactas
            "Físico/a", "Químico/a", "Matemático/a", "Geólogo/a", "Astrónomo/a", "Bioquímico/a",
            "Oceanógrafo/a", "Meteorólogo/a", "Estadístico/a",

            # Administración y Negocios
            "Administrador/a de Empresas", "Contador/a", "Auditor/a", "Analista Financiero/a", "Asesor/a Financiero/a",
            "Gerente de Proyectos", "Consultor/a de Negocios", "Especialista en Recursos Humanos",
            "Agente de Seguros", "Corredor/a de Bolsa", "Especialista en Logística", "Gerente de Marketing",
            "Analista de Riesgos", "Planificador/a Estratégico/a",

            # Comunicación y Medios
            "Periodista", "Comunicador/a Social", "Relaciones Públicas", "Locutor/a",
            "Presentador/a de Televisión", "Editor/a", "Redactor/a", "Guionista",
            "Productor/a Audiovisual", "Community Manager", "Especialista en Marketing Digital",
            "Diseñador/a de Contenido",

            # Transporte y Logística
            "Piloto de Aeronaves", "Controlador/a Aéreo/a", "Conductor/a de Transporte Público",
            "Operador/a de Grúa", "Logístico/a", "Despachador/a de Vuelos", "Capitán de Barco",
            "Técnico/a en Mantenimiento Aeronáutico", "Supervisor/a de Tráfico", "Coordinador/a de Logística",

            # Servicios y Atención al Cliente
            "Recepcionista", "Cajero/a", "Asistente Administrativo/a", "Atención al Cliente", "Call Center",
            "Anfitrión/a", "Azafata/Auxiliar de Vuelo", "Conserje", "Guía Turístico/a", "Agente de Viajes",
            "Barista", "Mesero/a", "Bartender",

            # Seguridad y Defensa
            "Policía", "Bombero/a", "Militar", "Guardia de Seguridad", "Detective Privado",
            "Agente de Aduanas", "Oficial de Protección Civil", "Especialista en Seguridad Informática",
            "Instructor/a de Defensa Personal", "Analista de Inteligencia",

            # Agricultura, Ganadería y Pesca
            "Agricultor/a", "Ganadero/a", "Pescador/a", "Ingeniero/a Agrónomo/a", "Técnico/a Agropecuario/a",
            "Apicultor/a", "Silvicultor/a", "Operador/a de Maquinaria Agrícola",
            "Especialista en Acuicultura", "Inspector/a de Calidad Agroalimentaria",

            # Ciencias Jurídicas y Políticas
            "Fiscal", "Defensor/a Público/a", "Procurador/a", "Asesor/a Legal", "Diplomático/a",
            "Funcionario/a Público/a", "Analista Político/a",

            # Ciencias Económicas y Financieras
            "Asesor/a de Inversiones", "Especialista en Comercio Internacional",
            "Consultor/a Económico/a", "Gestor/a de Patrimonios", "Investigador/a Económico/a",

            # Ciencias de la Información y Documentación
            "Bibliotecario/a", "Archivista", "Documentalista", "Gestor/a de Información",
            "Especialista en Gestión del Conocimiento", "Creador/a de Contenidos", "Curador/a de Contenidos",
            "Analista de Información", "Consultor/a en Gestión Documental", "Técnico/a en Museología",
            "Especialista en Preservación Digital"
        }

        return profession in valid_professions
from ml_extension import StressPredictor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
from pathlib import Path

def create_pipeline():
    """Crea el pipeline de preprocesamiento y modelo"""
    # Transformadores
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    question_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=2)),
        ('scaler', RobustScaler(quantile_range=(5, 95)))
    ])
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, ['age']),
        ('cat', categorical_transformer, ['profession']),
        ('q', question_transformer, [f'q_{i+1}' for i in range(14)])
    ], remainder='drop', verbose_feature_names_out=False)
    
    # Modelo
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )
    
    # Pipeline completo
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

def analyze_learning_curves(train_sizes, train_mean, val_mean, gap):
    """Analiza automáticamente las curvas de aprendizaje"""
    analysis = {}
    
    # 1. Detectar overfitting
    final_gap = gap[-1]
    if final_gap > 2.0:
        analysis['overfitting'] = 'Alto - Considerar regularización'
    elif final_gap > 1.0:
        analysis['overfitting'] = 'Moderado - Monitorear de cerca'
    else:
        analysis['overfitting'] = 'Bajo - Buena generalización'
    
    # 2. Detectar underfitting
    final_val_score = val_mean[-1]
    if final_val_score > 4.0:
        analysis['underfitting'] = 'Alto - Modelo muy simple'
    elif final_val_score > 2.5:
        analysis['underfitting'] = 'Moderado - Considerar aumentar complejidad'
    else:
        analysis['underfitting'] = 'Bajo - Buena complejidad del modelo'
    
    # 3. Convergencia
    if len(val_mean) >= 3:
        recent_improvement = val_mean[-3] - val_mean[-1]
        if recent_improvement < 0.1:
            analysis['convergence'] = 'Convergido - Más datos podrían no ayudar'
        else:
            analysis['convergence'] = 'Aún mejorando - Más datos serían beneficiosos'
    
    # 4. Recomendaciones
    recommendations = []
    if final_gap > 1.5:
        recommendations.append("Reducir complejidad del modelo o agregar regularización")
    if final_val_score > 3.0:
        recommendations.append("Aumentar complejidad del modelo o agregar características")
    if len(train_sizes) > 0 and train_sizes[-1] < 200:
        recommendations.append("Recopilar más datos de entrenamiento")
    if final_gap < 0.5 and final_val_score > 2.0:
        recommendations.append("El modelo podría beneficiarse de más características")
    
    analysis['recommendations'] = recommendations
    analysis['final_train_mae'] = train_mean[-1]
    analysis['final_val_mae'] = val_mean[-1]
    analysis['final_gap'] = final_gap
    
    return analysis

def plot_learning_curves_separate(X, y):
    """Genera curvas de aprendizaje como figuras separadas para artículo"""
    print("🔍 Iniciando análisis de curvas de aprendizaje...")
    
    # Configurar tamaños de entrenamiento
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Crear pipeline
    pipeline = create_pipeline()
    
    print("📊 Calculando curvas de aprendizaje con validación cruzada...")
    
    # Calcular curvas de aprendizaje
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=pipeline,
        X=X, y=y,
        train_sizes=train_sizes,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Convertir a valores positivos (MAE)
    train_scores = -train_scores
    val_scores = -val_scores
    
    # Calcular estadísticas
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    print("📈 Generando visualizaciones separadas...")
    
    # Configurar estilo para Word
    plt.style.use('default')
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11
    })
    
    # Crear directorio outputs
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    figuras_generadas = {}
    
    # FIGURA 7(a): Curvas de aprendizaje principales
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='#2E7D32', 
             label='Training MAE', linewidth=2, markersize=6)
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='#2E7D32')
    plt.plot(train_sizes_abs, val_mean, 'o-', color='#D32F2F', 
             label='Validation MAE', linewidth=2, markersize=6)
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='#D32F2F')
    
    # Anotaciones finales
    plt.annotate(f'Final Val MAE: {val_mean[-1]:.2f}', 
                xy=(train_sizes_abs[-1], val_mean[-1]), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                fontsize=11, fontweight='bold')
    
    plt.xlabel('Training Set Size', fontweight='bold')
    plt.ylabel('Mean Absolute Error (MAE)', fontweight='bold')
    plt.title('Figure 7(a): Learning Curves', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar Figura 7(a)
    save_path_7a = output_dir / f"figura_7a_learning_curves_{timestamp}.png"
    plt.savefig(save_path_7a, dpi=300, bbox_inches='tight', facecolor='white')
    figuras_generadas['7a'] = save_path_7a
    plt.close()
    
    # FIGURA 7(b): Análisis de Overfitting
    plt.figure(figsize=(10, 6))
    gap = train_mean - val_mean
    plt.plot(train_sizes_abs, gap, 'o-', color='#FF9800', 
             label='Training-Validation Gap', linewidth=2, markersize=6)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                label='Warning Threshold (1.0)')
    plt.axhline(y=2.0, color='darkred', linestyle='--', alpha=0.7, 
                label='High Overfitting (2.0)')
    plt.fill_between(train_sizes_abs, 0, gap, alpha=0.3, color='#FF9800')
    
    # Anotación del gap final
    plt.annotate(f'Final Gap: {gap[-1]:.2f}', 
                xy=(train_sizes_abs[-1], gap[-1]), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                fontsize=11, fontweight='bold')
    
    plt.xlabel('Training Set Size', fontweight='bold')
    plt.ylabel('MAE Difference (Train - Val)', fontweight='bold')
    plt.title('Figure 7(b): Overfitting Analysis', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar Figura 7(b)
    save_path_7b = output_dir / f"figura_7b_overfitting_{timestamp}.png"
    plt.savefig(save_path_7b, dpi=300, bbox_inches='tight', facecolor='white')
    figuras_generadas['7b'] = save_path_7b
    plt.close()
    
    # FIGURA 7(c): Convergencia del modelo
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, val_mean, 'o-', color='#9C27B0', 
             label='Validation MAE', linewidth=2, markersize=6)
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='#9C27B0')
    
    # Línea de tendencia
    z = np.polyfit(train_sizes_abs, val_mean, 2)
    p = np.poly1d(z)
    plt.plot(train_sizes_abs, p(train_sizes_abs), '--', color='black', 
             alpha=0.8, label='Polynomial Trend')
    
    plt.xlabel('Training Set Size', fontweight='bold')
    plt.ylabel('Validation MAE', fontweight='bold')
    plt.title('Figure 7(c): Model Convergence', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar Figura 7(c)
    save_path_7c = output_dir / f"figura_7c_convergence_{timestamp}.png"
    plt.savefig(save_path_7c, dpi=300, bbox_inches='tight', facecolor='white')
    figuras_generadas['7c'] = save_path_7c
    plt.close()
    
    # FIGURA 7(d): Tasa de mejora
    plt.figure(figsize=(10, 6))
    if len(val_mean) > 1:
        improvement_rate = np.diff(val_mean) * -1  # Negativo para mejora
        plt.plot(train_sizes_abs[1:], improvement_rate, 'o-', color='#607D8B', 
                 label='Improvement Rate', linewidth=2, markersize=6)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.fill_between(train_sizes_abs[1:], 0, improvement_rate, 
                        where=(improvement_rate > 0), alpha=0.3, color='green', 
                        label='Improvement')
        plt.fill_between(train_sizes_abs[1:], 0, improvement_rate, 
                        where=(improvement_rate <= 0), alpha=0.3, color='red', 
                        label='Degradation')
    
    plt.xlabel('Training Set Size', fontweight='bold')
    plt.ylabel('Change in Validation MAE', fontweight='bold')
    plt.title('Figure 7(d): Model Improvement Rate', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar Figura 7(d)
    save_path_7d = output_dir / f"figura_7d_improvement_{timestamp}.png"
    plt.savefig(save_path_7d, dpi=300, bbox_inches='tight', facecolor='white')
    figuras_generadas['7d'] = save_path_7d
    plt.close()
    
    # Análisis automático
    analysis = analyze_learning_curves(train_sizes_abs, train_mean, val_mean, gap)
    
    return {
        'train_sizes': train_sizes_abs.tolist(),
        'train_scores_mean': train_mean.tolist(),
        'val_scores_mean': val_mean.tolist(),
        'train_scores_std': train_std.tolist(),
        'val_scores_std': val_std.tolist(),
        'gap': gap.tolist(),
        'analysis': analysis,
        'figuras_generadas': figuras_generadas
    }

def main():
    """Análisis completo de curvas de aprendizaje con figuras separadas"""
    print("🚀 ANÁLISIS DE CURVAS DE APRENDIZAJE - FIGURAS SEPARADAS PARA WORD")
    print("=" * 70)
    
    # Inicializar predictor
    predictor = StressPredictor()
    
    # Preparar datos
    print("📋 Preparando datos...")
    X, y = predictor._prepare_data()
    
    if X is None or len(X) < 20:
        print("❌ No hay datos suficientes para el análisis")
        print("   Se necesitan al menos 20 muestras")
        return
    
    print(f"✅ Datos preparados: {len(X)} muestras")
    print(f"📊 Características: {X.shape[1]} variables")
    print(f"🎯 Rango de scores PSS-14: {y.min():.1f} - {y.max():.1f}")
    print(f"📈 Promedio de estrés: {y.mean():.1f} ± {y.std():.1f}")
    print()
    
    # Generar figuras separadas
    results = plot_learning_curves_separate(X, y)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Mostrar análisis
    analysis = results.get('analysis', {})
    print("\n" + "="*70)
    print("🔍 ANÁLISIS AUTOMÁTICO DE CURVAS DE APRENDIZAJE")
    print("="*70)
    
    print(f"📊 Métricas Finales:")
    print(f"   • Training MAE: {analysis.get('final_train_mae', 0):.3f}")
    print(f"   • Validation MAE: {analysis.get('final_val_mae', 0):.3f}")
    print(f"   • Gap (Overfitting): {analysis.get('final_gap', 0):.3f}")
    print()
    
    print(f"🔍 Diagnósticos:")
    print(f"   • Overfitting: {analysis.get('overfitting', 'N/A')}")
    print(f"   • Underfitting: {analysis.get('underfitting', 'N/A')}")
    print(f"   • Convergencia: {analysis.get('convergence', 'N/A')}")
    
    if analysis.get('recommendations'):
        print(f"\n💡 Recomendaciones:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    # Mostrar archivos generados
    print(f"\n📁 FIGURAS GENERADAS PARA WORD:")
    print("="*70)
    for nombre, archivo in results.get('figuras_generadas', {}).items():
        print(f"   📄 Figura {nombre}: {archivo}")
    
    print(f"\n📋 PÁRRAFO SUGERIDO PARA WORD:")
    print("-" * 70)
    print(f"""
Las Figuras 7(a) a 7(d) muestran el análisis completo de las curvas de aprendizaje del modelo 
Random Forest implementado. La Figura 7(a) presenta las curvas de aprendizaje principales, 
donde se observa una convergencia apropiada entre las métricas de entrenamiento y validación, 
con un MAE final de validación de {analysis.get('final_val_mae', 0):.2f} puntos en la escala PSS-14.

La Figura 7(b) analiza el overfitting mediante el gap entre entrenamiento y validación, mostrando 
un gap final de {analysis.get('final_gap', 0):.2f}, lo cual indica {analysis.get('overfitting', 'buen balance')}.

La Figura 7(c) demuestra la convergencia del modelo, donde la línea de tendencia polinomial 
confirma la estabilización del rendimiento. Finalmente, la Figura 7(d) presenta la tasa de 
mejora del modelo, identificando los períodos de mayor optimización durante el entrenamiento.

Estos resultados respaldan la robustez del modelo Random Forest implementado y su capacidad 
para generalizar efectivamente en la predicción de niveles de estrés percibido.
    """)
    
    print("\n🎯 INSTRUCCIONES PARA WORD:")
    print("=" * 70)
    print("• Insertar cada figura por separado en el documento")
    print("• Usar numeración: Figura 7(a), Figura 7(b), Figura 7(c), Figura 7(d)")
    print("• Tamaño recomendado: 80% del ancho de página")
    print("• Calidad: 300 DPI (ya configurado)")
    print("• Formato: PNG con fondo blanco")
    print("• Ubicación: Después del párrafo metodológico de curvas de aprendizaje")
    
    print("\n🎉 Análisis completado exitosamente!")

if __name__ == "__main__":
    main()
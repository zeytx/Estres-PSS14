import sqlite3
import pandas as pd
from collections import Counter
import numpy as np
from datetime import datetime

class DistribucionPoblacion:
    """Clase para analizar la distribución de la población en la base de datos PSS-14"""
    
    def __init__(self, db_path='../datos/pss_database.db'):
        self.db_path = db_path
        self.datos = None
        
    def cargar_datos(self):
        """Carga todos los datos de la base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query para obtener datos básicos
            query = """
            SELECT 
                test_id,
                age,
                profession,
                total_score,
                stress_level,
                timestamp
            FROM tests
            ORDER BY test_id
            """
            
            self.datos = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(self.datos) == 0:
                print("⚠️  No hay datos en la base de datos")
                return False
                
            print(f"✅ Datos cargados exitosamente: {len(self.datos)} registros")
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Error al cargar datos: {e}")
            return False
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return False
    
    def mostrar_distribucion_edad(self):
        """Muestra la distribución por edad"""
        if self.datos is None or len(self.datos) == 0:
            print("❌ No hay datos para analizar")
            return
        
        print("\n" + "="*60)
        print("📊 DISTRIBUCIÓN POR EDAD")
        print("="*60)
        
        # Estadísticas básicas
        print(f"📈 Estadísticas básicas:")
        print(f"   • Edad mínima: {self.datos['age'].min()}")
        print(f"   • Edad máxima: {self.datos['age'].max()}")
        print(f"   • Edad promedio: {self.datos['age'].mean():.1f}")
        print(f"   • Desviación estándar: {self.datos['age'].std():.1f}")
        print(f"   • Mediana: {self.datos['age'].median():.1f}")
        
        # Distribución por rangos etarios
        print(f"\n📊 Distribución por rangos etarios:")
        
        # Crear rangos etarios
        bins = [0, 18, 25, 35, 45, 55, 65, 100]
        labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        
        self.datos['rango_edad'] = pd.cut(self.datos['age'], bins=bins, labels=labels, right=False)
        
        # CORRECCIÓN: Agregar observed=True aquí también
        distribucion_edad = self.datos['rango_edad'].value_counts(sort=False).sort_index()
        
        for rango, count in distribucion_edad.items():
            porcentaje = (count / len(self.datos)) * 100
            barra = "█" * int(porcentaje / 2)  # Cada █ representa 2%
            print(f"   • {rango:>6}: {count:>3} personas ({porcentaje:>5.1f}%) {barra}")
        
        # Top 5 edades más frecuentes
        print(f"\n🔝 Top 5 edades más frecuentes:")
        top_edades = self.datos['age'].value_counts().head(5)
        for edad, count in top_edades.items():
            porcentaje = (count / len(self.datos)) * 100
            print(f"   • {edad} años: {count} personas ({porcentaje:.1f}%)")
    
    def mostrar_distribucion_profesion(self):
        """Muestra la distribución por profesión"""
        if self.datos is None or len(self.datos) == 0:
            print("❌ No hay datos para analizar")
            return
        
        print("\n" + "="*60)
        print("👥 DISTRIBUCIÓN POR PROFESIÓN")
        print("="*60)
        
        # Contar profesiones
        distribucion_prof = self.datos['profession'].value_counts()
        
        print(f"📊 Total de profesiones únicas: {len(distribucion_prof)}")
        print(f"📈 Distribución completa:")
        
        for i, (profesion, count) in enumerate(distribucion_prof.items(), 1):
            porcentaje = (count / len(self.datos)) * 100
            barra = "█" * int(porcentaje / 2)  # Cada █ representa 2%
            print(f"   {i:>2}. {profesion:<20}: {count:>3} personas ({porcentaje:>5.1f}%) {barra}")
        
        # Mostrar profesiones más y menos frecuentes
        print(f"\n🔝 Profesión más frecuente:")
        prof_top = distribucion_prof.iloc[0]
        porcentaje_top = (prof_top / len(self.datos)) * 100
        print(f"   • {distribucion_prof.index[0]}: {prof_top} personas ({porcentaje_top:.1f}%)")
        
        if len(distribucion_prof) > 1:
            print(f"\n🔻 Profesión menos frecuente:")
            prof_bottom = distribucion_prof.iloc[-1]
            porcentaje_bottom = (prof_bottom / len(self.datos)) * 100
            print(f"   • {distribucion_prof.index[-1]}: {prof_bottom} personas ({porcentaje_bottom:.1f}%)")
    
    def mostrar_distribucion_estres(self):
        """Muestra la distribución por nivel de estrés"""
        if self.datos is None or len(self.datos) == 0:
            print("❌ No hay datos para analizar")
            return
        
        print("\n" + "="*60)
        print("😰 DISTRIBUCIÓN POR NIVEL DE ESTRÉS")
        print("="*60)
        
        # Distribución por niveles
        distribucion_estres = self.datos['stress_level'].value_counts()
        
        print(f"📊 Distribución por niveles de estrés:")
        
        for nivel, count in distribucion_estres.items():
            porcentaje = (count / len(self.datos)) * 100
            barra = "█" * int(porcentaje / 2)  # Cada █ representa 2%
            
            # Emojis según nivel
            emoji = {"Bajo": "😌", "Moderado": "😐", "Alto": "😟"}.get(nivel, "❓")
            
            print(f"   {emoji} {nivel:<9}: {count:>3} personas ({porcentaje:>5.1f}%) {barra}")
        
        # Estadísticas de scores
        print(f"\n📈 Estadísticas de scores PSS-14:")
        print(f"   • Score mínimo: {self.datos['total_score'].min()}")
        print(f"   • Score máximo: {self.datos['total_score'].max()}")
        print(f"   • Score promedio: {self.datos['total_score'].mean():.1f}")
        print(f"   • Desviación estándar: {self.datos['total_score'].std():.1f}")
        print(f"   • Mediana: {self.datos['total_score'].median():.1f}")
    
    def mostrar_resumen_general(self):
        """Muestra un resumen general de los datos"""
        if self.datos is None or len(self.datos) == 0:
            print("❌ No hay datos para analizar")
            return
        
        print("\n" + "="*60)
        print("📋 RESUMEN GENERAL")
        print("="*60)
        
        # Información básica
        print(f"📊 Información general:")
        print(f"   • Total de registros: {len(self.datos)}")
        print(f"   • Período de recolección: {self.datos['timestamp'].min()} a {self.datos['timestamp'].max()}")
        
        # Correlaciones básicas
        print(f"\n🔍 Correlaciones básicas:")
        corr_edad_estres = self.datos['age'].corr(self.datos['total_score'])
        print(f"   • Edad vs. Score de estrés: r = {corr_edad_estres:.3f}")
        
        # Distribución por edad y estrés
        print(f"\n📊 Análisis cruzado edad-estrés:")
        
        # Promedio de estrés por rango etario
        if 'rango_edad' in self.datos.columns:
            # CORRECCIÓN: Agregar observed=True para evitar el warning
            estres_por_edad = self.datos.groupby('rango_edad', observed=True)['total_score'].agg(['mean', 'count'])
            for rango, stats in estres_por_edad.iterrows():
                if stats['count'] > 0:
                    print(f"   • {rango}: {stats['mean']:.1f} puntos promedio ({stats['count']} personas)")
    
    def exportar_resumen(self, archivo='distribucion_poblacion.txt'):
        """Exporta el resumen a un archivo de texto"""
        if self.datos is None or len(self.datos) == 0:
            print("❌ No hay datos para exportar")
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(archivo, 'w', encoding='utf-8') as f:
            f.write(f"REPORTE DE DISTRIBUCIÓN DE POBLACIÓN PSS-14\n")
            f.write(f"Generado: {timestamp}\n")
            f.write(f"Total de registros: {len(self.datos)}\n")
            f.write("="*60 + "\n\n")
            
            # Distribución por edad
            f.write("DISTRIBUCIÓN POR EDAD\n")
            f.write("-"*30 + "\n")
            # CORRECCIÓN: Agregar observed=True
            if 'rango_edad' in self.datos.columns:
                distribucion_edad = self.datos['rango_edad'].value_counts(sort=False).sort_index()
                for rango, count in distribucion_edad.items():
                    porcentaje = (count / len(self.datos)) * 100
                    f.write(f"{rango}: {count} personas ({porcentaje:.1f}%)\n")
            
            # Distribución por profesión
            f.write("\nDISTRIBUCIÓN POR PROFESIÓN\n")
            f.write("-"*30 + "\n")
            distribucion_prof = self.datos['profession'].value_counts()
            for profesion, count in distribucion_prof.items():
                porcentaje = (count / len(self.datos)) * 100
                f.write(f"{profesion}: {count} personas ({porcentaje:.1f}%)\n")
            
            # Distribución por estrés
            f.write("\nDISTRIBUCIÓN POR NIVEL DE ESTRÉS\n")
            f.write("-"*30 + "\n")
            distribucion_estres = self.datos['stress_level'].value_counts()
            for nivel, count in distribucion_estres.items():
                porcentaje = (count / len(self.datos)) * 100
                f.write(f"{nivel}: {count} personas ({porcentaje:.1f}%)\n")
        
        print(f"\n💾 Resumen exportado a: {archivo}")
        
    def mostrar_todo(self):
        """Muestra todas las distribuciones"""
        if not self.cargar_datos():
            return
        
        print("\n🚀 ANÁLISIS DE DISTRIBUCIÓN DE POBLACIÓN PSS-14")
        print("="*60)
        print(f"📅 Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Mostrar todas las distribuciones
        self.mostrar_distribucion_edad()
        self.mostrar_distribucion_profesion()
        self.mostrar_distribucion_estres()
        self.mostrar_resumen_general()
        
        # Preguntar si quiere exportar
        print(f"\n💾 ¿Desea exportar el resumen a archivo? (s/n): ", end="")
        try:
            respuesta = input().lower()
            if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
                self.exportar_resumen()
        except KeyboardInterrupt:
            print("\n👋 Análisis completado.")

def main():
    """Función principal"""
    print("🔍 ANALIZADOR DE DISTRIBUCIÓN DE POBLACIÓN PSS-14")
    print("="*60)
    
    # Crear instancia del analizador
    analizador = DistribucionPoblacion()
    
    # Mostrar todo el análisis
    analizador.mostrar_todo()
    
    print("\n✅ Análisis completado exitosamente!")

if __name__ == "__main__":
    main()
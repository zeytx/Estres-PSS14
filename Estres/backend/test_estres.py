from database import init_db, save_to_db
from csv_handler import save_to_csv

# Lista de preguntas CORRECTAMENTE ESCRITA
PREGUNTAS = [
    "1. ¿Con qué frecuencia ha estado molesto porque algo que sucedió inesperadamente?",
    "2. ¿Con qué frecuencia ha sentido que no podía controlar las cosas importantes en su vida?",
    "3. ¿Con qué frecuencia se ha sentido nervioso o estresado?",
    "4. ¿Con qué frecuencia ha confiado en su capacidad para manejar sus problemas personales?",  # Invertida
    "5. ¿Con qué frecuencia ha sentido que las cosas le iban bien?",  # Invertida
    "6. ¿Con qué frecuencia ha encontrado que no podía hacer frente a todas las cosas que tenía que hacer?",
    "7. ¿Con qué frecuencia ha podido controlar las irritaciones en su vida?",  # Invertida
    "8. ¿Con qué frecuencia ha sentido que estaba en la cima de las cosas?",  # Invertida
    "9. ¿Con qué frecuencia se ha enfadado porque las cosas que sucedieron estaban fuera de su control?",
    "10. ¿Con qué frecuencia ha pensado que las dificultades se acumulaban tanto que no podía superarlas?",
    "11. ¿Con qué frecuencia ha sentido que podía controlar la manera en que pasaban las cosas?",  # Invertida
    "12. ¿Con qué frecuencia ha sentido que las dificultades no se podían superar?", 
    "13. ¿Con qué frecuencia ha sentido que no podía afrontar sus responsabilidades?", 
    "14. ¿Con qué frecuencia ha sentido que todo estaba bajo control?"  # Invertida
]

def realizar_test():
    init_db()  # Inicializar la base de datos
    
    print("\n=== TEST DE ESTRÉS PSS-14 ===")
    
    # 1. Recoger datos demográficos
    while True:
        try:
            edad = int(input("\nIngrese su edad: "))
            if 1 <= edad <= 120:
                break
            print("Por favor ingrese una edad válida (1-120).")
        except ValueError:
            print("Debe ingresar un número válido.")
    
    profesion = input("Ingrese su profesión: ").strip()
    
    # 2. Mostrar instrucciones
    print("\nInstrucciones:")
    print("Por favor califique cada afirmación de 0 a 4, donde:")
    print("0 = Nunca, 1 = Casi nunca, 2 = A veces, 3 = Frecuentemente, 4 = Muy frecuentemente\n")
    
    # 3. Recoger respuestas
    respuestas = []
    for i, pregunta in enumerate(PREGUNTAS):  # Usando la variable correcta PREGUNTAS
        while True:
            try:
                respuesta = int(input(f"{pregunta} (0-4): "))
                if 0 <= respuesta <= 4:
                    respuestas.append(respuesta)
                    break
                print("¡Error! Ingrese un valor entre 0 y 4.")
            except ValueError:
                print("¡Error! Debe ingresar un número.")
    
    # 4. Procesar y guardar resultados
    test_id, puntuacion, nivel = save_to_db(edad, profesion, respuestas)
    save_to_csv(edad, profesion, respuestas)
    
    # 5. Mostrar resultados
    print("\n=== RESULTADOS ===")
    print(f"ID del Test: {test_id}")
    print(f"Puntuación Total: {puntuacion}/56")
    print(f"Nivel de Estrés: {nivel}")
    
    input("\nPresione Enter para finalizar...")

if __name__ == "__main__":
    realizar_test()
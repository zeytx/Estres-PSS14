document.addEventListener('DOMContentLoaded', () => {
    // Lista completa de preguntas PSS-14
    const questions = [
        "1. ¿Con qué frecuencia ha estado molesto porque algo que sucedió inesperadamente?",
        "2. ¿Con qué frecuencia ha sentido que no podía controlar las cosas importantes en su vida?",
        "3. ¿Con qué frecuencia se ha sentido nervioso o estresado?",
        "4. ¿Con qué frecuencia ha confiado en su capacidad para manejar sus problemas personales?",
        "5. ¿Con qué frecuencia ha sentido que las cosas le iban bien?",
        "6. ¿Con qué frecuencia ha encontrado que no podía hacer frente a todas las cosas que tenía que hacer?",
        "7. ¿Con qué frecuencia ha podido controlar las irritaciones en su vida?",
        "8. ¿Con qué frecuencia ha sentido que estaba en la cima de las cosas?",
        "9. ¿Con qué frecuencia se ha enfadado porque las cosas que sucedieron estaban fuera de su control?",
        "10. ¿Con qué frecuencia ha pensado que las dificultades se acumulaban tanto que no podía superarlas?",
        "11. ¿Con qué frecuencia ha sentido que podía controlar la manera en que pasaban las cosas?",
        "12. ¿Con qué frecuencia ha sentido que las dificultades no se podían superar?", 
        "13. ¿Con qué frecuencia ha sentido que no podía afrontar sus responsabilidades?", 
        "14. ¿Con qué frecuencia ha sentido que todo estaba bajo control?"
    ];
    
    const container = document.getElementById('questionsContainer');

    // CREAR SELECT DE PROFESIONES PRIMERO
    const professionContainer = document.getElementById('professionContainer');
    const professionsByCategory = {
  "Ingenierías": [
    "Ingeniero Agrónomo", "Ingeniero Ambiental", "Ingeniero Biomédico", "Ingeniero Civil", "Ingeniero en Computación", "Ingeniero Eléctrico",
    "Ingeniero Electrónico", "Ingeniero en Telecomunicaciones", "Ingeniero Industrial", "Ingeniero Mecánico", "Ingeniero Mecatrónico",
    "Ingeniero Químico", "Ingeniero Petrolero", "Ingeniero Naval", "Ingeniero Aeronáutico", "Ingeniero de Sistemas", "Ingeniero de Software",
    "Ingeniero en Alimentos", "Ingeniero Forestal", "Ingeniero Geólogo", "Ingeniero Minero", "Ingeniero Metalúrgico", "Ingeniero Pesquero",
    "Ingeniero Textil", "Ingeniero Acústico", "Ingeniero Automotriz", "Ingeniero de Materiales", "Ingeniero Genético", "Ingeniero en Energías Renovables",
    "Ingeniero en Robótica", "Ingeniero en FinTech"
  ],
  "Ciencias de la Salud": [
    "Médico General", "Cirujano", "Pediatra", "Cardiólogo", "Neurólogo", "Psiquiatra", "Enfermero/a", "Odontólogo", "Veterinario", "Nutriólogo",
    "Fisioterapeuta", "Biólogo", "Químico Farmacéutico", "Técnico en Enfermería", "Partera/Comadrona", "Logopeda", "Terapeuta Ocupacional",
    "Farmacéutico/a", "Especialista en Biotecnología", "Masoterapeuta"
  ],
  "Tecnología y Ciencias de la Computación": [
    "Programador", "Desarrollador Web", "Analista de Datos", "Especialista en Inteligencia Artificial", "Administrador de Redes", "Diseñador UX/UI",
    "Especialista en Ciberseguridad", "Técnico en Computación", "Técnico en Telecomunicaciones", "Desarrollador de Videojuegos", "Especialista en Big Data",
    "Desarrollador de Software", "Arquitecto de Computación", "Especialista en E-commerce", "Desarrollador de Aplicaciones Móviles"
  ],
  "Ciencias Sociales y Humanidades": [
    "Psicólogo", "Sociólogo", "Antropólogo", "Economista", "Abogado", "Profesor", "Trabajador Social",
    "Historiador/a", "Politólogo/a", "Geógrafo/a", "Pedagogo/a", "Notario/a", "Juez/a", "Investigador/a Criminológico/a",
    "Perito en Lingüística Forense"
  ],
  "Artes y Humanidades": [
    "Arquitecto", "Diseñador Gráfico", "Escritor", "Músico", "Actor", "Artista Plástico", "Diseñador de Moda",
    "Fotógrafo/a", "Compositor/a Musical", "Director/a de Museos", "Modelo", "Escultor/a", "Pintor/a",
    "Bailarín/a", "Curador/a de Arte"
  ],
  "Oficios Técnicos y Manuales": [
    "Técnico Electricista", "Técnico Mecánico", "Técnico en Computación", "Técnico en Telecomunicaciones", "Carpintero/a",
    "Plomero/a", "Soldador/a", "Albañil", "Mecánico Automotriz", "Técnico en Refrigeración", "Técnico en Electrónica",
    "Técnico en Energías Renovables", "Operador/a de Maquinaria Pesada", "Instalador/a de Paneles Solares", "Técnico en Climatización"
  ],
  "Educación y Formación": [
    "Maestro/a de Educación Primaria", "Maestro/a de Educación Secundaria", "Profesor/a Universitario/a", "Educador/a Infantil",
    "Pedagogo/a", "Orientador/a Educativo/a", "Instructor/a de Formación Profesional", "Docente de Educación Especial",
    "Formador/a de Adultos", "Tutor/a en Línea", "Diseñador/a Instruccional"
  ],
  "Ciencias Naturales y Exactas": [
    "Físico/a", "Químico/a", "Biólogo/a", "Matemático/a", "Geólogo/a", "Astrónomo/a", "Bioquímico/a",
    "Oceanógrafo/a", "Meteorólogo/a", "Estadístico/a"
  ],
  "Administración y Negocios": [
    "Administrador/a de Empresas", "Contador/a", "Auditor/a", "Analista Financiero/a", "Asesor/a Financiero/a", "Gerente de Proyectos",
    "Consultor/a de Negocios", "Especialista en Recursos Humanos", "Agente de Seguros", "Corredor/a de Bolsa", "Especialista en Logística",
    "Gerente de Marketing", "Analista de Riesgos", "Planificador/a Estratégico/a"
  ],
  "Comunicación y Medios": [
    "Periodista", "Comunicador/a Social", "Relaciones Públicas", "Locutor/a", "Presentador/a de Televisión", "Editor/a",
    "Redactor/a", "Guionista", "Productor/a Audiovisual", "Community Manager", "Especialista en Marketing Digital",
    "Diseñador/a de Contenido"
  ],
  "Transporte y Logística": [
    "Piloto de Aeronaves", "Controlador/a Aéreo/a", "Conductor/a de Transporte Público", "Operador/a de Grúa",
    "Logístico/a", "Despachador/a de Vuelos", "Capitán de Barco", "Técnico/a en Mantenimiento Aeronáutico",
    "Supervisor/a de Tráfico", "Coordinador/a de Logística"
  ],
  "Servicios y Atención al Cliente": [
    "Recepcionista", "Cajero/a", "Asistente Administrativo/a", "Atención al Cliente", "Call Center", "Anfitrión/a",
    "Azafata/Auxiliar de Vuelo", "Conserje", "Guía Turístico/a", "Agente de Viajes", "Barista", "Mesero/a", "Bartender"
  ],
  "Seguridad y Defensa": [
    "Policía", "Bombero/a", "Militar", "Guardia de Seguridad", "Detective Privado", "Agente de Aduanas", "Oficial de Protección Civil", "Especialista en Seguridad Informática",
    "Instructor/a de Defensa Personal", "Analista de Inteligencia"
  ],
  "Agricultura, Ganadería y Pesca": [
    "Agricultor/a", "Ganadero/a", "Pescador/a", "Ingeniero/a Agrónomo/a", "Técnico/a Agropecuario/a", "Apicultor/a", "Silvicultor/a","Operador/a de Maquinaria Agrícola",
    "Especialista en Acuicultura", "Inspector/a de Calidad Agroalimentaria"
  ],
  "Ciencias Jurídicas y Políticas": [
    "Abogado/a", "Juez/a", "Fiscal", "Notario/a", "Defensor/a Público/a", "Procurador/a", "Asesor/a Legal", "Diplomático/a",
    "Funcionario/a Público/a", "Analista Político/a"
  ],
  "Ciencias Económicas y Financieras": [
    "Economista", "Contador/a", "Auditor/a", "Analista Financiero/a", "Asesor/a de Inversiones", "Corredor/a de Bolsa",
    "Especialista en Comercio Internacional", "Consultor/a Económico/a", "Gestor/a de Patrimonios", "Investigador/a Económico/a"
  ],
  "Ciencias de la Información y Documentación": [
    "Bibliotecario/a", "Archivista", "Documentalista", "Gestor/a de Información", "Especialista en Gestión del Conocimiento", "Creador/a de Contenidos", "Curador/a de Contenidos",
    "Analista de Información", "Consultor/a en Gestión Documental", "Técnico/a en Museología", "Especialista en Preservación Digital"
  ]
};

    // Crear elemento select de profesiones
    const professionSelect = document.createElement('select');
    professionSelect.id = 'profession';
    professionSelect.name = 'profession';
    professionSelect.required = true;
    professionSelect.className = 'profession-select';
    
    // Añadir opción por defecto
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- Selecciona tu profesión --';
    defaultOption.disabled = true;
    defaultOption.selected = true;
    professionSelect.appendChild(defaultOption);

    // Añadir profesiones organizadas por categorías
    for (const [category, professions] of Object.entries(professionsByCategory)) {
        // Crear optgroup para la categoría
        const optgroup = document.createElement('optgroup');
        optgroup.label = `── ${category} ──`;
        
        // Añadir profesiones a este grupo
        professions.forEach(profession => {
            const option = document.createElement('option');
            option.value = profession;
            option.textContent = profession;
            optgroup.appendChild(option);
        });
        
        professionSelect.appendChild(optgroup);
    }

    // Añadir select al contenedor
    if (professionContainer) {
        professionContainer.appendChild(professionSelect);
    }
    
    // Crear elementos de loading y error si no existen
    let loadingIndicator = document.getElementById('loadingIndicator');
    if (!loadingIndicator) {
        loadingIndicator = document.createElement('div');
        loadingIndicator.id = 'loadingIndicator';
        loadingIndicator.textContent = 'Enviando datos...';
        loadingIndicator.style.display = 'none';
        document.querySelector('.container').insertBefore(loadingIndicator, document.getElementById('testForm'));
    }
    
    let errorContainer = document.getElementById('errorContainer');
    if (!errorContainer) {
        errorContainer = document.createElement('div');
        errorContainer.id = 'errorContainer';
        errorContainer.style.display = 'none';
        errorContainer.style.color = 'red';
        document.querySelector('.container').insertBefore(errorContainer, document.getElementById('testForm'));
    }

    
    // Generar preguntas dinámicamente
    questions.forEach((q, i) => {
        const div = document.createElement('div');
        div.className = 'question-group';
        div.innerHTML = `
            <label>${q}</label>
            <select name="q${i+1}" required class="response-select">
                <option value="" disabled selected>Seleccione</option>
                <option value="0">0 - Nunca</option>
                <option value="1">1 - Casi nunca</option>
                <option value="2">2 - A veces</option>
                <option value="3">3 - Frecuentemente</option>
                <option value="4">4 - Muy frecuentemente</option>
            </select>
            <span class="error-message" id="error-q${i+1}" style="color:red; display:none;">Por favor seleccione una opción</span>
        `;
        container.appendChild(div);
    });
    
    // Validación en tiempo real
    document.querySelectorAll('.response-select').forEach(select => {
        select.addEventListener('change', function() {
            const errorElement = document.getElementById(`error-${this.name}`);
            if (this.value === "") {
                errorElement.style.display = 'block';
            } else {
                errorElement.style.display = 'none';
            }
        });
    });

    // Variables de estado
    let isSubmitting = false;
    let submitCompleted = false;
    
    // Manejar envío del formulario
    document.getElementById('testForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log("Formulario enviado");
        
        const submitBtn = document.getElementById('submitBtn');
        // Marcar como enviando
        isSubmitting = true;
        submitCompleted = false;
        // Deshabilitar el botón inmediatamente para prevenir doble envío
        submitBtn.disabled = true;
        submitBtn.textContent = 'Enviando...';
        submitBtn.style.opacity = '0.6';
        
        // Validar edad
        const ageInput = document.getElementById('age');
        const age = parseInt(ageInput.value);
        if (isNaN(age) || age < 18 || age > 100) {
            alert("Por favor ingrese una edad válida (entre 18 y 100 años)");
            ageInput.focus();
            // Rehabilitar botón si hay error de validación
            submitBtn.disabled = false;
            submitBtn.textContent = 'Enviar Test';
            submitBtn.style.opacity = '1';
            return;
        }
        
        // Validar profesión
        const professionSelect = document.getElementById('profession');
        const profession = professionSelect.value;
        if (profession === "") {
            alert("Por favor seleccione su profesión");
            professionSelect.focus();
            // Rehabilitar botón si hay error de validación
            submitBtn.disabled = false;
            submitBtn.textContent = 'Enviar Test';
            submitBtn.style.opacity = '1';
            return;
        }
        
        // Validar respuestas
        let allAnswered = true;
        const responses = [];
        
        questions.forEach((_, i) => {
            const select = document.querySelector(`select[name="q${i+1}"]`);
            if (select.value === "") {
                document.getElementById(`error-q${i+1}`).style.display = 'block';
                allAnswered = false;
            } else {
                responses.push(parseInt(select.value));
            }
        });
        
        if (!allAnswered) {
            alert("Por favor responda todas las preguntas");
            // Rehabilitar botón si hay error de validación
            submitBtn.disabled = false;
            submitBtn.textContent = 'Enviar Test';
            submitBtn.style.opacity = '1';
            return;
        }
        
        // Mostrar indicador de carga
        loadingIndicator.style.display = 'block';
        errorContainer.style.display = 'none';
        
        try {
            const response = await fetch('/api/submit-test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    age: age,
                    profession: profession,
                    responses: responses
                })
            });
            
            console.log("Respuesta HTTP status:", response.status);
            console.log("Content-Type:", response.headers.get("content-type"));
            
            // Verificar si la respuesta es JSON antes de parsear
            const contentType = response.headers.get("content-type");
            if (!contentType || !contentType.includes("application/json")) {
                const textResponse = await response.text();
                console.error("Respuesta no es JSON:", textResponse);
                throw new Error("El servidor devolvió una respuesta no válida. Verifica la consola del servidor.");
            }
            
            const result = await response.json();
            console.log("Respuesta del servidor:", result);
            
            if (result.success) {
                submitCompleted = true;
                isSubmitting = false;
                // Guardar datos en sessionStorage para la página de resultados
                sessionStorage.setItem('testResults', JSON.stringify({
                    testId: result.test_id,
                    stressLevel: result.stress_level,
                    age: age,
                    profession: profession,
                    timestamp: new Date().toISOString()
                }));
                
                // Cambiar estado del botón a exitoso y mantenerlo deshabilitado permanentemente
                submitBtn.disabled = true;
                submitBtn.textContent = '✓ Test Enviado Exitosamente';
                submitBtn.style.background = '#28a745';
                submitBtn.style.color = 'white';
                submitBtn.style.opacity = '1';
                
                // Mostrar mensaje de éxito temporal
                const successMessage = document.createElement('div');
                successMessage.className = 'success-alert';
                successMessage.innerHTML = `
                    <div style="background: #d4edda; border: 1px solid #c3e6cb; color: #155724; 
                               padding: 15px; border-radius: 5px; margin: 15px 0; text-align: center;">
                        <strong>¡Test enviado exitosamente!</strong><br>
                        Redirigiendo a resultados...
                    </div>
                `;
                document.getElementById('testForm').appendChild(successMessage);
                
                // Redirigir a resultados después de mostrar el mensaje
                setTimeout(() => {
                    window.location.href = result.redirect_url;
                }, 1500);
                
            } else {
                throw new Error(result.error || 'Error al procesar el test');
            }
            
        } catch (error) {
            // En caso de error, permitir salir
            isSubmitting = false;
            submitCompleted = false;
            console.error('Error completo:', error);
            errorContainer.textContent = `Error: ${error.message}`;
            errorContainer.style.display = 'block';
            
            // Solo rehabilitar el botón si hubo un error real (no éxito)
            submitBtn.disabled = false;
            submitBtn.textContent = 'Enviar Test';
            submitBtn.style.opacity = '1';
            submitBtn.style.background = '';
            submitBtn.style.color = '';
            
            // Scroll al error para mejor UX
            errorContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
        } finally {
            loadingIndicator.style.display = 'none';
        }
    });

    // Prevenir cierre accidental durante envío
    window.addEventListener('beforeunload', function(e) {
        // Solo mostrar advertencia si se está enviando y no se ha completado
        if (isSubmitting && !submitCompleted) {
            const message = '¿Estás seguro de querer salir? El test se está enviando y perderás el progreso.';
            e.preventDefault();
            e.returnValue = message;
            return message;
        }
    });
    
    // Si estamos en la página de resultados, cargar los datos
    if (window.location.pathname.includes('/results')) {
        loadResults();
    }
});


// Función para cargar y mostrar resultados
async function loadResults() {
    console.log("Función loadResults ejecutada");
    const resultsContainer = document.getElementById('resultsContainer');
    const loadingResults = document.getElementById('loadingResults');
    const errorResults = document.getElementById('errorResults');
    
    loadingResults.style.display = 'block';
    
    console.log("Ubicación actual:", window.location.pathname);
    
    // Extraer ID del test de la URL
    const testId = window.location.pathname.split('/').pop();
    console.log("Test ID extraído:", testId);
    
    // Eliminar la dependencia del sessionStorage
    try {
        console.log("Solicitando análisis para test ID:", testId);
        const response = await fetch(`/api/get-analysis/${testId}`);
        console.log("Respuesta recibida:", response.status);
        
        if (!response.ok) {
            throw new Error(`Error HTTP: ${response.status}`);
        }
        
        const analysis = await response.json();
        console.log("Análisis recibido:", analysis);
        
        // Obtener datos básicos del análisis
        const age = analysis.age;
        const profession = analysis.profession;
        const stressLevel = analysis.stress_level;
        
        // Mostrar resultados
        resultsContainer.innerHTML = `
            <h2>Resultados de tu Test de Estrés</h2>
            <div class="result-summary">
                <p><strong>Edad:</strong> ${age}</p>
                <p><strong>Profesión:</strong> ${profession}</p>
                <p><strong>Nivel de estrés:</strong> <span class="stress-level ${stressLevel.toLowerCase()}">${stressLevel}</span></p>
            </div>
            
            <div class="analysis-section">
                <h3>Análisis Profesional</h3>
                <p>${analysis.insight || analysis.professional_insight || 'No disponible'}</p>
            </div>
            
            <div class="recommendations-section">
                <h3>Recomendaciones Personalizadas</h3>
                <ol>
                    ${analysis.recommendations && Array.isArray(analysis.recommendations)
                        ? analysis.recommendations.map(r => `<li>${r}</li>`).join('') 
                        : '<li>No hay recomendaciones disponibles</li>'}
                </ol>
            </div>
            
            ${analysis.patterns && Array.isArray(analysis.patterns) ? `
            <div class="patterns-section">
                <h3>Patrones Detectados</h3>
                <ul>
                    ${analysis.patterns.map(p => `<li>${p}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            <button onclick="window.print()" class="print-btn">Imprimir Resultados</button>
        `;
        
    } catch (error) {
        console.error('Error al cargar resultados:', error);
        errorResults.textContent = `Error al cargar resultados: ${error.message}`;
        errorResults.style.display = 'block';
    } finally {
        loadingResults.style.display = 'none';
    }
}
document.addEventListener('DOMContentLoaded', () => {
    if (window.location.pathname.includes('/results/')) {
        console.log("Página de resultados detectada, llamando a loadResults()");
        loadResults();
    }
});
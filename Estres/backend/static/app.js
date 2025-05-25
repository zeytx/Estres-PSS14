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
    
    // Manejar envío del formulario
    document.getElementById('testForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log("Formulario enviado"); // Verifica si llega aquí
        
        // Validar edad
        const ageInput = document.getElementById('age');
        const age = parseInt(ageInput.value);
        if (isNaN(age) || age < 18 || age > 100) {
            alert("Por favor ingrese una edad válida (entre 18 y 100 años)");
            ageInput.focus();
            return;
        }
        
        // Validar profesión
        const professionInput = document.getElementById('profession');
        const profession = professionInput.value.trim();
        if (profession === "") {
            alert("Por favor ingrese su profesión");
            professionInput.focus();
            return;
        }
        
        // Validar respuestas
        let allAnswered = true;
        const responses = [];
        const errorMessages = [];

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
            return;
        }
        
        // Mostrar carga
        loadingIndicator.style.display = 'block';
        errorContainer.style.display = 'none';
        const submitBtn = document.getElementById('submitBtn');
        if (submitBtn) submitBtn.disabled = true;
        
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
            console.log("Headers:", response.headers);
            const result = await response.json();
            console.log("Respuesta del servidor:", result);
            
            /*if (!response.ok) {
                throw new Error(result.error || 'Error desconocido');
            }*/
            
            if (result.success) {
                // Guardar datos en sessionStorage para la página de resultados
                sessionStorage.setItem('testResults', JSON.stringify({
                    testId: result.test_id,
                    stressLevel: result.stress_level,
                    age: age,
                    profession: profession,
                    timestamp: new Date().toISOString()
                }));
                
                // Redirigir a resultados
                window.location.href = result.redirect_url;
            } else {
                throw new Error(result.error || 'Error al procesar el test');
            }
        } catch (error) {
            console.error('Error:', error);
            errorContainer.textContent = `Error: ${error.message}`;
            errorContainer.style.display = 'block';
            // Mostrar error más detallado en consola
            if (error.response) {
                console.error("Detalles del error:", await error.response.text());
            }
        } finally {
            loadingIndicator.style.display = 'none';
            document.getElementById('submitBtn').disabled = false;
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
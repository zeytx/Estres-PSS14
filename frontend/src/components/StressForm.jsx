import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { submitTest } from '../services/api';
import { Loader2, ArrowRight, CheckCircle } from 'lucide-react';

const questions = [
    "1. En el último mes, ¿con qué frecuencia ha estado afectado por algo que ocurrió inesperadamente?",
    "2. En el último mes, ¿con qué frecuencia ha sentido que no podía controlar las cosas importantes en su vida?",
    "3. En el último mes, ¿con qué frecuencia se ha sentido nervioso o estresado?",
    "4. En el último mes, ¿con qué frecuencia ha manejado con éxito los pequeños problemas irritantes de la vida?",
    "5. En el último mes, ¿con qué frecuencia ha sentido que ha afrontado efectivamente los cambios importantes en su vida?",
    "6. En el último mes, ¿con qué frecuencia ha estado seguro sobre su capacidad para manejar sus problemas personales?",
    "7. En el último mes, ¿con qué frecuencia ha sentido que las cosas le van bien?",
    "8. En el último mes, ¿con qué frecuencia ha sentido que no podía afrontar todas las cosas que tenía que hacer?",
    "9. En el último mes, ¿con qué frecuencia ha podido controlar las dificultades de su vida?",
    "10. En el último mes, ¿con qué frecuencia se ha sentido que tenía todo bajo control?",
    "11. En el último mes, ¿con qué frecuencia ha estado enfadado porque las cosas que le ocurrieron estaban fuera de su control?",
    "12. En el último mes, ¿con qué frecuencia ha pensado sobre las cosas que le quedan por hacer?",
    "13. En el último mes, ¿con qué frecuencia ha podido controlar la forma de pasar el tiempo?",
    "14. En el último mes, ¿con qué frecuencia ha sentido que las dificultades se acumulaban tanto que no podía superarlas?",
];

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
        "Agricultor/a", "Ganadero/a", "Pescador/a", "Ingeniero/a Agrónomo/a", "Técnico/a Agropecuario/a", "Apicultor/a", "Silvicultor/a", "Operador/a de Maquinaria Agrícola",
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

const StressForm = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        age: '',
        profession: '',
        responses: Array(14).fill(0), // Default to 0 (Nunca)
        free_text: '' // Texto libre para describir cómo se siente
    });

    const handleResponseChange = (index, value) => {
        const newResponses = [...formData.responses];
        newResponses[index] = parseInt(value);
        setFormData({ ...formData, responses: newResponses });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        // Validation for Custom Profession
        let finalProfession = formData.profession;
        if (formData.profession === 'Otro') {
            const custom = formData.customProfession?.trim();
            if (!custom || custom.length < 4 || /^(no\s*se|ninguna|nada|idk|n\/a)$/i.test(custom)) {
                alert("Por favor especifica una profesión válida.");
                setLoading(false);
                return;
            }
            finalProfession = custom;
        }

        try {
            const result = await submitTest({
                age: parseInt(formData.age),
                profession: finalProfession,
                responses: formData.responses,
                free_text: formData.free_text.trim() || undefined
            });

            if (result.success) {
                // Navigate to results page with token
                const token = result.redirect_url.split('token=')[1];
                navigate(`/results/${result.test_id}?token=${token}`);
            }
        } catch (error) {
            console.error("Error submitting test:", error);
            alert("Hubo un error al enviar el test. Por favor intente nuevamente.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-3xl mx-auto space-y-8">
                <div className="text-center">
                    <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight sm:text-5xl mb-2">
                        Test de Estrés PSS-14
                    </h1>
                    <p className="text-lg text-gray-600">
                        Evalúa tu nivel de estrés percibido con precisión impulsada por IA.
                    </p>
                </div>

                <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100 p-8">
                    <form onSubmit={handleSubmit} className="space-y-8">

                        {/* Demographics Section */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6 bg-blue-50 rounded-xl border border-blue-100">
                            <div>
                                <label htmlFor="age" className="block text-sm font-medium text-gray-700 mb-2">Edad</label>
                                <input
                                    type="number"
                                    id="age"
                                    required
                                    min="12"
                                    max="120"
                                    className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all outline-none"
                                    value={formData.age}
                                    onChange={(e) => setFormData({ ...formData, age: e.target.value })}
                                />
                            </div>
                            <div>
                                <label htmlFor="profession" className="block text-sm font-medium text-gray-700 mb-2">Profesión</label>
                                <select
                                    id="profession"
                                    required
                                    className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all outline-none bg-white"
                                    value={formData.profession}
                                    onChange={(e) => setFormData({ ...formData, profession: e.target.value })}
                                >
                                    <option value="">Selecciona tu profesión</option>
                                    {Object.entries(professionsByCategory).map(([category, professions]) => (
                                        <optgroup key={category} label={`── ${category} ──`}>
                                            {professions.map(p => (
                                                <option key={p} value={p}>{p}</option>
                                            ))}
                                        </optgroup>
                                    ))}
                                    <option value="Otro">Otro (Especificar)</option>
                                </select>
                            </div>
                        </div>

                        {/* Custom Profession Input */}
                        {formData.profession === 'Otro' && (
                            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200 animate-accordion-down">
                                <label htmlFor="customProfession" className="block text-sm font-medium text-yellow-800 mb-2">
                                    Por favor especifica tu profesión o ocupación:
                                </label>
                                <input
                                    type="text"
                                    id="customProfession"
                                    placeholder="Ej: Estudiante de Arquitectura, Ama de casa, Freelance..."
                                    className="w-full px-4 py-2 rounded-lg border border-yellow-300 focus:ring-2 focus:ring-yellow-500 focus:border-transparent transition-all outline-none"
                                    value={formData.customProfession || ''}
                                    onChange={(e) => setFormData({ ...formData, customProfession: e.target.value })}
                                />
                                <p className="text-xs text-yellow-600 mt-1">
                                    * Debe ser una descripción válida (mínimo 4 caracteres). No uses "no sé", "ninguna", etc.
                                </p>
                            </div>
                        )}

                        {/* Free Text - Emotional Context Section */}
                        <div className="p-6 bg-purple-50 rounded-xl border border-purple-100">
                            <label htmlFor="free_text" className="block text-sm font-medium text-purple-800 mb-2">
                                💬 ¿Cómo te sientes? (Opcional)
                            </label>
                            <p className="text-xs text-purple-600 mb-3">
                                Describe con tus propias palabras cómo te has sentido últimamente.
                                Nuestra IA analizará tu estado emocional junto con las respuestas del test para darte un análisis más personalizado.
                            </p>
                            <textarea
                                id="free_text"
                                rows="4"
                                maxLength="2000"
                                placeholder="Ej: Últimamente me he sentido muy abrumado con el trabajo, duermo poco y me cuesta concentrarme..."
                                className="w-full px-4 py-3 rounded-lg border border-purple-200 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all outline-none resize-none"
                                value={formData.free_text}
                                onChange={(e) => setFormData({ ...formData, free_text: e.target.value })}
                            />
                            <p className="text-xs text-purple-400 mt-1 text-right">
                                {formData.free_text.length}/2000 caracteres
                            </p>
                        </div>

                        {/* Questions Section */}
                        <div className="space-y-8">
                            {questions.map((q, idx) => (
                                <div key={idx} className="bg-white p-4 rounded-lg hover:bg-gray-50 transition-colors border-b border-gray-100 last:border-0 pb-6">
                                    <p className="text-gray-800 font-medium mb-4 text-lg">{q}</p>
                                    <div className="grid grid-cols-5 gap-2 sm:gap-4">
                                        {[0, 1, 2, 3, 4].map((val) => (
                                            <button
                                                key={val}
                                                type="button"
                                                onClick={() => handleResponseChange(idx, val)}
                                                className={`
                                                    py-3 rounded-lg text-sm font-medium transition-all duration-200
                                                    flex flex-col items-center justify-center gap-1
                                                    ${formData.responses[idx] === val
                                                        ? 'bg-blue-600 text-white shadow-lg scale-105'
                                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}
                                                `}
                                            >
                                                <span className="text-lg font-bold">{val}</span>
                                                <span className="text-xs opacity-90 hidden sm:block">
                                                    {['Nunca', 'Casi nunca', 'A veces', 'Frecuentemente', 'Muy frecuentemente'][val]}
                                                </span>
                                            </button>
                                        ))}
                                    </div>
                                    <div className="flex justify-between mt-2 px-1 text-xs text-gray-400 sm:hidden">
                                        <span>Nunca</span>
                                        <span>Muy frecuentemente</span>
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="pt-6">
                            <button
                                type="submit"
                                disabled={loading}
                                className={`
                                    w-full flex justify-center items-center py-4 px-6 border border-transparent rounded-xl shadow-lg text-lg font-bold text-white 
                                    bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 
                                    focus:outline-none focus:ring-4 focus:ring-blue-300 transition-all duration-300 transform hover:-translate-y-1
                                    ${loading ? 'opacity-75 cursor-not-allowed' : ''}
                                `}
                            >
                                {loading ? (
                                    <>
                                        <Loader2 className="animate-spin -ml-1 mr-3 h-6 w-6" />
                                        Analizando respuestas con IA...
                                    </>
                                ) : (
                                    <>
                                        Obtener Análisis Completo <ArrowRight className="ml-2 h-6 w-6" />
                                    </>
                                )}
                            </button>
                            <p className="text-center text-sm text-gray-500 mt-4">
                                Sus datos son procesados de forma anónima y segura.
                            </p>
                        </div>
                    </form>
                </div>
            </div>
            {/* Footer discreto */}
            <footer className="text-center py-6 mt-8">
                <p className="text-gray-300 text-xs">
                    © 2025-2026 PSS-14 Stress Predictor — <a href="/admin" className="text-gray-400 hover:text-gray-500 transition-colors">Alvaro Nunez</a>
                </p>
            </footer>
        </div>
    );
};

export default StressForm;

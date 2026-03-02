import React, { useEffect, useState } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { getTestResults } from '../services/api';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import { Loader2, AlertTriangle, ShieldCheck, Activity, Brain, Heart, RotateCcw } from 'lucide-react';

const ResultsDashboard = () => {
    const { id } = useParams();
    const [searchParams] = useSearchParams();
    const token = searchParams.get('token');
    const navigate = useNavigate();

    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const result = await getTestResults(id, token);
                setData(result);
            } catch (err) {
                setError(err.message || 'Error cargando resultados');
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [id, token]);

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
                <div className="text-center">
                    <Loader2 className="h-12 w-12 animate-spin text-blue-600 mx-auto mb-4" />
                    <p className="text-gray-600 text-lg">Generando reporte personalizado...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
                <div className="bg-red-50 p-8 rounded-xl border border-red-200 text-center max-w-md">
                    <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
                    <h2 className="text-xl font-bold text-red-800 mb-2">Error de Acceso</h2>
                    <p className="text-red-600">{error}</p>
                </div>
            </div>
        );
    }

    if (!data) return null;

    // Data processing for charts
    const scoreData = [
        { name: 'Nivel Actual', value: data.ml_score || data.score, color: getStressColor(data.stress_level) },
        { name: 'Restante', value: 56 - (data.ml_score || data.score), color: '#e5e7eb' }
    ];

    const confidence = data.ml_prediction?.confidence || 0.85; // Fallback

    return (
        <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-6xl mx-auto space-y-8">

                {/* Header */}
                <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100 flex flex-col md:flex-row justify-between items-center gap-6">
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900 mb-2">Reporte de Análisis de Estrés</h1>
                        <p className="text-gray-500">ID de Evaluación: #{id} | {new Date().toLocaleDateString()}</p>
                    </div>
                    <div className="flex items-center gap-3 bg-blue-50 px-4 py-2 rounded-full border border-blue-100">
                        <Brain className="h-5 w-5 text-blue-600" />
                        <span className="text-blue-800 font-medium text-sm">Análisis potenciado por IA v2.0</span>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                    {/* Main Score Card - Triangle Visualization */}
                    <div className="lg:col-span-1 bg-white rounded-2xl p-8 shadow-lg border border-gray-100 flex flex-col items-center justify-center relative overflow-hidden transition-all hover:shadow-xl">
                        <div className="absolute top-0 w-full h-2 bg-gradient-to-r from-blue-500 to-purple-500"></div>
                        <h3 className="text-gray-500 font-medium mb-8 uppercase tracking-wider text-sm">Nivel de Estrés Percibido</h3>

                        {/* Custom Triangle Visualization */}
                        <div className="relative w-64 h-64 flex items-center justify-center">
                            <svg width="200" height="200" viewBox="0 0 200 200" className="drop-shadow-xl">
                                {/* Defs for gradients and masks */}
                                <defs>
                                    <linearGradient id="stressGradient" x1="0" x2="0" y1="1" y2="0">
                                        <stop offset="0%" stopColor="#22c55e" /> {/* Green */}
                                        <stop offset="50%" stopColor="#eab308" /> {/* Yellow */}
                                        <stop offset="100%" stopColor="#ef4444" /> {/* Red */}
                                    </linearGradient>
                                    <mask id="fillMask">
                                        {/* This rect grows from bottom to top based on score */}
                                        <rect
                                            x="0"
                                            y={200 - ((data.ml_score || data.score) / 56 * 200)}
                                            width="200"
                                            height="200"
                                            fill="white"
                                            className="transition-all duration-1000 ease-out"
                                        />
                                    </mask>
                                </defs>

                                {/* Background Triangle (Empty/Gray) */}
                                <path
                                    d="M100 10 L190 190 L10 190 Z"
                                    fill="#f3f4f6"
                                    stroke="#e5e7eb"
                                    strokeWidth="2"
                                />

                                {/* Filled Triangle (Masked) */}
                                <path
                                    d="M100 10 L190 190 L10 190 Z"
                                    fill="url(#stressGradient)"
                                    mask="url(#fillMask)"
                                    opacity="0.9"
                                />

                                {/* Levels Markers */}
                                <line x1="60" y1="130" x2="140" y2="130" stroke="white" strokeWidth="1" strokeDasharray="4 2" opacity="0.5" />
                                <text x="145" y="133" fontSize="10" fill="#000000" fontWeight="bold">Moderado</text>

                                <line x1="80" y1="70" x2="120" y2="70" stroke="white" strokeWidth="1" strokeDasharray="4 2" opacity="0.5" />
                                <text x="125" y="73" fontSize="10" fill="#000000" fontWeight="bold">Alto</text>
                            </svg>

                            {/* Score Text Overlay */}
                            <div className="absolute inset-0 flex flex-col items-center justify-center pt-20 pointer-events-none">
                                <span className={`text-4xl font-extrabold text-black drop-shadow-sm`}>
                                    {data.ml_score || data.score}
                                </span>
                                <span className="text-gray-400 text-xs font-medium">de 56</span>
                            </div>
                        </div>

                        <div className={`mt-4 px-6 py-2 rounded-full font-bold text-lg border ${getStressBadgeParams(data.stress_level)} transition-transform hover:scale-105`}>
                            {data.stress_level.toUpperCase()}
                        </div>

                        <div className="mt-6 w-full pt-6 border-t border-gray-100">
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-gray-500">Confianza del modelo:</span>
                                <span className="font-semibold text-gray-700 flex items-center gap-1">
                                    <ShieldCheck className="h-4 w-4 text-green-500" />
                                    {(confidence * 100).toFixed(1)}%
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* AI Insights Panel */}
                    <div className="lg:col-span-2 bg-white rounded-2xl p-8 shadow-sm border border-gray-100 flex flex-col gap-6">
                        <div className="flex items-center gap-3 mb-2">
                            <Activity className="h-6 w-6 text-indigo-600" />
                            <h2 className="text-xl font-bold text-gray-900">Análisis e Insights GPT</h2>
                        </div>

                        <div className="bg-indigo-50 p-6 rounded-xl border border-indigo-100">
                            <p className="text-gray-800 leading-relaxed italic">
                                "{data.analysis?.insight || "Analizando patrones..."}"
                            </p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-2">
                            <div>
                                <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                                    <span className="w-2 h-8 bg-blue-500 rounded-full"></span>
                                    Recomendaciones Clave
                                </h4>
                                <ul className="space-y-3">
                                    {data.analysis?.recommendations?.map((rec, i) => (
                                        <li key={i} className="flex gap-3 text-gray-600 text-sm">
                                            <span className="text-blue-500 font-bold">•</span>
                                            {rec}
                                        </li>
                                    )) || <li>Cargando recomendaciones...</li>}
                                </ul>
                            </div>

                            <div>
                                <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                                    <span className="w-2 h-8 bg-purple-500 rounded-full"></span>
                                    Patrones Detectados
                                </h4>
                                <ul className="space-y-3">
                                    {data.analysis?.patterns?.map((pat, i) => (
                                        <li key={i} className="flex gap-3 text-gray-600 text-sm">
                                            <span className="text-purple-500 font-bold">•</span>
                                            {pat}
                                        </li>
                                    )) || <li>Analizando patrones...</li>}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Emotional Analysis Section */}
                {data.analysis?.emotional_analysis && data.analysis.emotional_analysis !== 'No proporcionado' && (
                    <div className="bg-white rounded-2xl p-8 shadow-sm border border-purple-100">
                        <div className="flex items-center gap-3 mb-4">
                            <Heart className="h-6 w-6 text-purple-600" />
                            <h3 className="text-xl font-bold text-gray-900">Análisis Emocional</h3>
                        </div>
                        <div className="bg-purple-50 p-6 rounded-xl border border-purple-100">
                            <p className="text-gray-800 leading-relaxed">
                                {data.analysis.emotional_analysis}
                            </p>
                        </div>
                    </div>
                )}

                {/* Professional Advice Section */}
                {data.analysis?.professional_advice && (
                    <div className="bg-gradient-to-r from-gray-900 to-gray-800 rounded-2xl p-8 text-white shadow-xl">
                        <h3 className="text-lg font-bold mb-3 text-blue-200 uppercase tracking-wide">Consejo Profesional ({data.profession})</h3>
                        <p className="text-gray-300 leading-relaxed text-lg">
                            {data.analysis.professional_advice}
                        </p>
                    </div>
                )}

                {/* Botón Repetir Test */}
                <div className="flex flex-col items-center gap-4 pt-4 pb-8">
                    <button
                        onClick={() => navigate('/')}
                        className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold text-lg rounded-xl shadow-lg transition-all duration-300 transform hover:-translate-y-1 hover:shadow-xl"
                    >
                        <RotateCcw className="h-5 w-5" />
                        Realizar otro test
                    </button>
                    <p className="text-gray-400 text-sm">
                        Tus resultados anteriores quedan guardados de forma segura.
                    </p>
                </div>
            </div>
        </div>
    );
};

// Helpers
const getStressColor = (level) => {
    switch (level?.toLowerCase()) {
        case 'bajo': return '#22c55e'; // Green
        case 'moderado': return '#eab308'; // Yellow
        case 'alto': return '#ef4444'; // Red
        default: return '#3b82f6';
    }
};

const getStressBadgeParams = (level) => {
    switch (level?.toLowerCase()) {
        case 'bajo': return 'bg-green-50 text-green-700 border-green-200';
        case 'moderado': return 'bg-yellow-50 text-yellow-700 border-yellow-200';
        case 'alto': return 'bg-red-50 text-red-700 border-red-200';
        default: return 'bg-gray-100 text-gray-700';
    }
};

const getTextColor = (level) => {
    switch (level?.toLowerCase()) {
        case 'bajo': return 'text-green-600';
        case 'moderado': return 'text-yellow-600';
        case 'alto': return 'text-red-600';
        default: return 'text-gray-600';
    }
};

export default ResultsDashboard;

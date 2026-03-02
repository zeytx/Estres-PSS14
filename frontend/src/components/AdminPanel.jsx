import { useState, useEffect, useCallback } from 'react';
import { Lock, LogOut, BarChart3, Brain, Users, RefreshCw, Clock, TrendingUp, AlertTriangle, CheckCircle, Activity } from 'lucide-react';

const API_BASE = '';

function AdminLogin({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_BASE}/api/admin/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ username, password })
      });
      const data = await res.json();
      if (res.ok && data.success) {
        onLogin();
      } else {
        setError(data.error || 'Error de autenticación');
      }
    } catch {
      setError('Error de conexión al servidor');
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 w-full max-w-md border border-white/20 shadow-2xl">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-purple-500/30 rounded-full flex items-center justify-center mx-auto mb-4">
            <Lock className="w-8 h-8 text-purple-300" />
          </div>
          <h1 className="text-2xl font-bold text-white">Panel de Administración</h1>
          <p className="text-purple-300 text-sm mt-1">PSS-14 Stress Predictor</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-purple-200 text-sm mb-1">Usuario</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-500"
              placeholder="Ingresa tu usuario"
              required
            />
          </div>
          <div>
            <label className="block text-purple-200 text-sm mb-1">Contraseña</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-500"
              placeholder="Ingresa tu contraseña"
              required
            />
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-300 bg-red-500/10 px-3 py-2 rounded-lg text-sm">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800 text-white font-semibold rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Lock className="w-4 h-4" />}
            {loading ? 'Verificando...' : 'Iniciar Sesión'}
          </button>
        </form>
      </div>
    </div>
  );
}

function StatCard({ icon, title, value, subtitle, color = 'purple' }) {
  const IconComponent = icon;
  const colors = {
    purple: 'from-purple-500/20 to-purple-600/10 border-purple-500/30',
    blue: 'from-blue-500/20 to-blue-600/10 border-blue-500/30',
    green: 'from-green-500/20 to-green-600/10 border-green-500/30',
    amber: 'from-amber-500/20 to-amber-600/10 border-amber-500/30',
    red: 'from-red-500/20 to-red-600/10 border-red-500/30',
  };

  return (
    <div className={`bg-gradient-to-br ${colors[color]} border rounded-xl p-5`}>
      <div className="flex items-center gap-3 mb-2">
        <IconComponent className="w-5 h-5 text-white/70" />
        <span className="text-white/60 text-sm">{title}</span>
      </div>
      <p className="text-3xl font-bold text-white">{value}</p>
      {subtitle && <p className="text-white/50 text-xs mt-1">{subtitle}</p>}
    </div>
  );
}

function AdminDashboard({ onLogout }) {
  const [stats, setStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [recentTests, setRecentTests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, modelRes, testsRes] = await Promise.all([
        fetch(`${API_BASE}/api/admin/stats`, { credentials: 'include' }),
        fetch(`${API_BASE}/api/admin/model-info`, { credentials: 'include' }),
        fetch(`${API_BASE}/api/admin/recent-tests?limit=30`, { credentials: 'include' })
      ]);

      if (statsRes.ok) setStats(await statsRes.json());
      if (modelRes.ok) setModelInfo(await modelRes.json());
      if (testsRes.ok) {
        const data = await testsRes.json();
        setRecentTests(data.tests || []);
      }
    } catch (err) {
      console.error('Error fetching admin data:', err);
    }
    setLoading(false);
  }, []);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { fetchData(); }, []);

  const handleRetrain = async () => {
    if (!confirm('¿Reentrenar el modelo? Esto puede tomar varios minutos.')) return;
    setRetraining(true);
    setRetrainResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/admin/retrain`, {
        method: 'POST',
        credentials: 'include'
      });
      const data = await res.json();
      setRetrainResult(data);
      if (data.success) fetchData();
    } catch {
      setRetrainResult({ error: 'Error de conexión' });
    }
    setRetraining(false);
  };

  const handleLogout = async () => {
    await fetch(`${API_BASE}/api/admin/logout`, { method: 'POST', credentials: 'include' });
    onLogout();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <RefreshCw className="w-8 h-8 text-purple-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="bg-white/5 border-b border-white/10 px-6 py-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-purple-400" />
            <h1 className="text-xl font-bold text-white">Admin Dashboard</h1>
            <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded-full">
              {stats?.database || 'N/A'}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <button onClick={fetchData} className="text-white/60 hover:text-white transition-colors">
              <RefreshCw className="w-5 h-5" />
            </button>
            <button onClick={handleLogout} className="flex items-center gap-2 text-red-300 hover:text-red-200 transition-colors text-sm">
              <LogOut className="w-4 h-4" /> Cerrar Sesión
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard icon={Users} title="Total Tests" value={stats.total_tests} subtitle="Evaluaciones completadas" color="purple" />
            <StatCard icon={TrendingUp} title="Score Promedio" value={`${stats.avg_score}/56`} subtitle="Escala PSS-14" color="blue" />
            <StatCard icon={Clock} title="Tests Hoy" value={stats.tests_today} subtitle={new Date().toLocaleDateString()} color="green" />
            <StatCard icon={Brain} title="Modelo" value={modelInfo?.model_type || 'N/A'} subtitle={modelInfo?.model_loaded ? 'Cargado' : 'No cargado'} color="amber" />
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-2 border-b border-white/10 pb-2">
          {[
            { id: 'overview', label: 'Resumen', icon: BarChart3 },
            { id: 'model', label: 'Modelo ML', icon: Brain },
            { id: 'tests', label: 'Tests Recientes', icon: Users },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors ${
                activeTab === tab.id
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                  : 'text-white/50 hover:text-white/80'
              }`}
            >
              <tab.icon className="w-4 h-4" /> {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && stats && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Distribución de estrés */}
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-purple-400" /> Distribución de Niveles
              </h3>
              <div className="space-y-3">
                {Object.entries(stats.stress_distribution).map(([level, count]) => {
                  const pct = stats.total_tests > 0 ? ((count / stats.total_tests) * 100).toFixed(1) : 0;
                  const colors = { 'Bajo': 'bg-green-500', 'Moderado': 'bg-amber-500', 'Alto': 'bg-red-500' };
                  return (
                    <div key={level}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-white/80">{level}</span>
                        <span className="text-white/50">{count} ({pct}%)</span>
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-2.5">
                        <div className={`${colors[level] || 'bg-purple-500'} h-2.5 rounded-full transition-all`}
                             style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Top Profesiones */}
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                <Users className="w-5 h-5 text-blue-400" /> Top Profesiones
              </h3>
              <div className="space-y-2">
                {Object.entries(stats.top_professions).map(([prof, count], i) => (
                  <div key={prof} className="flex justify-between items-center text-sm">
                    <span className="text-white/70">
                      <span className="text-white/30 mr-2">#{i + 1}</span>{prof}
                    </span>
                    <span className="text-purple-300 font-mono">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'model' && modelInfo && (
          <div className="space-y-6">
            {/* Info del modelo */}
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-amber-400" /> Información del Modelo
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-white/50 block">Tipo</span>
                  <span className="text-white font-mono">{modelInfo.model_type}</span>
                </div>
                <div>
                  <span className="text-white/50 block">Versión</span>
                  <span className="text-white font-mono">{modelInfo.model_version || 'N/A'}</span>
                </div>
                <div>
                  <span className="text-white/50 block">Features</span>
                  <span className="text-white font-mono">{modelInfo.features_count}</span>
                </div>
                <div>
                  <span className="text-white/50 block">Último Entrenamiento</span>
                  <span className="text-white font-mono text-xs">{modelInfo.last_training?.split('T')[0] || 'Nunca'}</span>
                </div>
              </div>

              {/* Métricas */}
              {modelInfo.latest_metrics && (
                <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { label: 'MAE', value: modelInfo.latest_metrics.mae?.toFixed(2), good: v => v < 2 },
                    { label: 'Precision', value: (modelInfo.latest_metrics.precision * 100)?.toFixed(1) + '%', good: v => v > 90 },
                    { label: 'Recall', value: (modelInfo.latest_metrics.recall * 100)?.toFixed(1) + '%', good: v => v > 90 },
                    { label: 'F1 Score', value: (modelInfo.latest_metrics.f1 * 100)?.toFixed(1) + '%', good: v => v > 90 },
                  ].map(metric => (
                    <div key={metric.label} className="bg-white/5 rounded-lg p-3 text-center">
                      <span className="text-white/50 text-xs block">{metric.label}</span>
                      <span className="text-2xl font-bold text-white">{metric.value || 'N/A'}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Feature Importance */}
            {modelInfo.feature_importance?.length > 0 && (
              <div className="bg-white/5 border border-white/10 rounded-xl p-6">
                <h3 className="text-white font-semibold mb-4">Feature Importance (Top 15)</h3>
                <div className="space-y-2">
                  {modelInfo.feature_importance.map((f, i) => (
                    <div key={f.feature} className="flex items-center gap-3">
                      <span className="text-white/30 text-xs w-6 text-right">{i + 1}</span>
                      <span className="text-white/70 text-sm w-48 truncate font-mono">{f.feature}</span>
                      <div className="flex-1 bg-white/10 rounded-full h-2">
                        <div className="bg-purple-500 h-2 rounded-full" style={{ width: `${(f.importance * 100 / (modelInfo.feature_importance[0]?.importance || 1))}%` }} />
                      </div>
                      <span className="text-purple-300 font-mono text-xs w-16 text-right">{(f.importance * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Botón de reentrenamiento */}
            <div className="bg-white/5 border border-white/10 rounded-xl p-6">
              <h3 className="text-white font-semibold mb-2">Reentrenar Modelo</h3>
              <p className="text-white/50 text-sm mb-4">
                Reentrenar con todos los datos actuales. Esto evaluará RandomForest, GradientBoosting, XGBoost y LightGBM.
              </p>
              <button
                onClick={handleRetrain}
                disabled={retraining}
                className="flex items-center gap-2 px-6 py-3 bg-amber-600 hover:bg-amber-700 disabled:bg-amber-800 text-white rounded-lg transition-colors"
              >
                <RefreshCw className={`w-4 h-4 ${retraining ? 'animate-spin' : ''}`} />
                {retraining ? 'Reentrenando... (puede tomar 15-20 min)' : 'Reentrenar Modelo'}
              </button>

              {retrainResult && (
                <div className={`mt-4 p-4 rounded-lg text-sm ${
                  retrainResult.success ? 'bg-green-500/10 border border-green-500/30' : 'bg-red-500/10 border border-red-500/30'
                }`}>
                  {retrainResult.success ? (
                    <div className="flex items-center gap-2 text-green-300">
                      <CheckCircle className="w-4 h-4" />
                      <span>Modelo reentrenado exitosamente</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-red-300">
                      <AlertTriangle className="w-4 h-4" />
                      <span>{retrainResult.error || 'Error en reentrenamiento'}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'tests' && (
          <div className="bg-white/5 border border-white/10 rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-white/5">
                  <tr className="text-white/50 text-left">
                    <th className="px-4 py-3">ID</th>
                    <th className="px-4 py-3">Edad</th>
                    <th className="px-4 py-3">Profesión</th>
                    <th className="px-4 py-3">Score</th>
                    <th className="px-4 py-3">Nivel</th>
                    <th className="px-4 py-3">ML Score</th>
                    <th className="px-4 py-3">Fecha</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {recentTests.map(test => {
                    const levelColors = {
                      'Bajo': 'text-green-400 bg-green-500/10',
                      'Moderado': 'text-amber-400 bg-amber-500/10',
                      'Alto': 'text-red-400 bg-red-500/10'
                    };
                    return (
                      <tr key={test.test_id} className="text-white/70 hover:bg-white/5">
                        <td className="px-4 py-3 font-mono">{test.test_id}</td>
                        <td className="px-4 py-3">{test.age}</td>
                        <td className="px-4 py-3 max-w-[200px] truncate">{test.profession}</td>
                        <td className="px-4 py-3 font-mono">{test.total_score || test.score}/56</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 rounded text-xs ${levelColors[test.stress_level || test.level] || 'text-white/50'}`}>
                            {test.stress_level || test.level}
                          </span>
                        </td>
                        <td className="px-4 py-3 font-mono text-purple-300">{test.ml_score ? test.ml_score.toFixed(1) : '—'}</td>
                        <td className="px-4 py-3 text-xs text-white/40">{test.timestamp?.split('T')[0] || test.timestamp?.split(' ')[0]}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default function AdminPanel() {
  const [authenticated, setAuthenticated] = useState(false);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/api/admin/check`, { credentials: 'include' })
      .then(res => {
        if (res.ok) setAuthenticated(true);
        setChecking(false);
      })
      .catch(() => setChecking(false));
  }, []);

  if (checking) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <RefreshCw className="w-8 h-8 text-purple-400 animate-spin" />
      </div>
    );
  }

  if (!authenticated) {
    return <AdminLogin onLogin={() => setAuthenticated(true)} />;
  }

  return <AdminDashboard onLogout={() => setAuthenticated(false)} />;
}


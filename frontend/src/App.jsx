import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import StressForm from './components/StressForm';
import ResultsDashboard from './components/ResultsDashboard';
import AdminPanel from './components/AdminPanel';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<StressForm />} />
        <Route path="/results/:id" element={<ResultsDashboard />} />
        <Route path="/admin" element={<AdminPanel />} />
      </Routes>
    </Router>
  );
}

export default App;

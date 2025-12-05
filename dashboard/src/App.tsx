import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import References from './pages/References';

const App: React.FC = () => {
  return (
    <Router basename="/multimodal-oil-gas-benchmark">
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/references" element={<References />} />
      </Routes>
    </Router>
  );
};

export default App;
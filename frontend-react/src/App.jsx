import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, PlusCircle, MessageSquare, TrendingUp, PieChart as PieIcon, Settings } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import AddTransaction from './pages/AddTransaction';
import Forecast from './pages/Forecast';
import Chatbot from './pages/Chatbot';

function App() {
  return (
    <Router>
      <AppLayout />
    </Router>
  );
}

// Separate component to use useLocation hook
const AppLayout = () => {
  const { pathname } = useLocation();

  return (
    <div className="flex min-h-screen bg-dark text-gray-100 font-sans selection:bg-primary/30">
      {/* Sidebar */}
      <aside className="w-64 bg-card border-r border-gray-800 flex flex-col fixed h-full z-10 transition-all duration-300">
        <div className="p-6 border-b border-gray-800 flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-tr from-primary to-blue-400 rounded-lg flex items-center justify-center text-white font-bold text-xl">
            $
          </div>
          <span className="text-xl font-bold tracking-tight text-white">
            FinAI <span className="text-secondary font-normal text-sm">Pro</span>
          </span>
        </div>

        <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 px-2">
            Overview
          </div>
          <NavLink to="/" icon={<LayoutDashboard size={18} />} label="Dashboard" active={pathname === '/'} />
          <NavLink to="/add" icon={<PlusCircle size={18} />} label="Add Transaction" active={pathname === '/add'} />

          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 mt-6 px-2">
            Intelligence
          </div>
          <NavLink to="/predict" icon={<TrendingUp size={18} />} label="AI Forecaster" active={pathname === '/predict'} />
          <NavLink to="/chat" icon={<MessageSquare size={18} />} label="AI Advisor" active={pathname === '/chat'} />
        </nav>

        <div className="p-4 border-t border-gray-800">
          <div className="flex items-center gap-3 p-2 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
              S
            </div>
            <div className="overflow-hidden">
              <p className="text-sm font-medium text-white truncate">Sudhan Student</p>
              <p className="text-xs text-gray-400 truncate">Admin</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 ml-64 p-8">
        <div className="max-w-7xl mx-auto animation-fade-in">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/add" element={<AddTransaction />} />
            <Route path="/predict" element={<Forecast />} />
            <Route path="/chat" element={<Chatbot />} />
          </Routes>
        </div>
      </main>
    </div>
  );
};

const NavLink = ({ to, icon, label, active }) => (
  <Link
    to={to}
    className={`
      flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-200 group
      ${active
        ? 'bg-primary/10 text-primary border border-primary/20'
        : 'text-gray-400 hover:bg-hover hover:text-white border border-transparent'
      }
    `}
  >
    {React.cloneElement(icon, {
      className: active ? 'text-primary' : 'text-gray-500 group-hover:text-white',
      size: 18
    })}
    <span className="font-medium text-sm">{label}</span>
  </Link>
);

export default App;

import React, { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Activity, Cpu, Wrench, Calendar, Home } from 'lucide-react';
import { DashboardOverview } from '@/components/Dashboard/DashboardOverview';
import { DigitalTwinPage } from '@/components/DigitalTwin';
import { PredictivePage } from '@/components/Predictive';
import { SchedulingPage } from '@/components/Scheduling';
import { wsService } from '@/services/websocket';

function App() {
  useEffect(() => {
    // Connect WebSocket on mount
    wsService.connect();

    return () => {
      wsService.disconnect();
    };
  }, []);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <nav className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex">
                <div className="flex-shrink-0 flex items-center">
                  <Activity className="h-8 w-8 text-primary-600" />
                  <span className="ml-2 text-xl font-bold text-gray-900">
                    Digital Twin Factory
                  </span>
                </div>

                <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                  <NavLink to="/" icon={<Home className="h-5 w-5" />}>
                    Dashboard
                  </NavLink>
                  <NavLink to="/digital-twin" icon={<Cpu className="h-5 w-5" />}>
                    Digital Twin
                  </NavLink>
                  <NavLink to="/predictive" icon={<Wrench className="h-5 w-5" />}>
                    Predictive Maintenance
                  </NavLink>
                  <NavLink to="/scheduling" icon={<Calendar className="h-5 w-5" />}>
                    Scheduling
                  </NavLink>
                </div>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<DashboardOverview />} />
            <Route path="/digital-twin" element={<DigitalTwinPage />} />
            <Route path="/predictive" element={<PredictivePage />} />
            <Route path="/scheduling" element={<SchedulingPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

interface NavLinkProps {
  to: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

function NavLink({ to, icon, children }: NavLinkProps) {
  return (
    <Link
      to={to}
      className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:border-gray-300 hover:text-gray-700 transition-colors"
    >
      <span className="mr-2">{icon}</span>
      {children}
    </Link>
  );
}

export default App;

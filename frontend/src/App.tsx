import { useState } from 'react';
import { Dashboard } from './components/Dashboard';
import { QaoaPsoComparison } from './components/QaoaPsoComparison';

function App() {
  const [selectedView, setSelectedView] = useState<'vedant' | 'aditya' | 'comparison' | null>(null);

  if (selectedView === 'comparison') {
    return <QaoaPsoComparison onBack={() => setSelectedView(null)} />;
  }

  if (selectedView === 'vedant' || selectedView === 'aditya') {
    return (
      <Dashboard
        networkType={selectedView}
        onBack={() => setSelectedView(null)}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-between selection:bg-indigo-500 selection:text-white relative overflow-hidden">
      {/* Decorative background glows */}
      <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full bg-blue-900/10 blur-[150px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] rounded-full bg-purple-900/10 blur-[150px] pointer-events-none" />

      {/* Main Content Container */}
      <div className="max-w-7xl mx-auto px-6 py-12 flex-grow flex flex-col justify-center w-full z-10">
        
        {/* Header Section */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 bg-indigo-950/50 border border-indigo-800/40 rounded-full px-4 py-1.5 text-sm font-semibold text-indigo-300 mb-6 backdrop-blur-md shadow-sm">
            <span className="flex h-2 w-2 rounded-full bg-indigo-400" />
            Research Project Portal
          </div>
          
          <h1 className="text-4xl md:text-6xl font-black tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400 leading-tight">
            PSO Traffic Control Center
          </h1>
          <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Select a traffic network model below to simulate, configure capacities, or compare QAOA Quantum & PSO Swarm optimization algorithms.
          </p>
        </div>

        {/* Network Selection Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto w-full">
          
          {/* Card 1: Vedant's Network */}
          <div className="bg-slate-900/40 border border-slate-800/60 backdrop-blur-xl rounded-3xl p-8 hover:border-blue-500/50 hover:shadow-[0_0_40px_rgba(59,130,246,0.1)] transition-all duration-300 flex flex-col justify-between group">
            <div>
              <div className="h-12 w-12 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mb-6 font-bold text-xl group-hover:scale-110 transition-transform">
                V
              </div>
              <h2 className="text-2xl font-bold mb-3 text-slate-100 group-hover:text-blue-400 transition-colors">
                Vedant's Network
              </h2>
              <p className="text-slate-400 text-sm leading-relaxed mb-6">
                Standard 5-node traffic model with distinct arterial and expressway properties, featuring a reference edge and STQGCN flow predictions.
              </p>
              
              {/* Specs */}
              <div className="space-y-3 mb-8 border-t border-slate-800/80 pt-6">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Topology</span>
                  <span className="text-slate-300 font-semibold">5 Nodes (A, B, C, D, E)</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Total Edges</span>
                  <span className="text-slate-300 font-semibold">14 Directed Edges</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Threshold Formula</span>
                  <span className="text-slate-300 font-semibold text-right">
                    Arterial: 85% | Expressway: 95%
                  </span>
                </div>
              </div>
            </div>

            <button
              onClick={() => setSelectedView('vedant')}
              className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3.5 px-6 rounded-2xl transition-all duration-200 flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-blue-900/20"
            >
              Access Vedant's Network
              <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
              </svg>
            </button>
          </div>

          {/* Card 2: Aditya's Network */}
          <div className="bg-slate-900/40 border border-slate-800/60 backdrop-blur-xl rounded-3xl p-8 hover:border-purple-500/50 hover:shadow-[0_0_40px_rgba(168,85,247,0.1)] transition-all duration-300 flex flex-col justify-between group">
            <div>
              <div className="h-12 w-12 rounded-2xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 mb-6 font-bold text-xl group-hover:scale-110 transition-transform">
                A
              </div>
              <h2 className="text-2xl font-bold mb-3 text-slate-100 group-hover:text-purple-400 transition-colors">
                Aditya's Network
              </h2>
              <p className="text-slate-400 text-sm leading-relaxed mb-6">
                Fully bidirectional 4-node network topology (complete graph K4) with updated edge capacities, initial predicted loads, and dynamic optimization.
              </p>
              
              {/* Specs */}
              <div className="space-y-3 mb-8 border-t border-slate-800/80 pt-6">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Topology</span>
                  <span className="text-slate-300 font-semibold">4 Nodes (A, B, C, D)</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Total Edges</span>
                  <span className="text-slate-300 font-semibold">12 Bidirectional Edges</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Threshold Formula</span>
                  <span className="text-slate-300 font-semibold text-right">
                    Uniform: 60% of capacity
                  </span>
                </div>
              </div>
            </div>

            <button
              onClick={() => setSelectedView('aditya')}
              className="w-full bg-purple-600 hover:bg-purple-500 text-white font-semibold py-3.5 px-6 rounded-2xl transition-all duration-200 flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-purple-900/20"
            >
              Access Aditya's Network
              <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
              </svg>
            </button>
          </div>

          {/* Card 3: QAOA & PSO Comparison */}
          <div className="bg-slate-900/40 border border-slate-800/60 backdrop-blur-xl rounded-3xl p-8 hover:border-emerald-500/50 hover:shadow-[0_0_40px_rgba(16,185,129,0.1)] transition-all duration-300 flex flex-col justify-between group">
            <div>
              <div className="h-12 w-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 mb-6 font-bold text-xl group-hover:scale-110 transition-transform">
                Q
              </div>
              <h2 className="text-2xl font-bold mb-3 text-slate-100 group-hover:text-emerald-400 transition-colors">
                QAOA & PSO Comparison
              </h2>
              <p className="text-slate-400 text-sm leading-relaxed mb-6">
                Parallel side-by-side optimization of QAOA packet rerouting vs. PSO swarm signal timing on Aditya's graph with interactive load controls.
              </p>
              
              {/* Specs */}
              <div className="space-y-3 mb-8 border-t border-slate-800/80 pt-6">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Parallel Engines</span>
                  <span className="text-slate-300 font-semibold">QAOA vs. PSO Swarm</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Live Load Control</span>
                  <span className="text-slate-300 font-semibold">Interactive Sliders</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Key Benchmarks</span>
                  <span className="text-emerald-400 font-semibold text-right">
                    Latency | Peak % | Load Dist %
                  </span>
                </div>
              </div>
            </div>

            <button
              onClick={() => setSelectedView('comparison')}
              className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-3.5 px-6 rounded-2xl transition-all duration-200 flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-emerald-900/20"
            >
              Access QAOA vs PSO Comparison
              <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path>
              </svg>
            </button>
          </div>

        </div>
      </div>

      {/* Footer */}
      <footer className="w-full text-center py-8 border-t border-slate-900 text-slate-600 text-xs z-10">
        &copy; 2026 PSO Traffic Optimization Lab. All rights reserved.
      </footer>
    </div>
  );
}

export default App;

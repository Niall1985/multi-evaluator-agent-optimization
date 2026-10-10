import React from 'react';
import { Layers, Activity, Cpu, Sparkles, RefreshCw, Zap } from 'lucide-react';

export default function Navbar({ telemetry, currentGen, isRunning, onRefresh }) {
  return (
    <header className="border-b border-slate-800/80 bg-slate-900/90 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Layers className="w-6 h-6 text-slate-950 font-bold" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold text-white tracking-tight">Multi-Evaluator Agent Optimizer</h1>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Gaps 1-6 Active
              </span>
            </div>
            <p className="text-xs text-slate-400">Adaptive Evaluation • fANOVA • Dimension Elites • GP-UCB</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="hidden md:flex items-center space-x-3 text-xs">
            <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60">
              <Activity className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-slate-400">Gen:</span>
              <span className="font-semibold text-white">{currentGen}</span>
            </div>
            <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-slate-400">Best Fitness:</span>
              <span className="font-semibold text-amber-300">{telemetry?.best_fitness?.toFixed(3) || '0.000'}</span>
            </div>
          </div>

          <button
            onClick={onRefresh}
            disabled={isRunning}
            className="p-2 text-slate-400 hover:text-white rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 transition"
            title="Refresh Data"
          >
            <RefreshCw className={`w-4 h-4 ${isRunning ? 'animate-spin text-emerald-400' : ''}`} />
          </button>
        </div>
      </div>
    </header>
  );
}

import React from 'react';
import { LayoutDashboard, Scissors, SlidersHorizontal, DollarSign, Cpu, Archive, GitFork, CheckSquare, Database } from 'lucide-react';

export default function NavigationTabs({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard, tag: 'Studio' },
    { id: 'batch_benchmark', label: 'Batch Benchmark (100 Runs)', icon: Database, tag: '300 Agents' },
    { id: 'gap1', label: 'Gap 1: Pruning', icon: Scissors, tag: '2-Tier' },
    { id: 'gap2', label: 'Gap 2: Bayesian w', icon: SlidersHorizontal, tag: 'Simplex' },
    { id: 'gap3', label: 'Gap 3: Cost Fitness', icon: DollarSign, tag: '-λ·c' },
    { id: 'gap4', label: 'Gap 4: fANOVA', icon: Cpu, tag: 'Sobol' },
    { id: 'gap5', label: 'Gap 5: Archive', icon: Archive, tag: 'Elites' },
    { id: 'gap6', label: 'Gap 6: Joint Search', icon: GitFork, tag: 'GP-UCB' },
    { id: 'tests', label: 'Test Suite Execution', icon: CheckSquare, tag: 'In/Out' },
  ];

  return (
    <nav className="flex space-x-1 p-1.5 bg-slate-900/90 border border-slate-800 rounded-xl overflow-x-auto shadow-md backdrop-blur">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all duration-150 ${
              isActive
                ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 shadow-sm shadow-emerald-950'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent'
            }`}
          >
            <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-emerald-400' : 'text-slate-400'}`} />
            <span>{tab.label}</span>
            <span
              className={`text-[9px] uppercase px-1.5 py-0.2 rounded font-mono ${
                isActive
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : 'bg-slate-800 text-slate-400'
              }`}
            >
              {tab.tag}
            </span>
          </button>
        );
      })}
    </nav>
  );
}

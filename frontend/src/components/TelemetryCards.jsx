import React from 'react';
import { DollarSign, Award, TrendingUp, ShieldAlert, Cpu, Database } from 'lucide-react';

export default function TelemetryCards({ telemetry, fanova }) {
  const cards = [
    {
      title: 'Best Population Fitness',
      value: telemetry?.best_fitness?.toFixed(4) || '0.0000',
      subtitle: `Agent ID: ${telemetry?.best_agent_id || 'N/A'}`,
      icon: Award,
      color: 'from-amber-500/20 to-orange-500/5 text-amber-400 border-amber-500/30',
    },
    {
      title: 'Adaptive Cost Spent',
      value: `$${telemetry?.cumulative_cost_spent?.toFixed(5) || '0.00000'}`,
      subtitle: `Naive Full Cost: $${telemetry?.cumulative_naive_cost?.toFixed(5) || '0.00000'}`,
      icon: DollarSign,
      color: 'from-emerald-500/20 to-teal-500/5 text-emerald-400 border-emerald-500/30',
    },
    {
      title: 'Net Cost Savings',
      value: `${telemetry?.cost_saved_pct?.toFixed(1) || '0.0'}%`,
      subtitle: `Saved $${telemetry?.cost_saved_usd?.toFixed(5) || '0.00000'} total`,
      icon: TrendingUp,
      color: 'from-blue-500/20 to-indigo-500/5 text-blue-400 border-blue-500/30',
    },
    {
      title: 'Pareto Optimal Pool',
      value: `${telemetry?.pareto_size || 0} / ${telemetry?.archive_size || 0}`,
      subtitle: 'Non-dominated candidate agents',
      icon: Database,
      color: 'from-purple-500/20 to-pink-500/5 text-purple-400 border-purple-500/30',
    },
    {
      title: 'fANOVA Variance Status',
      value: `${fanova?.active_informative_count ?? 6} Active`,
      subtitle: `${fanova?.redundant_evaluators?.length || 0} redundant pruned`,
      icon: Cpu,
      color: 'from-teal-500/20 to-cyan-500/5 text-teal-400 border-teal-500/30',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className={`p-4 rounded-xl bg-gradient-to-b ${card.color} bg-slate-900/60 border backdrop-blur flex flex-col justify-between`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-slate-400">{card.title}</span>
              <Icon className="w-4 h-4 opacity-80" />
            </div>
            <div>
              <div className="text-xl font-bold text-white tracking-tight">{card.value}</div>
              <div className="text-[11px] text-slate-400 mt-0.5 truncate">{card.subtitle}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

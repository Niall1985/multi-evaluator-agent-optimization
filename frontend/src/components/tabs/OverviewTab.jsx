import React from 'react';
import ParetoScatterPlot from '../ParetoScatterPlot';
import GenerationHistoryTable from '../GenerationHistoryTable';
import { Award, Zap, Code, ShieldCheck, ArrowUpRight } from 'lucide-react';

export default function OverviewTab({ status, telemetry, paretoAgents, historyRecords, onSwitchTab }) {
  const bestAgent = status?.best_agent;

  return (
    <div className="space-y-6">
      {/* Top Highlight Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Best Performing Agent Card */}
        <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur flex flex-col justify-between">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
            <div className="flex items-center space-x-2">
              <Award className="w-5 h-5 text-amber-400" />
              <div>
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <span>Current Best Evolved Agent</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
                    {bestAgent?.id || 'agent_seed_g0'}
                  </span>
                </h3>
                <p className="text-xs text-slate-400">Peak scalar fitness across all generations</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-slate-400">Fitness Score</div>
              <div className="text-xl font-bold font-mono text-amber-300">
                {bestAgent?.fitness?.toFixed(4) || '0.0000'}
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <span className="text-xs font-semibold text-slate-300">Active System Prompt:</span>
              <pre className="mt-1 bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-emerald-300/90 whitespace-pre-wrap font-mono max-h-32 overflow-y-auto">
                {bestAgent?.system_prompt || 'Loading prompt...'}
              </pre>
            </div>

            {bestAgent?.metrics && (
              <div>
                <span className="text-xs font-semibold text-slate-300">Evaluator Dimension Scores:</span>
                <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 mt-1">
                  {Object.entries(bestAgent.metrics).map(([k, v]) => (
                    <div key={k} className="p-2 rounded-lg bg-slate-950 border border-slate-800 text-center">
                      <div className="text-[10px] text-slate-400 truncate">{k.replace('mu_', 'μ')}</div>
                      <div className="text-xs font-bold font-mono text-white">{Number(v).toFixed(3)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Quick Gap Navigation Card */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white mb-2 flex items-center space-x-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Research Gaps Status</span>
            </h3>
            <p className="text-xs text-slate-400 mb-3">Explore individual gap solutions:</p>

            <div className="space-y-1.5 text-xs font-medium">
              {[
                { id: 'gap1', name: 'Gap 1: Selective Cascade Pruning', badge: 'Active' },
                { id: 'gap2', name: 'Gap 2: Bayesian Weight Simplex', badge: 'GP-BO' },
                { id: 'gap3', name: 'Gap 3: Cost-Penalized Fitness', badge: '-λ·c' },
                { id: 'gap4', name: 'Gap 4: Functional ANOVA', badge: 'Sobol S_j' },
                { id: 'gap5', name: 'Gap 5: Per-Evaluator Archive', badge: 'Elites' },
                { id: 'gap6', name: 'Gap 6: Joint Continuous Search', badge: 'GP-UCB' },
              ].map((gap) => (
                <button
                  key={gap.id}
                  onClick={() => onSwitchTab(gap.id)}
                  className="w-full flex items-center justify-between p-2 rounded-lg bg-slate-950/60 hover:bg-slate-800/80 border border-slate-800/80 text-left transition group"
                >
                  <span className="text-slate-300 group-hover:text-emerald-300">{gap.name}</span>
                  <span className="flex items-center space-x-1 text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 group-hover:bg-emerald-500/20 group-hover:text-emerald-300">
                    <span>{gap.badge}</span>
                    <ArrowUpRight className="w-3 h-3" />
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Pareto Scatter Plot */}
      <ParetoScatterPlot paretoAgents={paretoAgents} allRecords={historyRecords} />

      {/* Generation History Table */}
      <GenerationHistoryTable historyRecords={historyRecords} />
    </div>
  );
}

import React from 'react';
import { Scissors, CheckCircle, XCircle, ArrowRight, Zap, Filter } from 'lucide-react';

export default function Gap1PruningTab({ historyRecords }) {
  const coreMetrics = [
    { id: 'mu_1_correctness', label: 'μ1 Functional Correctness', cost: '$0.0010', type: 'Deterministic' },
    { id: 'mu_2_latency', label: 'μ2 Execution Latency', cost: '$0.0001', type: 'Deterministic' },
    { id: 'mu_3_conciseness', label: 'μ3 Token Conciseness', cost: '$0.0001', type: 'Deterministic' },
  ];

  const deepMetrics = [
    { id: 'mu_4_reasoning', label: 'μ4 LLM Reasoning Quality', cost: '$0.0050', type: 'LLM Judge' },
    { id: 'mu_5_edge_cases', label: 'μ5 Edge-Case Stress Test', cost: '$0.0050', type: 'Sandbox Exec' },
    { id: 'mu_6_safety', label: 'μ6 Safety & Non-Hallucination', cost: '$0.0050', type: 'LLM Judge' },
  ];

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-lg backdrop-blur">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Scissors className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <span>Gap 1: Selective Evaluator Pruning (2-Tier Cascade)</span>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                x_E ∈ &#123;0, 1&#125;⁶
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Solves uniform evaluation cost by gating expensive LLM judges behind fast deterministic core tests.
            </p>
          </div>
        </div>

        {/* 2-Tier Cascade Visual Flow */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          {/* Stage 1: Core Tier */}
          <div className="p-4 rounded-xl bg-slate-950/70 border border-emerald-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">Tier 1: Core Metrics</span>
              <span className="text-[11px] font-mono text-slate-400">$0.0012 total</span>
            </div>
            <p className="text-xs text-slate-400 mb-3">Always executed first on every candidate agent:</p>
            <div className="space-y-1.5 text-xs">
              {coreMetrics.map((m) => (
                <div key={m.id} className="flex justify-between items-center p-1.5 rounded bg-slate-900 border border-slate-800">
                  <span className="text-slate-300">{m.label}</span>
                  <span className="text-emerald-400 font-mono text-[11px]">{m.cost}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Decision Node */}
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-700/60 flex flex-col justify-between">
            <div className="flex items-center space-x-2 mb-2">
              <Filter className="w-4 h-4 text-amber-400" />
              <span className="text-xs font-bold text-amber-300 uppercase tracking-wider">Cascade Gate Trigger</span>
            </div>
            <div className="space-y-2 text-xs text-slate-300">
              <div className="p-2 rounded bg-slate-900 border border-slate-800 font-mono text-[11px]">
                Preview Score = avg(μ₁, μ₂, μ₃)
              </div>
              <div className="text-[11px] text-slate-400">
                • If Preview &ge; <span className="text-emerald-400 font-bold">0.65</span>: Candidate is promising &rarr; Trigger Deep Tier.
              </div>
              <div className="text-[11px] text-slate-400">
                • If Preview &lt; <span className="text-rose-400 font-bold">0.65</span>: Prune Deep Tier (save $0.015).
              </div>
              <div className="text-[11px] text-purple-300">
                • Soft-probe audit rate: <span className="font-bold">ε = 30%</span> (Gap 5).
              </div>
            </div>
          </div>

          {/* Stage 2: Deep Tier */}
          <div className="p-4 rounded-xl bg-slate-950/70 border border-indigo-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-indigo-400 uppercase tracking-wider">Tier 2: Deep Tier LLMs</span>
              <span className="text-[11px] font-mono text-slate-400">$0.0150 total</span>
            </div>
            <p className="text-xs text-slate-400 mb-3">Selectively triggered only for surviving candidates:</p>
            <div className="space-y-1.5 text-xs">
              {deepMetrics.map((m) => (
                <div key={m.id} className="flex justify-between items-center p-1.5 rounded bg-slate-900 border border-slate-800">
                  <span className="text-slate-300">{m.label}</span>
                  <span className="text-indigo-400 font-mono text-[11px]">{m.cost}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Per-Generation Pruning Log Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
        <h3 className="text-sm font-bold text-white mb-3">Evaluator Activation Vector x_E Across Generations</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/60 text-slate-400 border-b border-slate-800 uppercase text-[10px]">
              <tr>
                <th className="py-2.5 px-3">Gen</th>
                <th className="py-2.5 px-3">Agent ID</th>
                <th className="py-2.5 px-3">Strategy</th>
                <th className="py-2.5 px-3 text-center">μ1 Correct</th>
                <th className="py-2.5 px-3 text-center">μ2 Latency</th>
                <th className="py-2.5 px-3 text-center">μ3 Tokens</th>
                <th className="py-2.5 px-3 text-center">μ4 Reasoning</th>
                <th className="py-2.5 px-3 text-center">μ5 Edge</th>
                <th className="py-2.5 px-3 text-center">μ6 Safety</th>
                <th className="py-2.5 px-3 text-right">Cost Saved</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
              {historyRecords.slice().reverse().map((r) => {
                const active = new Set(r.active_evaluators || []);
                const saved = (r.naive_cost || 0) - (r.cost_spent || 0);
                return (
                  <tr key={r.child_id} className="hover:bg-slate-800/40">
                    <td className="py-2 px-3 text-slate-300 font-semibold">{r.generation}</td>
                    <td className="py-2 px-3 text-emerald-400">{r.child_id}</td>
                    <td className="py-2 px-3 text-slate-400 text-[10px]">{r.strategy}</td>
                    {['mu_1_correctness', 'mu_2_latency', 'mu_3_conciseness', 'mu_4_reasoning', 'mu_5_edge_cases', 'mu_6_safety'].map((m) => (
                      <td key={m} className="py-2 px-3 text-center">
                        {active.has(m) ? (
                          <span className="inline-block px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-[10px] font-bold">
                            RUN
                          </span>
                        ) : (
                          <span className="inline-block px-1.5 py-0.5 rounded bg-rose-500/10 text-rose-400 text-[10px]">
                            SKIP
                          </span>
                        )}
                      </td>
                    ))}
                    <td className="py-2 px-3 text-right font-semibold text-emerald-400">
                      ${saved > 0 ? saved.toFixed(5) : '0.00000'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

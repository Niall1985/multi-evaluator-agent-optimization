import React, { useState } from 'react';
import { History, ChevronDown, ChevronRight, Check, AlertCircle, Code, Shield } from 'lucide-react';

export default function GenerationHistoryTable({ historyRecords }) {
  const [expandedId, setExpandedId] = useState(null);

  if (!historyRecords || historyRecords.length === 0) {
    return null;
  }

  // Reverse to show latest generation first
  const records = [...historyRecords].reverse();

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-4">
        <div className="flex items-center space-x-2">
          <History className="w-5 h-5 text-blue-400" />
          <div>
            <h3 className="text-sm font-bold text-white">Generation History & Prompt Lineage</h3>
            <p className="text-xs text-slate-400">Detailed inspection of mutated prompts, evaluated scores, and selective cascades.</p>
          </div>
        </div>
        <span className="text-xs font-mono text-slate-400">{historyRecords.length} records logged</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/60 text-slate-400 border-b border-slate-800 uppercase text-[10px] tracking-wider">
            <tr>
              <th className="py-2.5 px-3">Gen</th>
              <th className="py-2.5 px-3">Agent ID</th>
              <th className="py-2.5 px-3">Strategy</th>
              <th className="py-2.5 px-3">Fitness</th>
              <th className="py-2.5 px-3">Δ Fitness</th>
              <th className="py-2.5 px-3">Cost ($)</th>
              <th className="py-2.5 px-3">Active Evaluators</th>
              <th className="py-2.5 px-3">Mutation</th>
              <th className="py-2.5 px-3 text-right">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {records.map((r) => {
              const isExpanded = expandedId === r.child_id;
              const hasReg = r.silent_regression?.has_regression;
              const isSoftProbe = r.soft_probe_triggered;

              return (
                <React.Fragment key={r.child_id}>
                  <tr
                    onClick={() => setExpandedId(isExpanded ? null : r.child_id)}
                    className={`hover:bg-slate-800/40 cursor-pointer transition ${
                      isExpanded ? 'bg-slate-800/30' : ''
                    }`}
                  >
                    <td className="py-2.5 px-3 text-slate-300 font-semibold">{r.generation}</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-semibold">{r.child_id}</td>
                    <td className="py-2.5 px-3 text-slate-300">
                      <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px]">
                        {r.strategy}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-white font-bold">{r.fitness.toFixed(4)}</td>
                    <td className={`py-2.5 px-3 font-semibold ${r.delta_f >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {r.delta_f >= 0 ? `+${r.delta_f.toFixed(4)}` : r.delta_f.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">${r.cost_spent.toFixed(5)}</td>
                    <td className="py-2.5 px-3">
                      <span className="flex items-center space-x-1">
                        <span className="text-slate-300">{r.active_evaluators?.length || 0} / 6</span>
                        {isSoftProbe && (
                          <span className="px-1.5 py-0.2 rounded bg-purple-500/20 text-purple-300 text-[9px] border border-purple-500/30">
                            Probe
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 truncate max-w-[140px]">{r.mutation_type}</td>
                    <td className="py-2.5 px-3 text-right">
                      {isExpanded ? (
                        <ChevronDown className="w-4 h-4 inline text-slate-400" />
                      ) : (
                        <ChevronRight className="w-4 h-4 inline text-slate-400" />
                      )}
                    </td>
                  </tr>

                  {/* Expanded Detail View */}
                  {isExpanded && (
                    <tr className="bg-slate-950/80 border-b border-slate-800">
                      <td colSpan={9} className="p-4 space-y-3 font-sans">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {/* System Prompt */}
                          <div>
                            <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-300 mb-1">
                              <Code className="w-3.5 h-3.5 text-emerald-400" />
                              <span>Evolved System Prompt:</span>
                            </div>
                            <pre className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-[11px] text-emerald-300/90 whitespace-pre-wrap max-h-48 overflow-y-auto font-mono">
                              {r.child_prompt}
                            </pre>
                          </div>

                          {/* Evaluator Scores and Solution */}
                          <div className="space-y-3">
                            <div>
                              <div className="text-xs font-semibold text-slate-300 mb-1">Evaluator Metrics (μ):</div>
                              <div className="grid grid-cols-3 gap-2">
                                {Object.entries(r.metrics || {}).map(([mName, mScore]) => (
                                  <div key={mName} className="p-2 rounded bg-slate-900 border border-slate-800 text-[11px]">
                                    <div className="text-slate-400 truncate">{mName.replace('mu_', 'μ')}</div>
                                    <div className="text-sm font-bold font-mono text-white mt-0.5">{mScore.toFixed(3)}</div>
                                  </div>
                                ))}
                              </div>
                            </div>

                            {/* Task Solution Preview */}
                            {r.solution && (
                              <div>
                                <div className="text-xs font-semibold text-slate-300 mb-1">Generated Solution Preview:</div>
                                <pre className="bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-[11px] text-slate-300 whitespace-pre-wrap max-h-24 overflow-y-auto font-mono">
                                  {r.solution}
                                </pre>
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

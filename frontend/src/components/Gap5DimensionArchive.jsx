import React from 'react';
import { Archive, ShieldAlert, CheckCircle2, Award, Sparkles, ExternalLink } from 'lucide-react';

export default function Gap5DimensionArchive({ dimensionElitesData }) {
  const elites = dimensionElitesData?.dimension_elites || {};
  const regressionInfo = dimensionElitesData?.silent_regression || { has_regression: false };

  const metricLabels = {
    mu_1_correctness: 'μ1 Functional Correctness',
    mu_2_latency: 'μ2 Execution Latency',
    mu_3_conciseness: 'μ3 Token Conciseness',
    mu_4_reasoning: 'μ4 LLM Reasoning Quality',
    mu_5_edge_cases: 'μ5 Edge Case Robustness',
    mu_6_safety: 'μ6 Safety & Non-Hallucination',
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-4">
        <div className="flex items-center space-x-2">
          <Archive className="w-5 h-5 text-purple-400" />
          <div>
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <span>Gap 5: Per-Evaluator Dimension Elite Archive</span>
              <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                MAP-Elites View
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Preserves peak-performing agent per dimension to safeguard against silent regression when metrics are pruned.
            </p>
          </div>
        </div>

        {/* Silent Regression Status Badge */}
        {regressionInfo.has_regression ? (
          <div className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-rose-500/20 border border-rose-500/40 text-rose-300 text-xs font-semibold animate-pulse">
            <ShieldAlert className="w-4 h-4 text-rose-400" />
            <span>Silent Regression Alert ({regressionInfo.regression_count})</span>
          </div>
        ) : (
          <div className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Zero Silent Regression</span>
          </div>
        )}
      </div>

      {/* Grid of Dimension Elite Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {Object.entries(metricLabels).map(([metricKey, label]) => {
          const elite = elites[metricKey];
          const isRegressed = regressionInfo.regressions?.some((r) => r.metric === metricKey);

          return (
            <div
              key={metricKey}
              className={`p-3.5 rounded-xl border transition ${
                isRegressed
                  ? 'bg-rose-950/20 border-rose-500/40'
                  : elite
                  ? 'bg-slate-950/60 border-slate-800 hover:border-purple-500/40'
                  : 'bg-slate-950/30 border-slate-900 opacity-60'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-slate-300 truncate">{label}</span>
                {elite ? (
                  <span className="text-xs font-mono font-bold text-purple-300 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">
                    {elite.score.toFixed(3)}
                  </span>
                ) : (
                  <span className="text-[11px] text-slate-500">Unreached</span>
                )}
              </div>

              {elite ? (
                <div className="space-y-1 text-[11px]">
                  <div className="flex justify-between text-slate-400">
                    <span>Elite Agent:</span>
                    <span className="font-mono text-slate-300">{elite.agent_id}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Gen Found:</span>
                    <span className="text-slate-300">Gen {elite.generation}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Mutation:</span>
                    <span className="text-slate-300 truncate max-w-[120px]">{elite.mutation}</span>
                  </div>
                  <div className="mt-2 pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 italic line-clamp-2">
                    "{elite.prompt}"
                  </div>
                </div>
              ) : (
                <div className="text-[11px] text-slate-500 py-3 text-center">No elite candidate logged yet</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

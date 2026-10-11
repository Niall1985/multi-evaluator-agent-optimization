import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { Cpu, AlertTriangle, CheckCircle, Info } from 'lucide-react';

export default function Gap4FanovaPanel({ fanova }) {
  if (!fanova || !fanova.evaluator_importances) {
    return (
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
        <h3 className="text-sm font-semibold text-white mb-2">Gap 4: Functional ANOVA Variance Decomposition</h3>
        <p className="text-xs text-slate-400">Waiting for generation data to compute Sobol sensitivity indices...</p>
      </div>
    );
  }

  const threshold = fanova.threshold_used || 0.05;
  const rawData = Object.entries(fanova.evaluator_importances).map(([metric, score]) => ({
    metric: metric.replace('mu_', 'μ').replace('_', ' '),
    rawMetric: metric,
    importance: score,
    share: (fanova.normalized_shares?.[metric] || 0) * 100,
    isRedundant: score < threshold,
  }));

  // Sort descending by importance
  const chartData = [...rawData].sort((a, b) => b.importance - a.importance);

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-4">
          <div className="flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-teal-400" />
            <div>
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <span>Gap 4: Functional ANOVA (fANOVA) Redundancy Detection</span>
                <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">
                  Sobol Indices S_j
                </span>
              </h3>
              <p className="text-xs text-slate-400">
                Decomposes surrogate model variance: V(f) = Σ V_j + Σ V_jk. Prunes low-importance evaluators (S_j &lt; {threshold}).
              </p>
            </div>
          </div>
          <div className="text-right">
            <span className="text-xs text-slate-400">Total Variance V(f):</span>
            <span className="ml-1 text-xs font-mono font-semibold text-teal-300">
              {fanova.total_variance?.toFixed(5) || '0.00000'}
            </span>
          </div>
        </div>

        {/* Bar Chart of Evaluator Importances */}
        <div className="h-56 w-full mb-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <XAxis
                dataKey="metric"
                stroke="#64748b"
                fontSize={10}
                tickLine={false}
                angle={-20}
                textAnchor="end"
              />
              <YAxis stroke="#64748b" fontSize={10} domain={[0, 'auto']} tickLine={false} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-slate-950 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs">
                        <div className="font-bold text-white mb-1">{data.rawMetric}</div>
                        <div className="text-teal-300">Sobol Index S_j: {data.importance.toFixed(4)}</div>
                        <div className="text-slate-400">Normalized Variance Share: {data.share.toFixed(1)}%</div>
                        <div className={`mt-1 font-semibold ${data.isRedundant ? 'text-rose-400' : 'text-emerald-400'}`}>
                          Status: {data.isRedundant ? 'Redundant / Pruned' : 'Active & Informative'}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <ReferenceLine y={threshold} stroke="#f43f5e" strokeDasharray="3 3" label={{ value: `Prune Threshold (${threshold})`, fill: '#f43f5e', fontSize: 10, position: 'top' }} />
              <Bar dataKey="importance" radius={[4, 4, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.isRedundant ? '#e11d48' : '#0d9488'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Redundancy Status Summary */}
      <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-xs gap-2">
        <div className="flex items-center space-x-2">
          <span className="text-slate-400">Redundant Evaluators Detected:</span>
          {fanova.redundant_evaluators && fanova.redundant_evaluators.length > 0 ? (
            fanova.redundant_evaluators.map((name) => (
              <span key={name} className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 font-mono text-[11px]">
                {name}
              </span>
            ))
          ) : (
            <span className="text-emerald-400 flex items-center space-x-1">
              <CheckCircle className="w-3.5 h-3.5" />
              <span>None (All evaluators providing signal)</span>
            </span>
          )}
        </div>
        <div className="text-slate-400 text-[11px]">
          Sample points: <span className="font-mono text-slate-200">{fanova.sample_count || 0}</span>
        </div>
      </div>
    </div>
  );
}

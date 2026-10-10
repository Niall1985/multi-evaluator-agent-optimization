import React from 'react';
import { DollarSign, TrendingDown, Percent, Coins, ShieldCheck } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function Gap3CostTab({ telemetry, historyRecords, lambdaPenalty = 0.5 }) {
  // Format cumulative cost data
  let runningAdaptive = 0;
  let runningNaive = 0;
  const costData = historyRecords.map((r) => {
    runningAdaptive += r.cost_spent || 0;
    runningNaive += r.naive_cost || 0.0162;
    return {
      gen: `G${r.generation}`,
      Adaptive: runningAdaptive,
      Naive: runningNaive,
      saved: runningNaive - runningAdaptive,
    };
  });

  const evaluatorPricing = [
    { name: 'μ1 Functional Correctness', cost: '$0.0010', type: 'AST / Unit Test Sandbox' },
    { name: 'μ2 Execution Latency', cost: '$0.0001', type: 'High-Precision Timer' },
    { name: 'μ3 Token Conciseness', cost: '$0.0001', type: 'Byte/Token Counter' },
    { name: 'μ4 LLM Reasoning Quality', cost: '$0.0050', type: 'Groq GPT-OSS / Judge' },
    { name: 'μ5 Edge Case Robustness', cost: '$0.0050', type: 'Isolated Docker Exec' },
    { name: 'μ6 Safety & Non-Hallucination', cost: '$0.0050', type: 'Groq GPT-OSS / Judge' },
  ];

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-lg backdrop-blur">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <DollarSign className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <span>Gap 3: Cost-Penalized Multi-Objective Scalarization</span>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                F = Σ w_j μ_j - λ Σ c_j
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Embeds evaluator dollar expenditure directly into the fitness function, penalizing wasteful LLM judge calls.
            </p>
          </div>
        </div>

        {/* Mathematical Equation & Metrics Box */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4 pt-4 border-t border-slate-800 text-xs">
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-slate-400">Penalty Factor (λ):</span>
            <div className="text-lg font-bold font-mono text-emerald-400 mt-0.5">{lambdaPenalty.toFixed(2)}</div>
            <span className="text-[11px] text-slate-500">Balances quality gain vs monetary budget</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-slate-400">Total Net Cost Saved:</span>
            <div className="text-lg font-bold font-mono text-white mt-0.5">
              ${telemetry?.cost_saved_usd?.toFixed(5) || '0.00000'}
            </div>
            <span className="text-[11px] text-emerald-400 font-semibold">{telemetry?.cost_saved_pct?.toFixed(1) || '0.0'}% saved vs naive</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-slate-400">Full Naive Cost (Without Pruning):</span>
            <div className="text-lg font-bold font-mono text-slate-400 mt-0.5">
              ${telemetry?.cumulative_naive_cost?.toFixed(5) || '0.00000'}
            </div>
            <span className="text-[11px] text-slate-500">Every evaluator run every step</span>
          </div>
        </div>
      </div>

      {/* Cumulative Cost Trajectory Chart */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
        <h3 className="text-sm font-bold text-white mb-3">Cumulative Cost Expenditure: Adaptive ($) vs. Naive Baseline ($)</h3>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={costData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <XAxis dataKey="gen" stroke="#64748b" fontSize={11} tickLine={false} />
              <YAxis stroke="#64748b" fontSize={11} tickLine={false} tickFormatter={(v) => `$${v.toFixed(3)}`} />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-slate-950 border border-slate-700 p-3 rounded-lg shadow-xl text-xs space-y-1">
                        <div className="font-bold text-white">{label} Cumulative Cost</div>
                        <div className="text-emerald-400">Adaptive Spent: ${payload[0]?.value?.toFixed(5)}</div>
                        <div className="text-rose-400">Naive Cost: ${payload[1]?.value?.toFixed(5)}</div>
                        <div className="text-slate-300 font-semibold">
                          Net Saved: ${((payload[1]?.value || 0) - (payload[0]?.value || 0)).toFixed(5)}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Line type="monotone" dataKey="Adaptive" stroke="#10b981" strokeWidth={2.5} dot={{ r: 3 }} name="Adaptive System ($)" />
              <Line type="monotone" dataKey="Naive" stroke="#f43f5e" strokeWidth={1.5} strokeDasharray="3 3" dot={false} name="Naive Full Cost ($)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Evaluator Costing Matrix */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
        <h3 className="text-sm font-bold text-white mb-3">Evaluator Unit Cost Matrix (c_j)</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
          {evaluatorPricing.map((item, idx) => (
            <div key={idx} className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex justify-between items-center">
              <div>
                <div className="font-semibold text-slate-200">{item.name}</div>
                <div className="text-[10px] text-slate-500">{item.type}</div>
              </div>
              <div className="font-mono font-bold text-emerald-400">{item.cost}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

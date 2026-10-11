import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { GitFork, Activity } from 'lucide-react';

export default function Gap6JointSearchTracker({ historyRecords, acquisitionType }) {
  if (!historyRecords || historyRecords.length === 0) {
    return null;
  }

  // Build weights evolution data
  const weightsData = historyRecords.map((r) => {
    const w = r.weights || {};
    return {
      gen: `G${r.generation}`,
      μ1: w.mu_1_correctness || 0,
      μ2: w.mu_2_latency || 0,
      μ3: w.mu_3_conciseness || 0,
      μ4: w.mu_4_reasoning || 0,
      μ5: w.mu_5_edge_cases || 0,
      μ6: w.mu_6_safety || 0,
    };
  });

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-4">
        <div className="flex items-center space-x-2">
          <GitFork className="w-5 h-5 text-indigo-400" />
          <div>
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <span>Gap 6: Continuous Simplex Weight Trajectory (GP-UCB / EI)</span>
              <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                {acquisitionType === 'gp_ucb' ? 'GP-UCB Regret Bound' : 'GP Expected Improvement'}
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Joint optimization over probability simplex Δ⁵ (Σ w_j = 1). Evaluator weights adapt dynamically per generation.
            </p>
          </div>
        </div>
      </div>

      <div className="h-60 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={weightsData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <XAxis dataKey="gen" stroke="#64748b" fontSize={11} tickLine={false} />
            <YAxis stroke="#64748b" fontSize={11} domain={[0, 1]} tickLine={false} />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-slate-950 border border-slate-700 p-3 rounded-lg shadow-xl text-xs space-y-1">
                      <div className="font-bold text-white mb-1.5">{label} - Weight Vector w</div>
                      {payload.map((entry) => (
                        <div key={entry.dataKey} className="flex justify-between space-x-4">
                          <span style={{ color: entry.color }}>{entry.dataKey}:</span>
                          <span className="font-mono text-white font-semibold">{(entry.value * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
            <Area type="monotone" dataKey="μ1" stackId="1" stroke="#10b981" fill="#10b981" fillOpacity={0.8} />
            <Area type="monotone" dataKey="μ2" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.8} />
            <Area type="monotone" dataKey="μ3" stackId="1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.8} />
            <Area type="monotone" dataKey="μ4" stackId="1" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.8} />
            <Area type="monotone" dataKey="μ5" stackId="1" stroke="#ec4899" fill="#ec4899" fillOpacity={0.8} />
            <Area type="monotone" dataKey="μ6" stackId="1" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.8} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

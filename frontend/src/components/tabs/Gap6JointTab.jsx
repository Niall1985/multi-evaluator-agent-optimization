import React from 'react';
import Gap6JointSearchTracker from '../Gap6JointSearchTracker';
import { GitFork, Activity, Shuffle, Cpu } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function Gap6JointTab({ historyRecords, acquisitionType }) {
  // Aggregate mutation arm counts
  const armCounts = {};
  historyRecords.forEach((r) => {
    const mut = r.mutation_type || 'seed';
    armCounts[mut] = (armCounts[mut] || 0) + 1;
  });

  const banditData = Object.entries(armCounts).map(([arm, count]) => ({
    arm: arm.replace('_', ' '),
    rawArm: arm,
    count: count,
  }));

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-lg backdrop-blur">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
            <GitFork className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <span>Gap 6: Grounded Joint Discrete + Continuous Search Space</span>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                S = X_prompt × &#123;0,1&#125;⁶ × Δ⁵
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Couples discrete UCB1 bandits for prompt mutations with continuous GP-UCB for weight simplex tuning with proven sublinear regret bounds.
            </p>
          </div>
        </div>

        {/* Theoretical Framework */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-800 text-xs">
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-indigo-400 font-semibold">Discrete Search Space (UCB1 Multi-Armed Bandit):</span>
            <p className="mt-1 text-[11px] text-slate-300 font-mono">
              Arm Selection: argmax_k [ X̄_k + c √(ln N / N_k) ]
            </p>
            <p className="mt-1 text-[11px] text-slate-400">
              Regret bound: O(√(K T ln T)). Explores Chain-of-Thought, Defensive retry, Boundary stress, and Algorithmic rewrites.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-indigo-400 font-semibold">Continuous Search Space (GP-UCB Acquisition):</span>
            <p className="mt-1 text-[11px] text-slate-300 font-mono">
              Acquisition: α(w) = μ(w) + β_t · σ(w)
            </p>
            <p className="mt-1 text-[11px] text-slate-400">
              Sublinear cumulative regret bound: O(√(T γ_T)). Dynamically balances exploitation of promising weights with simplex exploration.
            </p>
          </div>
        </div>
      </div>

      {/* Discrete Mutation Bandit Arm Usage Chart */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
        <div className="flex items-center space-x-2 pb-3 border-b border-slate-800/80 mb-4">
          <Shuffle className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-bold text-white">Discrete Mutation Operator Bandit Distribution (UCB1)</h3>
        </div>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={banditData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <XAxis dataKey="arm" stroke="#64748b" fontSize={10} tickLine={false} angle={-20} textAnchor="end" />
              <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-slate-950 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs">
                        <div className="font-bold text-white mb-1">{d.rawArm}</div>
                        <div className="text-indigo-400">Invocations: {d.count}</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Continuous Weights Area Chart */}
      <Gap6JointSearchTracker historyRecords={historyRecords} acquisitionType={acquisitionType} />
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import { SlidersHorizontal, Activity, Layers, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, ComposedChart, ReferenceLine } from 'recharts';

export default function Gap2BayesianTab({ historyRecords }) {
  const [targetIdx, setTargetIdx] = useState(0);
  const [gpSlice, setGpSlice] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const metricNames = [
    'μ1 Functional Correctness',
    'μ2 Execution Latency',
    'μ3 Token Conciseness',
    'μ4 LLM Reasoning Quality',
    'μ5 Edge Case Robustness',
    'μ6 Safety & Non-Hallucination',
  ];

  const fetchSlice = async (idx) => {
    setIsLoading(true);
    try {
      const res = await fetch(`/api/gp_slice?target_idx=${idx}`);
      const data = await res.json();
      setGpSlice(data);
    } catch (err) {
      console.warn('Failed to load GP slice:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSlice(targetIdx);
  }, [targetIdx]);

  // Latest generation weights
  const latestRecord = historyRecords.length > 0 ? historyRecords[historyRecords.length - 1] : null;
  const currentWeights = latestRecord?.weights || {};

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-lg backdrop-blur">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
            <SlidersHorizontal className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <span>Gap 2: Continuous Bayesian Weight Optimization</span>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                w ∈ Δ⁵ (Σ w_j = 1)
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Replaces static hand-weights with a Gaussian Process (Matérn 5/2) surrogate modeling fitness gains ΔF.
            </p>
          </div>
        </div>

        {/* Current Active Weights Bar */}
        <div className="mt-4 pt-4 border-t border-slate-800">
          <div className="flex justify-between items-center mb-2 text-xs">
            <span className="text-slate-300 font-semibold">Current Probability Simplex Distribution:</span>
            <span className="font-mono text-indigo-300">Sum = 100.0%</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 text-xs font-mono">
            {Object.entries(currentWeights).map(([mName, val]) => (
              <div key={mName} className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-center">
                <div className="text-[10px] text-slate-400 truncate">{mName.replace('mu_', 'μ')}</div>
                <div className="text-sm font-bold text-indigo-400 mt-0.5">{(Number(val) * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 1D GP Slice Interactive Posterior Plot */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
        <div className="flex flex-wrap items-center justify-between pb-3 border-b border-slate-800/80 mb-4 gap-2">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span>Gaussian Process Posterior Slice & Expected Improvement</span>
            </h3>
            <p className="text-xs text-slate-400">
              Visualizes GP posterior mean μ(w), ±1.96σ confidence bounds, and acquisition curve across candidate weight values.
            </p>
          </div>

          {/* Metric Selector Dropdown */}
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400">Target Weight:</span>
            <select
              value={targetIdx}
              onChange={(e) => setTargetIdx(Number(e.target.value))}
              className="bg-slate-950 border border-slate-700 text-white rounded-lg px-2.5 py-1 text-xs focus:ring-1 focus:ring-indigo-500"
            >
              {metricNames.map((name, idx) => (
                <option key={idx} value={idx}>
                  {name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* GP Slice Chart */}
        <div className="h-72 w-full">
          {gpSlice && gpSlice.curve ? (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={gpSlice.curve} margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
                <XAxis
                  dataKey="w"
                  stroke="#64748b"
                  fontSize={10}
                  tickLine={false}
                  label={{ value: `Weight w_${targetIdx + 1} on Simplex`, position: 'bottom', fill: '#64748b', fontSize: 10 }}
                />
                <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-slate-950 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs space-y-1">
                          <div className="font-bold text-white">Weight: {(d.w * 100).toFixed(1)}%</div>
                          <div className="text-indigo-400">GP Mean μ(w): {d.mu.toFixed(4)}</div>
                          <div className="text-slate-400">Uncertainty Bounds: [{d.lower.toFixed(3)}, {d.upper.toFixed(3)}]</div>
                          <div className="text-amber-400">Expected Improvement: {d.ei.toFixed(5)}</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                {/* Confidence Band */}
                <Area type="monotone" dataKey="upper" stroke="none" fill="#6366f1" fillOpacity={0.15} />
                <Area type="monotone" dataKey="lower" stroke="none" fill="#0b1120" fillOpacity={1} />
                {/* GP Posterior Mean */}
                <Line type="monotone" dataKey="mu" stroke="#6366f1" strokeWidth={2} dot={false} name="GP Mean μ" />
                {/* Acquisition Curve */}
                <Line type="monotone" dataKey="ei" stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="3 3" dot={false} name="Expected Improvement (EI)" />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-xs text-slate-500">
              Gathering generation observations to fit Gaussian Process regressor...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

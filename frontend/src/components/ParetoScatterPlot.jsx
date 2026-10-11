import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, ZAxis } from 'recharts';
import { TrendingUp } from 'lucide-react';

export default function ParetoScatterPlot({ paretoAgents, allRecords }) {
  if (!allRecords || allRecords.length === 0) {
    return null;
  }

  // Format all points
  const paretoIds = new Set(paretoAgents?.map((p) => p.id) || []);
  const nonParetoPoints = [];
  const paretoPoints = [];

  allRecords.forEach((r) => {
    const pt = {
      x: r.cost_spent,
      y: r.fitness,
      id: r.child_id,
      gen: r.generation,
      mutation: r.mutation_type,
    };
    if (paretoIds.has(r.child_id)) {
      paretoPoints.push(pt);
    } else {
      nonParetoPoints.push(pt);
    }
  });

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-4">
        <div className="flex items-center space-x-2">
          <TrendingUp className="w-5 h-5 text-emerald-400" />
          <div>
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <span>Pareto Frontier: Fitness vs. Evaluator Cost ($)</span>
              <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Non-Dominated Optimal
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Evaluates trade-off between scalar fitness and monetary API budget spent per candidate.
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-4 text-xs">
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
            <span className="text-slate-300">Pareto Elite ({paretoPoints.length})</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-600"></span>
            <span className="text-slate-400">Dominated</span>
          </div>
        </div>
      </div>

      <div className="h-60 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
            <XAxis
              type="number"
              dataKey="x"
              name="Cost ($)"
              unit="$"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              tickFormatter={(v) => `$${v.toFixed(4)}`}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Fitness"
              stroke="#64748b"
              fontSize={11}
              domain={['auto', 'auto']}
              tickLine={false}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <ZAxis range={[50, 150]} />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload;
                  return (
                    <div className="bg-slate-950 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs space-y-1">
                      <div className="font-bold text-white">{d.id} (Gen {d.gen})</div>
                      <div className="text-emerald-400">Fitness: {d.y.toFixed(4)}</div>
                      <div className="text-slate-300">Cost: ${d.x.toFixed(5)}</div>
                      <div className="text-slate-400">Mutation: {d.mutation}</div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Scatter name="Dominated" data={nonParetoPoints} fill="#475569" opacity={0.6} />
            <Scatter name="Pareto Elite" data={paretoPoints} fill="#10b981" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

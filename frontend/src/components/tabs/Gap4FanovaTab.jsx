import React from 'react';
import Gap4FanovaPanel from '../Gap4FanovaPanel';
import { Cpu, Info, BarChart2 } from 'lucide-react';

export default function Gap4FanovaTab({ fanova }) {
  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-lg backdrop-blur">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-2.5 rounded-lg bg-teal-500/10 border border-teal-500/20 text-teal-400">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <span>Gap 4: Functional ANOVA (fANOVA) Redundancy Detection</span>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-teal-500/10 text-teal-300 border border-teal-500/20">
                Sobol Indices S_j = V_j / V_total
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Measures each evaluator's true marginal contribution to surrogate model variance, pruning uninformative metrics automatically.
            </p>
          </div>
        </div>

        {/* Theoretical Description */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-800 text-xs">
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-teal-400 font-semibold">Variance Decomposition Formula:</span>
            <pre className="mt-1 text-[11px] font-mono text-slate-300">
              V(f) = Σ V_j + Σ V_jk + ...
            </pre>
            <p className="mt-1 text-[11px] text-slate-400">
              Where V_j = Var_μj( E[f(μ) | μ_j] ) is the main effect variance explained by evaluator j.
            </p>
          </div>
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-teal-400 font-semibold">Redundancy Decision Criterion:</span>
            <pre className="mt-1 text-[11px] font-mono text-slate-300">
              If S_j &lt; 0.05 &rarr; Prune / Downweight
            </pre>
            <p className="mt-1 text-[11px] text-slate-400">
              Evaluators explaining less than 5% of total variance are automatically identified as redundant noise.
            </p>
          </div>
        </div>
      </div>

      {/* Main Interactive Panel */}
      <Gap4FanovaPanel fanova={fanova} />
    </div>
  );
}

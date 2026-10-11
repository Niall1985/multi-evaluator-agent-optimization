import React from 'react';
import Gap5DimensionArchive from '../Gap5DimensionArchive';
import { Archive, ShieldAlert, Sparkles, RefreshCw } from 'lucide-react';

export default function Gap5ArchiveTab({ dimensionElitesData, historyRecords }) {
  const regressionInfo = dimensionElitesData?.silent_regression;
  const probeRecords = historyRecords.filter((r) => r.soft_probe_triggered);

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-lg backdrop-blur">
        <div className="flex items-center space-x-3 mb-3">
          <div className="p-2.5 rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-400">
            <Archive className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <span>Gap 5: Per-Evaluator Dimension Archive & Silent Regression Protection</span>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-purple-500/10 text-purple-300 border border-purple-500/20">
                MAP-Elites Dimension View
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Preserves the best-known agent on every single evaluation metric, coupled with ε-probing to prevent silent regression on unwatched dimensions.
            </p>
          </div>
        </div>

        {/* Epsilon-probing summary */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-800 text-xs">
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-purple-400 font-semibold">Random ε-Probing Audit Mechanism:</span>
            <p className="mt-1 text-[11px] text-slate-400">
              When an evaluator is soft-pruned, the system randomly forces evaluation with probability <span className="text-purple-300 font-mono font-bold">ε = 0.30</span> (30%) to audit unwatched metrics.
            </p>
          </div>
          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-purple-400 font-semibold">Total Soft-Probing Audits Triggered:</span>
            <div className="text-lg font-bold font-mono text-purple-300 mt-0.5">
              {probeRecords.length} audits logged
            </div>
            <p className="text-[11px] text-slate-500">Random audits verifying dropped dimensions</p>
          </div>
        </div>
      </div>

      {/* Main Dimension Elites Panel */}
      <Gap5DimensionArchive dimensionElitesData={dimensionElitesData} />
    </div>
  );
}

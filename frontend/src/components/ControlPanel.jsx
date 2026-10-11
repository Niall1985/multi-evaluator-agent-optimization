import React, { useState } from 'react';
import { Play, FastForward, RotateCcw, Sliders, CheckCircle2 } from 'lucide-react';

export default function ControlPanel({
  metadata,
  strategy,
  setStrategy,
  selectedTask,
  setSelectedTask,
  archetype,
  setArchetype,
  acquisitionType,
  setAcquisitionType,
  onRunStep,
  onReset,
  isRunning,
}) {
  const [batchSize, setBatchSize] = useState(5);

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800/80 mb-4">
        <div className="flex items-center space-x-2">
          <Sliders className="w-5 h-5 text-emerald-400" />
          <h2 className="text-base font-semibold text-white">Interactive Experiment Controls</h2>
        </div>
        <span className="text-xs text-slate-400">Configure parameters & step generation</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-5">
        {/* Strategy Selector */}
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1.5">Condition / Strategy</label>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            disabled={isRunning}
            className="w-full bg-slate-950 border border-slate-700/80 text-white rounded-lg px-3 py-2 text-xs focus:ring-1 focus:ring-emerald-500 focus:outline-none"
          >
            {metadata?.strategies?.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        </div>

        {/* Benchmark Task */}
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1.5">Benchmark Target Task</label>
          <select
            value={selectedTask || ''}
            onChange={(e) => setSelectedTask(e.target.value || null)}
            disabled={isRunning}
            className="w-full bg-slate-950 border border-slate-700/80 text-white rounded-lg px-3 py-2 text-xs focus:ring-1 focus:ring-emerald-500 focus:outline-none"
          >
            <option value="">All Benchmark Tasks (Suite / Random)</option>
            {metadata?.tasks?.map((t) => (
              <option key={t.id} value={t.id}>
                {t.id} - {t.name}
              </option>
            ))}
          </select>
        </div>

        {/* Initial Archetype */}
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1.5">Seed Archetype</label>
          <select
            value={archetype}
            onChange={(e) => setArchetype(e.target.value)}
            disabled={isRunning}
            className="w-full bg-slate-950 border border-slate-700/80 text-white rounded-lg px-3 py-2 text-xs focus:ring-1 focus:ring-emerald-500 focus:outline-none"
          >
            {metadata?.archetypes?.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>

        {/* Acquisition Type (Gap 6) */}
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1.5">BO Acquisition (Gap 6)</label>
          <select
            value={acquisitionType}
            onChange={(e) => setAcquisitionType(e.target.value)}
            disabled={isRunning}
            className="w-full bg-slate-950 border border-slate-700/80 text-white rounded-lg px-3 py-2 text-xs focus:ring-1 focus:ring-emerald-500 focus:outline-none"
          >
            {metadata?.acquisition_types?.map((acq) => (
              <option key={acq.id} value={acq.id}>
                {acq.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <div className="flex items-center space-x-3">
          <button
            onClick={() => onRunStep(1)}
            disabled={isRunning}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-emerald-900/30 transition active:scale-95"
          >
            <Play className="w-3.5 h-3.5" />
            <span>Run 1 Step</span>
          </button>

          <button
            onClick={() => onRunStep(batchSize)}
            disabled={isRunning}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-indigo-900/30 transition active:scale-95"
          >
            <FastForward className="w-3.5 h-3.5" />
            <span>Run {batchSize} Generations</span>
          </button>

          <div className="flex items-center space-x-1.5 bg-slate-950 border border-slate-800 rounded-lg px-2 py-1">
            <span className="text-[11px] text-slate-400">Batch:</span>
            {[3, 5, 10].map((num) => (
              <button
                key={num}
                onClick={() => setBatchSize(num)}
                className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                  batchSize === num
                    ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 font-semibold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {num}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={onReset}
          disabled={isRunning}
          className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-slate-800/80 hover:bg-rose-950/40 hover:text-rose-300 hover:border-rose-800/60 border border-slate-700/60 text-slate-300 text-xs font-medium transition active:scale-95"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Optimizer</span>
        </button>
      </div>
    </div>
  );
}

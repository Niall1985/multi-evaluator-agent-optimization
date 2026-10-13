import React, { useState } from 'react';
import { Play, FastForward, RotateCcw, Sliders, CheckCircle2, Code, Smile } from 'lucide-react';

const DOMAIN_OPTIONS = [
  {
    id: 'coding',
    name: 'Coding Agent',
    badge: 'Python & Algorithms',
    icon: Code,
    desc: 'HumanEval / MBPP benchmark tasks with unit & edge-case test suites',
    color: 'emerald',
  },
  {
    id: 'sentiment',
    name: 'Sentiment Analysis Agent',
    badge: 'SentiToolKit Neural',
    icon: Smile,
    desc: 'SentiAnalyzer neural classification (Positive, Negative, Neutral) with contrastive & negation suites',
    color: 'sky',
  },
];

export default function ControlPanel({
  agentDomain = 'coding',
  onDomainChange,
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

  // Filter tasks and archetypes by active domain
  const filteredTasks = (metadata?.tasks || []).filter(
    (t) => !t.domain || t.domain === agentDomain
  );

  const filteredArchetypes =
    metadata?.archetypes_by_domain?.[agentDomain] || metadata?.archetypes || [];

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur space-y-4">
      {/* Domain Selection Header */}
      <div>
        <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-3">
          <div className="flex items-center space-x-2">
            <Sliders className="w-5 h-5 text-emerald-400" />
            <h2 className="text-base font-semibold text-white">Target Agent Domain</h2>
          </div>
          <span className="text-xs text-slate-400">Choose specialized agent domain to benchmark & evolve</span>
        </div>

        {/* 2 Domain Radio Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {DOMAIN_OPTIONS.map((opt) => {
            const Icon = opt.icon;
            const isSelected = agentDomain === opt.id;
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => onDomainChange && onDomainChange(opt.id)}
                disabled={isRunning}
                className={`flex flex-col text-left p-3.5 rounded-xl border transition-all ${
                  isSelected
                    ? opt.id === 'coding'
                      ? 'bg-emerald-950/40 border-emerald-500/80 ring-1 ring-emerald-500/50 shadow-md shadow-emerald-950/40'
                      : opt.id === 'sentiment'
                      ? 'bg-sky-950/40 border-sky-500/80 ring-1 ring-sky-500/50 shadow-md shadow-sky-950/40'
                      : 'bg-amber-950/40 border-amber-500/80 ring-1 ring-amber-500/50 shadow-md shadow-amber-950/40'
                    : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200'
                }`}
              >
                <div className="flex items-center justify-between w-full mb-1">
                  <div className="flex items-center space-x-2">
                    <Icon
                      className={`w-4 h-4 ${
                        isSelected
                          ? opt.id === 'coding'
                            ? 'text-emerald-400'
                            : opt.id === 'sentiment'
                            ? 'text-sky-400'
                            : 'text-amber-400'
                          : 'text-slate-500'
                      }`}
                    />
                    <span className={`text-xs font-bold ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                      {opt.name}
                    </span>
                  </div>
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${
                      isSelected
                        ? opt.id === 'coding'
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                          : opt.id === 'sentiment'
                          ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30'
                          : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                        : 'bg-slate-800 text-slate-500'
                    }`}
                  >
                    {opt.badge}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed mt-1">{opt.desc}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Parameters Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
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
          <label className="block text-xs font-medium text-slate-400 mb-1.5">
            Target Benchmark Task ({filteredTasks.length} in domain)
          </label>
          <select
            value={selectedTask || ''}
            onChange={(e) => setSelectedTask(e.target.value || null)}
            disabled={isRunning}
            className="w-full bg-slate-950 border border-slate-700/80 text-white rounded-lg px-3 py-2 text-xs focus:ring-1 focus:ring-emerald-500 focus:outline-none"
          >
            <option value="">All Tasks in Domain (Randomized)</option>
            {filteredTasks.map((t) => (
              <option key={t.id} value={t.id}>
                [{t.task_type ? t.task_type.toUpperCase() : 'CODE'}] {t.name}
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
            {filteredArchetypes.map((a) => (
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

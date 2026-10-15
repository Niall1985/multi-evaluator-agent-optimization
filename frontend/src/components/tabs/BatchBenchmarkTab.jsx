import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Download,
  Play,
  RotateCcw,
  CheckCircle2,
  Clock,
  DollarSign,
  TrendingDown,
  TrendingUp,
  Layers,
  Search,
  Filter,
  Code,
  Smile,
  ShieldAlert,
  ChevronRight,
  Database,
  FileSpreadsheet,
  Sprout
} from 'lucide-react';

export default function BatchBenchmarkTab() {
  const [statusData, setStatusData] = useState(null);
  const [records, setRecords] = useState([]);
  const [seedRecords, setSeedRecords] = useState([]);
  const [comparisonRecords, setComparisonRecords] = useState([]);
  const [activeSheet, setActiveSheet] = useState('evolved'); // 'evolved' | 'seeds' | 'comparison'
  const [filterDomain, setFilterDomain] = useState('all');
  const [filterAblation, setFilterAblation] = useState('all');
  const [filterAcq, setFilterAcq] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [isStarting, setIsStarting] = useState(false);
  const [activePage, setActivePage] = useState(1);
  const rowsPerPage = 15;

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/batch_benchmark/status');
      const data = await res.json();
      setStatusData(data);
    } catch (e) {
      console.warn('Failed to fetch benchmark sweep status', e);
    }
  };

  const fetchResults = async () => {
    try {
      const res = await fetch('/api/batch_benchmark/results');
      const data = await res.json();
      if (data) {
        if (data.records) setRecords(data.records);
        if (data.seed_records) setSeedRecords(data.seed_records);
        if (data.comparison_records) setComparisonRecords(data.comparison_records);
      }
    } catch (e) {
      console.warn('Failed to fetch benchmark results', e);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchResults();
    const interval = setInterval(() => {
      fetchStatus();
      if (statusData?.is_running) {
        fetchResults();
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [statusData?.is_running]);

  const handleStartSweep = async () => {
    setIsStarting(true);
    try {
      await fetch('/api/batch_benchmark/start', { method: 'POST' });
      await fetchStatus();
      await fetchResults();
    } catch (e) {
      console.error('Failed to trigger sweep', e);
    } finally {
      setIsStarting(false);
    }
  };

  // Determine which dataset is currently active
  const currentDataset =
    activeSheet === 'evolved'
      ? records
      : activeSheet === 'seeds'
      ? seedRecords
      : comparisonRecords;

  // Filter active dataset
  const filteredRecords = currentDataset.filter((r) => {
    if (filterDomain !== 'all' && r.agent_domain !== filterDomain) return false;
    if (filterAblation !== 'all' && r.ablation_strategy !== filterAblation) return false;
    if (filterAcq !== 'all' && r.bo_acquisition !== filterAcq) return false;
    if (searchTerm) {
      const s = searchTerm.toLowerCase();
      const matchId = (r.agent_id || r.seed_agent_id || r.best_evolved_agent_id || '').toLowerCase().includes(s);
      const matchTask = (r.task_name || r.task_id || '').toLowerCase().includes(s);
      const matchArch = (r.seed_archetype || '').toLowerCase().includes(s);
      const matchPrompt = (r.system_prompt || '').toLowerCase().includes(s);
      if (!matchId && !matchTask && !matchArch && !matchPrompt) return false;
    }
    return true;
  });

  const totalPages = Math.ceil(filteredRecords.length / rowsPerPage) || 1;
  const paginatedRecords = filteredRecords.slice((activePage - 1) * rowsPerPage, activePage * rowsPerPage);

  const stats = statusData?.summary_stats || {};
  const isRunning = statusData?.is_running || false;
  const progressPct = statusData?.progress_pct || 0;

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-900/40 rounded-xl p-5 shadow-xl backdrop-blur relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="flex items-center space-x-2">
              <span className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                <Database className="w-5 h-5" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Multi-Domain Benchmark Studio (100 Runs / 300 Agents + 100 Seeds)
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                Separate Seed & Evolved Sheets
              </span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed max-w-3xl">
              Systematic evaluation across <strong>Coding</strong> (60 runs) and <strong>Sentiment Analysis</strong> (40 runs) agents over all 5 Ablations, Seed Archetypes, and BO Acquisitions across 3 generations, with baseline seed records stored in a dedicated comparison sheet.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <button
              onClick={handleStartSweep}
              disabled={isRunning || isStarting}
              className="flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 text-white text-xs font-bold shadow-lg shadow-emerald-950 transition active:scale-95"
            >
              <Play className={`w-3.5 h-3.5 ${isRunning ? 'animate-spin' : ''}`} />
              <span>{isRunning ? 'Sweep In Progress...' : 'Run Everything'}</span>
            </button>

            <a
              href="/api/batch_benchmark/download"
              download="comprehensive_multi_domain_benchmark_results.csv"
              className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-bold transition shadow-sm"
              title="Download 300 Evolved Agents CSV"
            >
              <Download className="w-3.5 h-3.5 text-indigo-400" />
              <span>Evolved CSV</span>
            </a>

            <a
              href="/api/batch_benchmark/download_seeds"
              download="seed_agents_benchmark_results.csv"
              className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-bold transition shadow-sm"
              title="Download 100 Seed Agents Baseline CSV"
            >
              <Sprout className="w-3.5 h-3.5 text-amber-400" />
              <span>Seeds CSV</span>
            </a>

            <a
              href="/api/batch_benchmark/download_excel"
              download="multi_domain_benchmark_results_all_sheets.xlsx"
              className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-emerald-950/60 hover:bg-emerald-900/80 border border-emerald-500/40 text-emerald-300 text-xs font-bold transition shadow-sm"
              title="Download Complete Excel Workbook (All 3 Sheets)"
            >
              <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
              <span>Excel (.xlsx)</span>
            </a>
          </div>
        </div>

        {/* Live Progress Bar */}
        {(isRunning || progressPct > 0) && (
          <div className="mt-5 pt-4 border-t border-slate-800/80 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 font-medium flex items-center space-x-2">
                <span className={`inline-block w-2 h-2 rounded-full ${isRunning ? 'bg-emerald-400 animate-ping' : 'bg-slate-500'}`} />
                <span>
                  {isRunning
                    ? `Running Run ${statusData?.current_run_idx || 0} / 100 • Agent Gen ${statusData?.current_generation || 0} / 300 (${statusData?.current_domain || ''})`
                    : `Completed ${records.length} Evolved Agents + ${seedRecords.length || 100} Seed Baselines across 100 Runs (100% Complete)`}
                </span>
              </span>
              <span className="text-emerald-400 font-mono font-bold">{progressPct}%</span>
            </div>
            <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
              <div
                className="bg-gradient-to-r from-indigo-500 via-teal-400 to-emerald-400 h-2 transition-all duration-300 rounded-full"
                style={{ width: `${Math.min(100, Math.max(0, progressPct))}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Experimental Design & Comparison KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-1 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Runs & Total Agents</span>
            <Database className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-xl font-bold text-white">100 Runs • 400 Agents</div>
          <div className="text-[11px] text-slate-400">
            300 Evolved (Gen 1-3) + 100 Seeds (Gen 0)
          </div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-1 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Fitness Optimization</span>
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-xl font-bold text-emerald-300 font-mono">
            {stats.mean_fitness || '0.0000'}
          </div>
          <div className="text-[11px] text-slate-400">
            Baseline Seed: <span className="text-slate-300 font-mono">{stats.seed_mean_fitness || '0.0000'}</span>{' '}
            <span className="text-emerald-400 font-bold font-mono">
              ({stats.fitness_gain >= 0 ? `+${stats.fitness_gain}` : stats.fitness_gain})
            </span>
          </div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-1 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Test Pass Rate Gain</span>
            <CheckCircle2 className="w-4 h-4 text-teal-400" />
          </div>
          <div className="text-xl font-bold text-teal-300 font-mono">
            {stats.mean_test_pass_rate !== undefined ? `${stats.mean_test_pass_rate}%` : 'N/A'}
          </div>
          <div className="text-[11px] text-slate-400">
            Seed Pass Rate: <span className="text-slate-300 font-mono">{stats.seed_mean_pass_rate || 0}%</span>{' '}
            <span className="text-teal-400 font-bold font-mono">
              ({stats.pass_rate_gain >= 0 ? `+${stats.pass_rate_gain}%` : `${stats.pass_rate_gain}%`})
            </span>
          </div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-1 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Adaptive Net Savings</span>
            <TrendingDown className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-xl font-bold text-indigo-300 font-mono">
            ${stats.total_saved_dollars !== undefined ? stats.total_saved_dollars.toFixed(4) : '0.0000'}
          </div>
          <div className="text-[11px] text-emerald-400 font-semibold font-mono">
            {stats.overall_pct_saved !== undefined ? `${stats.overall_pct_saved}% saved vs naive` : 'N/A'}
          </div>
        </div>
      </div>

      {/* Sheet View Switcher & Filters */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-3 backdrop-blur">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-slate-800/80">
          {/* Sheet Selector Tabs */}
          <div className="flex items-center space-x-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => {
                setActiveSheet('evolved');
                setActivePage(1);
              }}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                activeSheet === 'evolved'
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-950'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              <span>Evolved Agents ({records.length})</span>
            </button>

            <button
              onClick={() => {
                setActiveSheet('seeds');
                setActivePage(1);
              }}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                activeSheet === 'seeds'
                  ? 'bg-amber-600 text-white shadow-md shadow-amber-950'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sprout className="w-3.5 h-3.5" />
              <span>Seed Agents Gen 0 ({seedRecords.length || 100})</span>
            </button>

            <button
              onClick={() => {
                setActiveSheet('comparison');
                setActivePage(1);
              }}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                activeSheet === 'comparison'
                  ? 'bg-emerald-600 text-white shadow-md shadow-emerald-950'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <TrendingUp className="w-3.5 h-3.5" />
              <span>Run Comparison Summary ({comparisonRecords.length || 100})</span>
            </button>
          </div>

          {/* Search Box */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search agent ID, task, archetype..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setActivePage(1);
              }}
              className="bg-slate-950 border border-slate-700/80 rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 w-64"
            />
          </div>
        </div>

        {/* Filter Dropdowns */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
          <div>
            <label className="block text-[11px] text-slate-400 mb-1">Domain</label>
            <select
              value={filterDomain}
              onChange={(e) => {
                setFilterDomain(e.target.value);
                setActivePage(1);
              }}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none"
            >
              <option value="all">All Domains (Coding + Sentiment)</option>
              <option value="coding">Coding Agent Only</option>
              <option value="sentiment">Sentiment Agent Only</option>
            </select>
          </div>

          <div>
            <label className="block text-[11px] text-slate-400 mb-1">Ablation Condition</label>
            <select
              value={filterAblation}
              onChange={(e) => {
                setFilterAblation(e.target.value);
                setActivePage(1);
              }}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none"
            >
              <option value="all">All 5 Ablation Conditions</option>
              <option value="full_adaptive">Full Adaptive</option>
              <option value="no_pruning">Ablation 1: No Pruning</option>
              <option value="static_cascade">Ablation 2: Static Cascade</option>
              <option value="single_metric">Ablation 3: Single Metric</option>
              <option value="ucb1_bandit">Ablation 4: UCB1 Bandit</option>
            </select>
          </div>

          <div>
            <label className="block text-[11px] text-slate-400 mb-1">BO Acquisition</label>
            <select
              value={filterAcq}
              onChange={(e) => {
                setFilterAcq(e.target.value);
                setActivePage(1);
              }}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none"
            >
              <option value="all">All Acquisitions (GP-UCB & EI)</option>
              <option value="gp_ucb">GP-UCB</option>
              <option value="expected_improvement">Expected Improvement (EI)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Results Table */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <div className="space-y-0.5">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <span>
                {activeSheet === 'evolved' && 'Evolved Agents (300 Generations: Gen 1, 2, 3)'}
                {activeSheet === 'seeds' && 'Seed Agents Baseline Sheet (100 Runs: Generation 0 Roots)'}
                {activeSheet === 'comparison' && 'Run Comparison: Seed Baseline vs. Peak Evolved Agent (ΔF & Pass Gain)'}
              </span>
            </h3>
            <p className="text-[11px] text-slate-400">
              {activeSheet === 'comparison'
                ? 'Shows the exact performance improvements achieved by the multi-objective optimization loop for each run.'
                : 'Prioritized columns: 1. Identification → 2. Verification & Fitness → 3. Pruning & Economics → 4. Evaluator Costs → 5. fANOVA Sensitivity'}
            </p>
          </div>
          <div className="text-xs text-slate-400 font-mono">
            Page {activePage} of {totalPages}
          </div>
        </div>

        <div className="overflow-x-auto">
          {activeSheet === 'comparison' ? (
            /* Comparison Table */
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="px-3 py-3 font-semibold text-slate-300">Run</th>
                  <th className="px-3 py-3 font-semibold text-slate-300">Domain</th>
                  <th className="px-3 py-3 font-semibold text-slate-300">Ablation</th>
                  <th className="px-3 py-3 font-semibold text-slate-300">Seed Archetype</th>
                  <th className="px-3 py-3 font-semibold text-slate-300">BO Acq</th>
                  <th className="px-3 py-3 font-semibold text-amber-400">Seed Fitness (G0)</th>
                  <th className="px-3 py-3 font-semibold text-emerald-400">Peak Evolved Fitness</th>
                  <th className="px-3 py-3 font-semibold text-emerald-300">Fitness Gain (ΔF)</th>
                  <th className="px-3 py-3 font-semibold text-amber-400">Seed Pass %</th>
                  <th className="px-3 py-3 font-semibold text-teal-400">Peak Pass %</th>
                  <th className="px-3 py-3 font-semibold text-teal-300">Pass Gain %</th>
                  <th className="px-3 py-3 font-semibold text-indigo-400">Cost Spent</th>
                  <th className="px-3 py-3 font-semibold text-emerald-400">Net Saved ($)</th>
                  <th className="px-3 py-3 font-semibold text-emerald-400">% Saved</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                {paginatedRecords.length === 0 ? (
                  <tr>
                    <td colSpan="14" className="text-center py-8 text-slate-500">
                      No comparison records found.
                    </td>
                  </tr>
                ) : (
                  paginatedRecords.map((r, idx) => (
                    <tr key={`${r.run_id}_comp_${idx}`} className="hover:bg-slate-800/40 transition">
                      <td className="px-3 py-2.5 text-slate-300 font-semibold">#{r.run_id}</td>
                      <td className="px-3 py-2.5 font-sans">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          r.agent_domain === 'coding' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-sky-500/20 text-sky-300 border border-sky-500/30'
                        }`}>
                          {r.agent_domain}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 font-sans text-slate-300 text-[11px] truncate max-w-[120px]">{r.ablation_strategy}</td>
                      <td className="px-3 py-2.5 font-sans text-slate-300 text-[11px] truncate max-w-[140px]">{r.seed_archetype}</td>
                      <td className="px-3 py-2.5 font-sans text-slate-400 text-[10px]">{r.bo_acquisition === 'gp_ucb' ? 'GP-UCB' : 'EI'}</td>
                      <td className="px-3 py-2.5 text-amber-300">{Number(r.seed_fitness).toFixed(4)}</td>
                      <td className="px-3 py-2.5 text-emerald-400 font-bold">{Number(r.best_evolved_fitness).toFixed(4)} (G{r.best_evolved_generation})</td>
                      <td className="px-3 py-2.5">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          r.fitness_improvement_delta >= 0 ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'
                        }`}>
                          {r.fitness_improvement_delta >= 0 ? `+${Number(r.fitness_improvement_delta).toFixed(4)}` : Number(r.fitness_improvement_delta).toFixed(4)}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-amber-300">{r.seed_pass_rate_pct}%</td>
                      <td className="px-3 py-2.5 text-teal-300 font-bold">{r.best_evolved_pass_rate_pct}%</td>
                      <td className="px-3 py-2.5">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          r.pass_rate_gain_pct >= 0 ? 'bg-teal-500/20 text-teal-300' : 'bg-rose-500/20 text-rose-300'
                        }`}>
                          {r.pass_rate_gain_pct >= 0 ? `+${r.pass_rate_gain_pct}%` : `${r.pass_rate_gain_pct}%`}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-slate-300">${Number(r.adaptive_cost_spent).toFixed(4)}</td>
                      <td className="px-3 py-2.5 text-emerald-400 font-semibold">${Number(r.net_savings_dollars).toFixed(4)}</td>
                      <td className="px-3 py-2.5 text-emerald-300 font-bold">{r.net_savings_pct}%</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          ) : (
            /* Evolved and Seed Agents Table */
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="px-3 py-3 font-semibold text-slate-300">Run</th>
                  <th className="px-3 py-3 font-semibold text-slate-300">Domain</th>
                  <th className="px-3 py-3 font-semibold text-slate-300">Ablation</th>
                  <th className="px-3 py-3 font-semibold text-slate-300">Seed Archetype</th>
                  <th className="px-3 py-3 font-semibold text-slate-300">BO Acq</th>
                  <th className="px-2 py-3 font-semibold text-slate-300">Gen</th>

                  {/* 2. Verification & Performance (Primary) */}
                  <th className="px-3 py-3 font-semibold text-emerald-400">Fitness</th>
                  <th className="px-3 py-3 font-semibold text-teal-400">Pass Rate</th>
                  <th className="px-3 py-3 font-semibold text-teal-400">Cases</th>
                  <th className="px-3 py-3 font-semibold text-teal-400">μ₁ Correctness</th>

                  {/* 3. Pruning & Economics */}
                  <th className="px-3 py-3 font-semibold text-indigo-400">Pruning Status</th>
                  <th className="px-3 py-3 font-semibold text-indigo-400">Cost Spent</th>
                  <th className="px-3 py-3 font-semibold text-indigo-400">Naive Cost</th>
                  <th className="px-3 py-3 font-semibold text-emerald-400">Net Saved ($)</th>
                  <th className="px-3 py-3 font-semibold text-emerald-400">% Saved</th>

                  {/* 4. Evaluator Costs */}
                  <th className="px-2 py-3 font-semibold text-slate-400">c(μ₁)</th>
                  <th className="px-2 py-3 font-semibold text-slate-400">c(μ₂)</th>
                  <th className="px-2 py-3 font-semibold text-slate-400">c(μ₃)</th>
                  <th className="px-2 py-3 font-semibold text-slate-400">c(μ₄)</th>
                  <th className="px-2 py-3 font-semibold text-slate-400">c(μ₅)</th>
                  <th className="px-2 py-3 font-semibold text-slate-400">c(μ₆)</th>

                  {/* 5. fANOVA Sensitivity */}
                  <th className="px-3 py-3 font-semibold text-amber-400">fANOVA V(f)</th>
                  <th className="px-2 py-3 font-semibold text-amber-400">S₁</th>
                  <th className="px-2 py-3 font-semibold text-amber-400">S₂</th>
                  <th className="px-2 py-3 font-semibold text-amber-400">S₃</th>
                  <th className="px-2 py-3 font-semibold text-amber-400">S₄</th>
                  <th className="px-2 py-3 font-semibold text-amber-400">S₅</th>
                  <th className="px-2 py-3 font-semibold text-amber-400">S₆</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                {paginatedRecords.length === 0 ? (
                  <tr>
                    <td colSpan="26" className="text-center py-8 text-slate-500">
                      No records found. Click <strong>"Run Everything"</strong> to execute the full sweep.
                    </td>
                  </tr>
                ) : (
                  paginatedRecords.map((r, idx) => (
                    <tr key={`${r.run_id}_${r.generation}_${idx}`} className="hover:bg-slate-800/40 transition">
                      <td className="px-3 py-2.5 text-slate-300 font-semibold">#{r.run_id}</td>
                      <td className="px-3 py-2.5 font-sans">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          r.agent_domain === 'coding' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-sky-500/20 text-sky-300 border border-sky-500/30'
                        }`}>
                          {r.agent_domain}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 font-sans text-slate-300 text-[11px] truncate max-w-[120px]" title={r.ablation_strategy}>
                        {r.ablation_strategy}
                      </td>
                      <td className="px-3 py-2.5 font-sans text-slate-300 text-[11px] truncate max-w-[140px]" title={r.seed_archetype}>
                        {r.seed_archetype}
                      </td>
                      <td className="px-3 py-2.5 font-sans text-slate-400 text-[10px]">
                        {r.bo_acquisition === 'gp_ucb' ? 'GP-UCB' : 'EI'}
                      </td>
                      <td className="px-2 py-2.5 text-center text-slate-300 font-bold">
                        {r.generation === 0 ? <span className="text-amber-400">G0 (Seed)</span> : `G${r.generation}`}
                      </td>

                      {/* Primary Performance */}
                      <td className="px-3 py-2.5 text-emerald-400 font-bold">{Number(r.fitness).toFixed(4)}</td>
                      <td className="px-3 py-2.5">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          r.testcases_pass_rate_pct >= 80 ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'
                        }`}>
                          {r.testcases_pass_rate_pct}%
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-slate-300">
                        {r.testcases_passed} / {r.testcases_total}
                      </td>
                      <td className="px-3 py-2.5 text-teal-300">{Number(r.mu_1_functional_correctness).toFixed(3)}</td>

                      {/* Pruning & Economics */}
                      <td className="px-3 py-2.5 font-sans">
                        <span className={`px-2 py-0.5 rounded text-[10px] ${
                          r.is_deep_tier_pruned ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                        }`} title={r.pruning_status}>
                          {r.generation === 0 ? 'Seed Base' : r.is_deep_tier_pruned ? 'Pruned' : 'Deep Active'}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-slate-300">${Number(r.adaptive_cost_spent).toFixed(4)}</td>
                      <td className="px-3 py-2.5 text-slate-500">${Number(r.total_cost_without_pruning).toFixed(4)}</td>
                      <td className="px-3 py-2.5 text-emerald-400 font-semibold">${Number(r.net_savings_dollars).toFixed(4)}</td>
                      <td className="px-3 py-2.5 text-emerald-300 font-bold">{r.net_savings_pct}%</td>

                      {/* Evaluator Costs */}
                      <td className="px-2 py-2.5 text-slate-400">{r.cost_mu1_correctness ? `$${r.cost_mu1_correctness}` : '—'}</td>
                      <td className="px-2 py-2.5 text-slate-400">{r.cost_mu2_latency ? `$${r.cost_mu2_latency}` : '—'}</td>
                      <td className="px-2 py-2.5 text-slate-400">{r.cost_mu3_conciseness ? `$${r.cost_mu3_conciseness}` : '—'}</td>
                      <td className="px-2 py-2.5 text-slate-400">{r.cost_mu4_reasoning ? `$${r.cost_mu4_reasoning}` : '—'}</td>
                      <td className="px-2 py-2.5 text-slate-400">{r.cost_mu5_edge_cases ? `$${r.cost_mu5_edge_cases}` : '—'}</td>
                      <td className="px-2 py-2.5 text-slate-400">{r.cost_mu6_safety ? `$${r.cost_mu6_safety}` : '—'}</td>

                      {/* fANOVA Sensitivity */}
                      <td className="px-3 py-2.5 text-amber-300">{Number(r.fanova_total_variance).toFixed(4)}</td>
                      <td className="px-2 py-2.5 text-slate-300">{Number(r.fanova_S1_correctness).toFixed(2)}</td>
                      <td className="px-2 py-2.5 text-slate-300">{Number(r.fanova_S2_latency).toFixed(2)}</td>
                      <td className="px-2 py-2.5 text-slate-300">{Number(r.fanova_S3_conciseness).toFixed(2)}</td>
                      <td className="px-2 py-2.5 text-slate-300">{Number(r.fanova_S4_reasoning).toFixed(2)}</td>
                      <td className="px-2 py-2.5 text-slate-300">{Number(r.fanova_S5_edge_cases).toFixed(2)}</td>
                      <td className="px-2 py-2.5 text-slate-300">{Number(r.fanova_S6_safety).toFixed(2)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          )}
        </div>

        {/* Pagination Footer */}
        {totalPages > 1 && (
          <div className="p-3 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between">
            <span className="text-xs text-slate-400">
              Showing {(activePage - 1) * rowsPerPage + 1} - {Math.min(activePage * rowsPerPage, filteredRecords.length)} of {filteredRecords.length} records
            </span>
            <div className="flex items-center space-x-1.5">
              <button
                onClick={() => setActivePage((p) => Math.max(1, p - 1))}
                disabled={activePage === 1}
                className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-xs text-slate-300 font-medium"
              >
                Previous
              </button>
              <span className="text-xs text-slate-400 font-mono px-2">
                {activePage} / {totalPages}
              </span>
              <button
                onClick={() => setActivePage((p) => Math.min(totalPages, p + 1))}
                disabled={activePage === totalPages}
                className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-xs text-slate-300 font-medium"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

import React, { useState } from 'react';
import { CheckSquare, CheckCircle, XCircle, AlertTriangle, Code, Terminal, ChevronRight, Copy, Check } from 'lucide-react';

export default function TestSuiteTab({ historyRecords }) {
  const [selectedGen, setSelectedGen] = useState(
    historyRecords.length > 0 ? historyRecords[historyRecords.length - 1].generation : 0
  );
  const [copied, setCopied] = useState(false);

  if (!historyRecords || historyRecords.length === 0) {
    return (
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-8 text-center text-slate-400">
        No generation test executions logged yet. Run a step to inspect test cases.
      </div>
    );
  }

  // Find record for selected generation
  const activeRecord = historyRecords.find((r) => r.generation === selectedGen) || historyRecords[historyRecords.length - 1];
  const testSuite = activeRecord.test_suite;
  const cases = testSuite?.cases || [];
  const passedCount = cases.filter((c) => c.passed).length;
  const totalCount = cases.length;
  const passRate = totalCount > 0 ? (passedCount / totalCount) * 100 : 0;

  const handleCopyCode = () => {
    if (activeRecord.solution) {
      navigator.clipboard.writeText(activeRecord.solution);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Generation Selector Bar */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg backdrop-blur flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-2">
          <CheckSquare className="w-5 h-5 text-emerald-400" />
          <h2 className="text-sm font-bold text-white">Agent Execution & Test Suite Inspector</h2>
        </div>

        {/* Generation Pill Buttons */}
        <div className="flex items-center space-x-1.5 overflow-x-auto max-w-full py-1">
          <span className="text-xs text-slate-400 mr-1">Generation:</span>
          {historyRecords.map((r) => (
            <button
              key={r.generation}
              onClick={() => setSelectedGen(r.generation)}
              className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition ${
                selectedGen === r.generation
                  ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-900/40'
                  : 'bg-slate-950 text-slate-400 hover:text-white border border-slate-800'
              }`}
            >
              Gen {r.generation}
            </button>
          ))}
        </div>
      </div>

      {/* Selected Generation Summary Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Task & Pass Rate Summary */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <span className="text-[10px] uppercase font-bold text-slate-400">Agent Identifier</span>
              <div className="text-sm font-bold font-mono text-emerald-400">{activeRecord.child_id}</div>
            </div>
            <div className="text-right">
              <span className="text-[10px] uppercase font-bold text-slate-400">Task Target</span>
              <div className="text-xs font-mono text-slate-200">{activeRecord.task_id}</div>
            </div>
          </div>

          {/* Pass Rate Metric */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div>
              <div className="text-xs text-slate-400 font-medium">Test Suite Pass Rate</div>
              <div className="text-2xl font-bold font-mono text-white mt-0.5">
                {passRate.toFixed(1)}%
              </div>
              <div className="text-[11px] text-slate-400">
                {passedCount} of {totalCount} test assertions passed
              </div>
            </div>
            <div className={`w-12 h-12 rounded-full flex items-center justify-center border-2 ${
              passRate === 100
                ? 'border-emerald-500 bg-emerald-500/10 text-emerald-400'
                : passRate > 0
                ? 'border-amber-500 bg-amber-500/10 text-amber-400'
                : 'border-rose-500 bg-rose-500/10 text-rose-400'
            }`}>
              {passRate === 100 ? (
                <CheckCircle className="w-6 h-6" />
              ) : (
                <XCircle className="w-6 h-6" />
              )}
            </div>
          </div>

          {/* Mutation Info */}
          <div className="text-xs space-y-1.5 text-slate-400">
            <div className="flex justify-between">
              <span>Strategy:</span>
              <span className="font-mono text-slate-200">{activeRecord.strategy}</span>
            </div>
            <div className="flex justify-between">
              <span>Mutation Arm:</span>
              <span className="font-mono text-indigo-300">{activeRecord.mutation_type}</span>
            </div>
            <div className="flex justify-between">
              <span>Fitness Score:</span>
              <span className="font-mono text-white font-bold">{activeRecord.fitness?.toFixed(4)}</span>
            </div>
          </div>
        </div>

        {/* Code Output Generated by Agent */}
        <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
              <div className="flex items-center space-x-2">
                <Code className="w-4 h-4 text-emerald-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">Generated Code Output</h3>
              </div>
              <button
                onClick={handleCopyCode}
                className="flex items-center space-x-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy Code'}</span>
              </button>
            </div>
            <pre className="bg-slate-950 border border-slate-800/90 rounded-lg p-3.5 text-xs text-emerald-300/95 font-mono whitespace-pre-wrap max-h-56 overflow-y-auto">
              {activeRecord.solution || '# No code output generated'}
            </pre>
          </div>
        </div>
      </div>

      {/* Test Cases Pass / Fail Breakdown Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
          <div className="flex items-center space-x-2">
            <Terminal className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-bold text-white">Input & Output Assertions Breakdown</h3>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            {passedCount} Passed • {totalCount - passedCount} Failed
          </span>
        </div>

        {cases.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/70 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Type</th>
                  <th className="py-2.5 px-3">Input Arguments</th>
                  <th className="py-2.5 px-3">Expected Result</th>
                  <th className="py-2.5 px-3">Actual Result</th>
                  <th className="py-2.5 px-3">Error / Exception</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
                {cases.map((c, idx) => (
                  <tr key={idx} className={c.passed ? 'hover:bg-emerald-950/10' : 'bg-rose-950/10 hover:bg-rose-950/20'}>
                    <td className="py-2.5 px-3">
                      {c.passed ? (
                        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 text-[10px] font-bold">
                          <CheckCircle className="w-3 h-3 mr-0.5" />
                          <span>PASS</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-rose-500/20 border border-rose-500/30 text-rose-400 text-[10px] font-bold">
                          <XCircle className="w-3 h-3 mr-0.5" />
                          <span>FAIL</span>
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px] uppercase">
                      {c.type || 'test'}
                    </td>
                    <td className="py-2.5 px-3 text-slate-200 max-w-xs truncate" title={c.args}>
                      {c.args}
                    </td>
                    <td className="py-2.5 px-3 text-emerald-300 max-w-xs truncate" title={c.expected}>
                      {c.expected}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300 max-w-xs truncate" title={c.actual}>
                      {c.actual ?? 'None'}
                    </td>
                    <td className="py-2.5 px-3 text-rose-400 text-[11px] max-w-xs truncate" title={c.error}>
                      {c.error || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-6 text-center text-xs text-slate-500">
            No test case assertions executed for this generation.
          </div>
        )}
      </div>
    </div>
  );
}

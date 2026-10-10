import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import TelemetryCards from './components/TelemetryCards';
import ControlPanel from './components/ControlPanel';
import NavigationTabs from './components/NavigationTabs';

import OverviewTab from './components/tabs/OverviewTab';
import Gap1PruningTab from './components/tabs/Gap1PruningTab';
import Gap2BayesianTab from './components/tabs/Gap2BayesianTab';
import Gap3CostTab from './components/tabs/Gap3CostTab';
import Gap4FanovaTab from './components/tabs/Gap4FanovaTab';
import Gap5ArchiveTab from './components/tabs/Gap5ArchiveTab';
import Gap6JointTab from './components/tabs/Gap6JointTab';
import TestSuiteTab from './components/tabs/TestSuiteTab';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [status, setStatus] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [currentGen, setCurrentGen] = useState(0);
  const [fanova, setFanova] = useState(null);
  const [dimensionElites, setDimensionElites] = useState(null);
  const [historyRecords, setHistoryRecords] = useState([]);
  const [paretoAgents, setParetoAgents] = useState([]);
  const [metadata, setMetadata] = useState(null);

  // Form Controls
  const [strategy, setStrategy] = useState('full_adaptive');
  const [selectedTask, setSelectedTask] = useState(null);
  const [archetype, setArchetype] = useState('General Balanced Assistant');
  const [acquisitionType, setAcquisitionType] = useState('gp_ucb');
  const [isRunning, setIsRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Fetch all live data from backend
  const fetchData = async () => {
    try {
      const [statusRes, fanovaRes, elitesRes, histRes, paretoRes, metaRes] = await Promise.all([
        fetch('/api/status').then((r) => r.json()),
        fetch('/api/fanova').then((r) => r.json()),
        fetch('/api/dimension_elites').then((r) => r.json()),
        fetch('/api/history').then((r) => r.json()),
        fetch('/api/pareto').then((r) => r.json()),
        fetch('/api/metadata').then((r) => r.json()),
      ]);

      setStatus(statusRes);
      setTelemetry(statusRes.telemetry);
      setCurrentGen(statusRes.current_generation);
      setFanova(fanovaRes);
      setDimensionElites(elitesRes);
      setHistoryRecords(histRes.records || []);
      setParetoAgents(paretoRes || []);
      setMetadata(metaRes);
      setErrorMsg(null);
    } catch (err) {
      console.warn('Backend offline or initializing:', err);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Run Step(s)
  const handleRunStep = async (nSteps = 1) => {
    setIsRunning(true);
    setErrorMsg(null);
    try {
      const res = await fetch('/api/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_steps: nSteps,
          strategy: strategy,
          task_id: selectedTask,
        }),
      });
      const data = await res.json();
      if (data.success) {
        await fetchData();
      } else {
        setErrorMsg('Failed to run generation step.');
      }
    } catch (err) {
      setErrorMsg('Network error connecting to backend API.');
    } finally {
      setIsRunning(false);
    }
  };

  // Reset Optimizer
  const handleReset = async () => {
    setIsRunning(true);
    setErrorMsg(null);
    try {
      const res = await fetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          archetype: archetype,
          task_id: selectedTask,
          acquisition_type: acquisitionType,
        }),
      });
      const data = await res.json();
      if (data.success) {
        await fetchData();
      }
    } catch (err) {
      setErrorMsg('Failed to reset optimizer.');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar
        telemetry={telemetry}
        currentGen={currentGen}
        isRunning={isRunning}
        onRefresh={fetchData}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {errorMsg && (
          <div className="p-3 rounded-lg bg-rose-500/20 border border-rose-500/30 text-rose-300 text-xs font-semibold">
            {errorMsg}
          </div>
        )}

        {/* Global Telemetry KPI Cards */}
        <TelemetryCards telemetry={telemetry} fanova={fanova} />

        {/* Interactive Experiment Controls */}
        <ControlPanel
          metadata={metadata}
          strategy={strategy}
          setStrategy={setStrategy}
          selectedTask={selectedTask}
          setSelectedTask={setSelectedTask}
          archetype={archetype}
          setArchetype={setArchetype}
          acquisitionType={acquisitionType}
          setAcquisitionType={setAcquisitionType}
          onRunStep={handleRunStep}
          onReset={handleReset}
          isRunning={isRunning}
        />

        {/* Multi-Tab Navigation Bar */}
        <NavigationTabs activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Render Active Tab View */}
        <div className="transition-all duration-200">
          {activeTab === 'overview' && (
            <OverviewTab
              status={status}
              telemetry={telemetry}
              paretoAgents={paretoAgents}
              historyRecords={historyRecords}
              onSwitchTab={setActiveTab}
            />
          )}

          {activeTab === 'gap1' && (
            <Gap1PruningTab historyRecords={historyRecords} />
          )}

          {activeTab === 'gap2' && (
            <Gap2BayesianTab historyRecords={historyRecords} />
          )}

          {activeTab === 'gap3' && (
            <Gap3CostTab
              telemetry={telemetry}
              historyRecords={historyRecords}
              lambdaPenalty={status?.lambda_penalty || 0.5}
            />
          )}

          {activeTab === 'gap4' && (
            <Gap4FanovaTab fanova={fanova} />
          )}

          {activeTab === 'gap5' && (
            <Gap5ArchiveTab
              dimensionElitesData={dimensionElites}
              historyRecords={historyRecords}
            />
          )}

          {activeTab === 'gap6' && (
            <Gap6JointTab
              historyRecords={historyRecords}
              acquisitionType={acquisitionType}
            />
          )}

          {activeTab === 'tests' && (
            <TestSuiteTab historyRecords={historyRecords} />
          )}
        </div>
      </main>

      <footer className="border-t border-slate-800/80 bg-slate-900/40 py-4 text-center text-xs text-slate-500">
        Multi-Objective Agent Optimization Through Adaptive Evaluation • BCSE497J
      </footer>
    </div>
  );
}

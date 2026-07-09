import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getModels } from "../api.js";
import DataTable from "../components/DataTable.jsx";

const SUPERVISED = [
  {
    id: "linear-regression",
    path: "/linear-regression",
    number: "01",
    title: "Linear Regression",
    useCase: "Stock Price Change Prediction",
    desc: "Fits a straight-line relationship between today's price action and the size of tomorrow's move.",
    task: "regression",
  },
  {
    id: "logistic-regression",
    path: "/logistic-regression",
    number: "02",
    title: "Logistic Regression",
    useCase: "Stock Price Movement Prediction",
    desc: "Estimates the probability that tomorrow's close will rise or fall from today's price action.",
    task: "classification",
  },
  {
    id: "decision-tree",
    path: "/decision-tree",
    number: "03",
    title: "Decision Tree",
    useCase: "Stock Price Movement Prediction",
    desc: "Splits price and volume signals into simple yes/no rules to call the next move up or down.",
    task: "classification",
  },
  {
    id: "random-forest",
    path: "/random-forest",
    number: "04",
    title: "Random Forest",
    useCase: "Stock Price Movement Prediction",
    desc: "An ensemble of trees votes on whether the next close will rise or fall from today's OHLC and volume.",
    task: "classification",
  },
  {
    id: "gradient-boosting",
    path: "/gradient-boosting",
    number: "05",
    title: "Gradient Boosting",
    useCase: "Stock Price Change Prediction",
    desc: "An ensemble of shallow trees estimates the size of tomorrow's price move from today's OHLC and volume.",
    task: "regression",
  },
];

function taskBadgeClass(task) {
  if (task === "regression") return "border-signal2/30 !text-signal2";
  if (task === "classification") return "border-signal/30 !text-signal";
  return "border-warn/30 !text-warn";
}

function ModuleCard({ m, metrics }) {
  return (
    <Link
      to={m.path}
      className="group panel p-6 relative overflow-hidden hover:border-signal/30 transition"
    >
      <div className="absolute -right-6 -top-6 font-display text-8xl font-bold text-white/[0.03] group-hover:text-signal/[0.06] transition select-none">
        {m.number}
      </div>
      <div className="relative">
        <div className="flex items-center gap-2 mb-3">
          <span className="mono-tag">module {m.number}</span>
          <span
            className={`mono-tag !text-[10px] px-2 py-0.5 rounded-full border ${taskBadgeClass(
              m.task
            )}`}
          >
            {m.task}
          </span>
        </div>
        <h3 className="font-display text-xl font-semibold text-slate-50 group-hover:text-signal transition">
          {m.title}
        </h3>
        <p className="text-sm text-slate-500 mb-3">{m.useCase}</p>
        <p className="text-sm text-slate-400 leading-relaxed">{m.desc}</p>

        <div className="mt-5 flex items-center justify-between">
          <span className="text-signal text-sm font-medium flex items-center gap-1.5 group-hover:gap-2.5 transition-all">
            Open module →
          </span>
          {metrics && (
            <span className="mono-tag !text-slate-500">
              {Object.entries(metrics)
                .slice(0, 2)
                .map(([k, v]) => `${k}: ${v}`)
                .join(" · ")}
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

const PERFORMANCE_COLUMNS = ["Model", "Task", "Accuracy", "R2 Score", "MAE"];

function buildPerformanceRows(metrics) {
  return SUPERVISED.map((m) => {
    const met = metrics[m.id] || {};
    return {
      Model: m.title,
      Task: m.task,
      Accuracy: met.accuracy != null ? `${(met.accuracy * 100).toFixed(1)}%` : "—",
      "R2 Score": met.r2_score != null ? `${(met.r2_score * 100).toFixed(1)}%` : "—",
      MAE: met.mae != null ? met.mae.toFixed(2) : "—",
    };
  });
}

export default function Home() {
  const [metrics, setMetrics] = useState({});
  const [apiDown, setApiDown] = useState(false);

  useEffect(() => {
    getModels()
      .then((list) => {
        const map = {};
        list.forEach((m) => (map[m.id] = m.metrics));
        setMetrics(map);
      })
      .catch(() => setApiDown(true));
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-6 py-14">
      <div className="mb-14 max-w-2xl">
        <p className="mono-tag mb-3">5 trained models · live inference</p>
        <h1 className="font-display text-4xl md:text-5xl font-bold text-slate-50 leading-tight">
          Machine Learning, <span className="text-signal">signal by signal.</span>
        </h1>
        <p className="text-slate-400 mt-4 leading-relaxed">
          Five supervised algorithms, all trained on the same stock price
          dataset. Pick a module to inspect the training data, enter your own
          values, and run a live prediction against a model trained just now
          on the backend.
        </p>
        {apiDown && (
          <p className="mt-4 text-sm text-danger bg-danger/10 border border-danger/30 rounded-lg px-3 py-2 inline-block">
            Connection error — could not reach the backend API. Make sure the Flask server is running.
          </p>
        )}
      </div>

      <div className="flex items-center gap-3 mb-6">
        <h2 className="font-display text-2xl font-semibold text-slate-50">
          Supervised Learning
        </h2>
        <span className="mono-tag !text-slate-500">learns from labelled examples</span>
      </div>
      <div className="grid md:grid-cols-2 gap-5">
        {SUPERVISED.map((m) => (
          <ModuleCard key={m.id} m={m} metrics={metrics[m.id]} />
        ))}
      </div>

      <div className="flex items-center gap-3 mb-6 mt-14">
        <h2 className="font-display text-2xl font-semibold text-slate-50">
          Model Performance
        </h2>
        <span className="mono-tag !text-slate-500">held-out test metrics</span>
      </div>
      <div className="panel p-6">
        <DataTable
          columns={PERFORMANCE_COLUMNS}
          rows={buildPerformanceRows(metrics)}
        />
      </div>
    </div>
  );
}

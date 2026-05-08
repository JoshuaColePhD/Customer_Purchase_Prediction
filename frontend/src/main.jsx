import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const modelMetrics = [
  {
    name: "Random Forest",
    accuracy: 0.923,
    precision: 0.942,
    recall: 0.877,
    auc: 0.939,
    cv: "0.944 +/- 0.024",
  },
  {
    name: "Logistic Regression",
    accuracy: 0.843,
    precision: 0.855,
    recall: 0.769,
    auc: 0.903,
    cv: "0.887 +/- 0.027",
  },
];

const featureImportance = [
  ["Time Spent On Website", 0.1856],
  ["Discounts Availed", 0.1707],
  ["Age", 0.1465],
  ["Loyalty Program", 0.1432],
  ["Annual Income", 0.1226],
  ["Number Of Purchases", 0.1013],
  ["Avg Spending Per Purchase", 0.0951],
  ["Product Category", 0.0234],
  ["Gender", 0.0116],
];

const segmentRows = [
  ["Top 10%", "0.98", "Very high", "$1,900", "Premium offer"],
  ["Top 20%", "0.93", "High", "$3,300", "Primary campaign"],
  ["Top 30%", "0.81", "Medium", "$4,050", "Test discount"],
  ["Bottom 40%", "0.18", "Low", "$0", "Suppress spend"],
];

const rocPath = "M 26 214 C 58 168, 85 119, 124 92 C 165 63, 206 45, 258 35";
const logisticPath = "M 26 214 C 60 178, 96 140, 134 115 C 176 89, 214 71, 258 57";

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function money(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brandMark">CP</span>
        <div>
          <strong>Customer Purchase Prediction</strong>
          <span>ML portfolio product</span>
        </div>
      </div>
      <nav className="nav">
        {["Overview", "Model", "Campaign", "Drivers", "Segments"].map((item) => (
          <button className={item === "Overview" ? "active" : ""} key={item}>
            {item}
          </button>
        ))}
      </nav>
      <div className="sidebarNote">
        <span>Deployment candidate</span>
        <strong>Random Forest</strong>
        <p>Best balance of precision, recall, and ranking quality for campaign targeting.</p>
      </div>
    </aside>
  );
}

function MetricCard({ label, value, sub, tone = "teal" }) {
  return (
    <section className={`metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{sub}</small>
    </section>
  );
}

function ModelComparison() {
  return (
    <section className="panel modelPanel">
      <div className="panelHeader">
        <div>
          <h2>Model Comparison</h2>
          <p>Holdout performance with 5-fold CV stability check</p>
        </div>
      </div>
      <div className="modelRows">
        {modelMetrics.map((model) => (
          <div className="modelRow" key={model.name}>
            <div>
              <strong>{model.name}</strong>
              <span>CV ROC-AUC {model.cv}</span>
            </div>
            <Score label="ROC-AUC" value={model.auc} />
            <Score label="Precision" value={model.precision} />
            <Score label="Recall" value={model.recall} />
          </div>
        ))}
      </div>
    </section>
  );
}

function Score({ label, value }) {
  return (
    <div className="score">
      <span>{label}</span>
      <div className="track">
        <i style={{ width: `${value * 100}%` }} />
      </div>
      <strong>{value.toFixed(3)}</strong>
    </div>
  );
}

function ImpactSimulator() {
  const [outreachRate, setOutreachRate] = useState(20);
  const [revenue, setRevenue] = useState(100);

  const impact = useMemo(() => {
    const baseline = 0.4333;
    const targeted = Math.max(0.52, 1.035 - outreachRate * 0.0026);
    const contacted = Math.round(300 * (outreachRate / 100));
    const incrementalPurchases = Math.max(0, (targeted - baseline) * contacted);
    return {
      targeted,
      contacted,
      lift: targeted / baseline,
      incrementalPurchases,
      incrementalRevenue: incrementalPurchases * revenue,
    };
  }, [outreachRate, revenue]);

  return (
    <section className="panel impactPanel">
      <div className="panelHeader">
        <div>
          <h2>Business Impact Simulator</h2>
          <p>Translate propensity ranking into campaign economics</p>
        </div>
      </div>
      <div className="controlGrid">
        <label>
          <span>Outreach rate</span>
          <strong>{outreachRate}%</strong>
          <input
            type="range"
            min="10"
            max="50"
            step="5"
            value={outreachRate}
            onChange={(event) => setOutreachRate(Number(event.target.value))}
          />
        </label>
        <label>
          <span>Revenue per purchase</span>
          <strong>{money(revenue)}</strong>
          <input
            type="range"
            min="50"
            max="250"
            step="25"
            value={revenue}
            onChange={(event) => setRevenue(Number(event.target.value))}
          />
        </label>
      </div>
      <div className="impactGrid">
        <div>
          <span>Contacted customers</span>
          <strong>{impact.contacted}</strong>
        </div>
        <div>
          <span>Targeted conversion</span>
          <strong>{pct(impact.targeted)}</strong>
        </div>
        <div>
          <span>Conversion lift</span>
          <strong>{impact.lift.toFixed(2)}x</strong>
        </div>
        <div>
          <span>Incremental revenue</span>
          <strong>{money(impact.incrementalRevenue)}</strong>
        </div>
      </div>
    </section>
  );
}

function RocChart() {
  return (
    <section className="panel chartPanel">
      <div className="panelHeader">
        <div>
          <h2>ROC Curve</h2>
          <p>Ranking quality for likely purchasers</p>
        </div>
      </div>
      <svg viewBox="0 0 300 250" className="roc">
        <line x1="26" y1="214" x2="270" y2="214" />
        <line x1="26" y1="214" x2="26" y2="24" />
        <path className="baseline" d="M 26 214 L 270 24" />
        <path className="logistic" d={logisticPath} />
        <path className="forest" d={rocPath} />
        <text x="26" y="238">Wasted outreach</text>
        <text x="36" y="22">Purchasers captured</text>
      </svg>
      <div className="legend">
        <span><i className="forestDot" />Random Forest AUC 0.939</span>
        <span><i className="logisticDot" />Logistic Regression AUC 0.903</span>
      </div>
    </section>
  );
}

function ConfusionMatrix() {
  const cells = [
    ["True non-buyers", 163, "good"],
    ["Wasted outreach", 7, "warn"],
    ["Missed purchasers", 16, "warn"],
    ["Captured purchasers", 114, "good"],
  ];
  return (
    <section className="panel matrixPanel">
      <div className="panelHeader">
        <div>
          <h2>Campaign Outcomes</h2>
          <p>Random Forest confusion matrix</p>
        </div>
      </div>
      <div className="matrix">
        {cells.map(([label, value, tone]) => (
          <div className={tone} key={label}>
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function FeatureDrivers() {
  const max = Math.max(...featureImportance.map(([, value]) => value));
  return (
    <section className="panel driversPanel">
      <div className="panelHeader">
        <div>
          <h2>Top Purchase Drivers</h2>
          <p>Feature influence from Random Forest</p>
        </div>
      </div>
      <div className="driverList">
        {featureImportance.map(([feature, value]) => (
          <div className="driver" key={feature}>
            <span>{feature}</span>
            <div>
              <i style={{ width: `${(value / max) * 100}%` }} />
            </div>
            <strong>{value.toFixed(3)}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function SegmentTable() {
  return (
    <section className="panel segmentPanel">
      <div className="panelHeader">
        <div>
          <h2>Segment Strategy</h2>
          <p>Recommended action by propensity band</p>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Band</th>
            <th>Avg score</th>
            <th>Intent</th>
            <th>Est. revenue</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody>
          {segmentRows.map((row) => (
            <tr key={row[0]}>
              {row.map((cell) => (
                <td key={cell}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function App() {
  return (
    <main className="app">
      <Sidebar />
      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>Campaign Targeting Dashboard</h1>
            <p>Prioritize customers by purchase propensity and forecast conversion lift.</p>
          </div>
          <button>Export summary</button>
        </header>

        <section className="metricsGrid">
          <MetricCard label="ROC-AUC" value="0.939" sub="Random Forest holdout" />
          <MetricCard label="Precision" value="0.942" sub="Less wasted outreach" />
          <MetricCard label="Recall" value="0.877" sub="Purchasers captured" />
          <MetricCard label="Conversion lift" value="2.27x" sub="Top 20% segment" tone="coral" />
          <MetricCard label="Incremental revenue" value="$3,300" sub="$100/order scenario" tone="coral" />
        </section>

        <section className="dashboardGrid">
          <ModelComparison />
          <ImpactSimulator />
          <RocChart />
          <ConfusionMatrix />
          <FeatureDrivers />
          <SegmentTable />
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);

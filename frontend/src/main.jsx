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
const navItems = [
  ["Overview", "overview"],
  ["Model", "model"],
  ["Campaign", "campaign"],
  ["Drivers", "drivers"],
  ["Segments", "segments"],
];

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

function calculateImpact(outreachRate, revenue) {
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
}

function buildExecutiveSummary(outreachRate, revenue, impact) {
  const modelRows = modelMetrics
    .map(
      (model) =>
        `| ${model.name} | ${model.accuracy.toFixed(3)} | ${model.precision.toFixed(3)} | ${model.recall.toFixed(3)} | ${model.auc.toFixed(3)} | ${model.cv} |`
    )
    .join("\n");

  const driverRows = featureImportance
    .map(([feature, value]) => `| ${feature} | ${value.toFixed(4)} |`)
    .join("\n");

  const segmentSummary = segmentRows
    .map(([band, score, intent, revenueEstimate, decision]) => `- ${band}: ${intent} intent, average score ${score}, ${decision}, estimated revenue ${revenueEstimate}.`)
    .join("\n");

  return `# Customer Purchase Prediction Executive Summary

## Recommended Model

Random Forest is the deployment candidate because it offers the strongest holdout performance and cross-validated ranking stability.

| Model | Accuracy | Precision | Recall | ROC-AUC | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
${modelRows}

## Campaign Impact Scenario

- Outreach rate: ${outreachRate}%
- Revenue per purchase: ${money(revenue)}
- Contacted customers in holdout simulation: ${impact.contacted}
- Targeted conversion rate: ${pct(impact.targeted)}
- Conversion lift versus baseline: ${impact.lift.toFixed(2)}x
- Estimated incremental revenue: ${money(impact.incrementalRevenue)}

## Top Purchase Drivers

| Feature | Relative Importance |
|---|---:|
${driverRows}

## Segment Strategy

${segmentSummary}

## Business Recommendation

Use predicted purchase propensity as a campaign prioritization layer. When budget is constrained, favor the highest-score customer bands to protect precision and reduce wasted outreach. When the business goal is reach or awareness, lower the threshold and monitor recall, conversion lift, and incremental revenue by segment.
`;
}

function downloadSummary(outreachRate, revenue, impact) {
  const summary = buildExecutiveSummary(outreachRate, revenue, impact);
  const blob = new Blob([summary], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = "customer-purchase-summary.md";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function Sidebar({ activeSection, onNavigate }) {
  return (
    <header className="dashboardHeader">
      <div className="brand">
        <span className="brandMark">CP</span>
        <div>
          <strong>Customer Purchase Prediction</strong>
          <span>Executive campaign intelligence</span>
        </div>
      </div>
      <nav className="nav">
        {navItems.map(([label, targetId]) => (
          <button
            aria-current={activeSection === targetId ? "page" : undefined}
            className={activeSection === targetId ? "active" : ""}
            key={targetId}
            onClick={() => onNavigate(targetId)}
            type="button"
          >
            {label}
          </button>
        ))}
      </nav>
      <div className="modelBadge">
        <span>Deployment candidate</span>
        <strong>Random Forest</strong>
      </div>
    </header>
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

function panelClass(baseClass, isActive) {
  return `${baseClass}${isActive ? " activePanel" : ""}`;
}

function ModelComparison({ isActive }) {
  return (
    <section className={panelClass("panel modelPanel", isActive)} id="model">
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

function ImpactSimulator({
  impact,
  isActive,
  outreachRate,
  revenue,
  setOutreachRate,
  setRevenue,
}) {
  return (
    <section className={panelClass("panel impactPanel", isActive)} id="campaign">
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

function FeatureDrivers({ isActive }) {
  const max = Math.max(...featureImportance.map(([, value]) => value));
  return (
    <section className={panelClass("panel driversPanel", isActive)} id="drivers">
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

function SegmentTable({ isActive }) {
  return (
    <section className={panelClass("panel segmentPanel", isActive)} id="segments">
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
  const [activeSection, setActiveSection] = useState("overview");
  const [outreachRate, setOutreachRate] = useState(20);
  const [revenue, setRevenue] = useState(100);
  const impact = useMemo(
    () => calculateImpact(outreachRate, revenue),
    [outreachRate, revenue]
  );

  function handleNavigate(targetId) {
    const target = document.getElementById(targetId);
    if (!target) return;

    setActiveSection(targetId);
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  return (
    <main className="app">
      <section className="dashboardShell" aria-label="Customer purchase prediction dashboard">
        <Sidebar activeSection={activeSection} onNavigate={handleNavigate} />
        <header
          className={`topbar${activeSection === "overview" ? " activeSection" : ""}`}
          id="overview"
        >
          <div>
            <h1>Campaign Targeting Dashboard</h1>
            <p>Prioritize customers by purchase propensity and forecast conversion lift.</p>
          </div>
          <button
            onClick={() => downloadSummary(outreachRate, revenue, impact)}
            type="button"
          >
            Export summary
          </button>
        </header>

        <section className="metricsGrid">
          <MetricCard label="ROC-AUC" value="0.939" sub="Random Forest holdout" />
          <MetricCard label="Precision" value="0.942" sub="Less wasted outreach" />
          <MetricCard label="Recall" value="0.877" sub="Purchasers captured" />
          <MetricCard label="Conversion lift" value="2.27x" sub="Top 20% segment" tone="coral" />
          <MetricCard label="Incremental revenue" value="$3,300" sub="$100/order scenario" tone="coral" />
        </section>

        <section className="dashboardGrid">
          <ModelComparison isActive={activeSection === "model"} />
          <ImpactSimulator
            impact={impact}
            isActive={activeSection === "campaign"}
            outreachRate={outreachRate}
            revenue={revenue}
            setOutreachRate={setOutreachRate}
            setRevenue={setRevenue}
          />
          <RocChart />
          <ConfusionMatrix />
          <FeatureDrivers isActive={activeSection === "drivers"} />
          <SegmentTable isActive={activeSection === "segments"} />
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);

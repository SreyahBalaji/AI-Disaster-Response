
import { useEffect, useMemo, useState } from "react";
import "./AIInsights.css";
import FederatedResearch from "./FederatedResearch";
const API = "http://127.0.0.1:8000";

const RESOURCE_NAMES = {
  water: "Water",
  food: "Food",
  rescue: "Rescue",
  shelter: "Shelter",
  medicine: "Medical",
  medical: "Medical",
};

const RESOURCE_COLORS = [
  "#46c9ff",
  "#2ce6c4",
  "#a58bff",
  "#ffb86b",
  "#ff7595",
];

function numeric(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function PanelTitle({ eyebrow, title, right }) {
  return (
    <div className="mi-panel-header">
      <div>
        <span className="mi-eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
      </div>
      {right && <span className="mi-pill">{right}</span>}
    </div>
  );
}

function Metric({ label, value, description }) {
  return (
    <div className="mi-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {description && <small>{description}</small>}
    </div>
  );
}

function GATPanel({ data }) {
  const [selectedWard, setSelectedWard] = useState(
    data.wards?.[0]?.ward_no ?? 1
  );

  const ward =
    data.wards?.find(
      (item) => item.ward_no === Number(selectedWard)
    ) || data.wards?.[0];

  const features = Object.entries(ward?.features || {});
  const maximum = Math.max(
    0.01,
    ...features.map(([, value]) => Math.abs(Number(value)))
  );

  return (
    <section className="mi-panel" id="gat-insights">
      <PanelTitle
        eyebrow="GRAPH INTELLIGENCE"
        title="GAT Feature Explorer"
        right={`${data.ward_count} wards`}
      />

      <div className="mi-metric-row">
        <Metric label="Ward nodes" value={data.ward_count} />
        <Metric
          label="Embedding dimensions"
          value={data.feature_count}
        />
      </div>

      <div className="mi-select-row">
        <label htmlFor="mi-ward">Explore a ward</label>
        <select
          id="mi-ward"
          value={selectedWard}
          onChange={(event) =>
            setSelectedWard(Number(event.target.value))
          }
        >
          {(data.wards || []).map((item) => (
            <option key={item.ward_no} value={item.ward_no}>
              Ward {item.ward_no}
            </option>
          ))}
        </select>
      </div>

      <div className="mi-feature-chart">
        <div className="mi-chart-caption">
          <span>Negative</span>
          <span>0</span>
          <span>Positive</span>
        </div>

        {features.map(([key, value], index) => {
          const n = Number(value);
          const percent =
            (Math.abs(n) / maximum) * 50;

          return (
            <div className="mi-feature-row" key={key}>
              <strong>F{index + 1}</strong>

              <div
                className="mi-feature-track"
                title={`Feature ${index + 1}: ${n}`}
              >
                <span className="mi-zero-axis" />

                <span
                  className={
                    n >= 0
                      ? "mi-feature-fill positive"
                      : "mi-feature-fill negative"
                  }
                  style={{
                    width: `${percent}%`,
                    left:
                      n >= 0
                        ? "50%"
                        : `${50 - percent}%`,
                  }}
                />
              </div>

              <output>{numeric(n, 3)}</output>
            </div>
          );
        })}
      </div>

      <div className="mi-inline-note">
        Learned graph embedding values — not flood
        probabilities or attention coefficients.
      </div>
    </section>
  );
}

function NLPPanel({ data }) {
  const [filter, setFilter] = useState("all");

  const mentions = Object.entries(
    data.resource_mentions || {}
  ).sort((a, b) => Number(b[1]) - Number(a[1]));

  const maxMentions = Math.max(
    1,
    ...mentions.map(([, count]) => Number(count))
  );

  const reports = (data.reports || []).filter(
    (report) =>
      filter === "all" ||
      (report.requested_resources || []).includes(filter)
  );

  const activeWards = Object.entries(
    data.ward_report_counts || {}
  ).sort((a, b) => Number(a[0]) - Number(b[0]));

  return (
    <section className="mi-panel" id="nlp-insights">
      <PanelTitle
        eyebrow="INCIDENT INTELLIGENCE"
        title="Disaster Report Analytics"
        right={`${data.report_count} reports`}
      />

      <div className="mi-metric-row">
        <Metric
          label="Reports processed"
          value={data.report_count}
        />
        <Metric
          label="Wards mentioned"
          value={activeWards.length}
        />
      </div>

      <h3 className="mi-subheading">Resource mentions</h3>

      <div className="mi-resource-chart">
        {mentions.map(([resource, count], index) => (
          <div className="mi-resource-row" key={resource}>
            <span>
              {RESOURCE_NAMES[resource] || resource}
            </span>

            <div className="mi-resource-track">
              <div
                className="mi-resource-fill"
                style={{
                  width: `${
                    (Number(count) / maxMentions) * 100
                  }%`,
                  background:
                    RESOURCE_COLORS[
                      index % RESOURCE_COLORS.length
                    ],
                }}
              />
            </div>

            <strong>{count}</strong>
          </div>
        ))}
      </div>

      <h3 className="mi-subheading">Incident locations</h3>

      <div className="mi-ward-chips">
        {activeWards.map(([ward, count]) => (
          <div className="mi-ward-chip" key={ward}>
            <span>W{ward}</span>
            <strong>{count}</strong>
          </div>
        ))}
      </div>

      <div className="mi-report-title">
        <h3 className="mi-subheading">Incident feed</h3>

        <select
          aria-label="Filter reports by resource"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        >
          <option value="all">All resources</option>
          {mentions.map(([resource]) => (
            <option key={resource} value={resource}>
              {RESOURCE_NAMES[resource] || resource}
            </option>
          ))}
        </select>
      </div>

      <div className="mi-reports">
        {reports.map((report) => (
          <article
            className="mi-report-card"
            key={report.report_id}
          >
            <div className="mi-report-top">
              <strong>{report.report_id}</strong>
              <span>
                Ward {report.detected_ward_no ?? "—"}
              </span>
            </div>

            <p>{report.text}</p>

            <div className="mi-report-tags">
              {(report.requested_resources || []).map(
                (resource) => (
                  <span key={resource}>
                    {RESOURCE_NAMES[resource] || resource}
                  </span>
                )
              )}
            </div>
          </article>
        ))}

        {reports.length === 0 && (
          <p className="mi-empty">
            No reports match this filter.
          </p>
        )}
      </div>
    </section>
  );
}

function MSEComparison({ mse }) {
  const values = [
    {
      name: "Untrained",
      value: Number(mse.untrained),
      color: "#ffb86b",
    },
    {
      name: "Federated",
      value: Number(mse.federated),
      color: "#2ce6c4",
    },
    {
      name: "Centralized",
      value: Number(mse.centralized),
      color: "#59aaff",
    },
  ].filter(
    (item) =>
      Number.isFinite(item.value) &&
      item.value > 0
  );

  if (!values.length) {
    return <p className="mi-empty">No MSE results found.</p>;
  }

  // Logarithmic scale is necessary because the
  // untrained error is much larger than the others.
  const minExponent = Math.floor(
    Math.log10(Math.min(...values.map((v) => v.value)))
  );

  const maxExponent = Math.max(
    minExponent + 1,
    Math.ceil(
      Math.log10(Math.max(...values.map((v) => v.value)))
    )
  );

  const span = maxExponent - minExponent;

  const ticks = Array.from(
    { length: span + 1 },
    (_, index) => minExponent + index
  );

  function position(value) {
    return (
      ((Math.log10(value) - minExponent) / span) * 100
    );
  }

  return (
    <div className="mi-mse-chart">
      <div className="mi-chart-help">
        Test MSE · logarithmic scale · lower is better
      </div>

      {values.map((item) => (
        <div className="mi-mse-row" key={item.name}>
          <span>{item.name}</span>

          <div className="mi-mse-track">
            {ticks.map((tick) => (
              <span
                className="mi-mse-gridline"
                key={tick}
                style={{
                  left: `${
                    ((tick - minExponent) / span) * 100
                  }%`,
                }}
              />
            ))}

            <span
              className="mi-mse-point"
              style={{
                left: `${position(item.value)}%`,
                background: item.color,
                boxShadow: `0 0 14px ${item.color}77`,
              }}
            />
          </div>

          <strong>{numeric(item.value, 6)}</strong>
        </div>
      ))}

      <div className="mi-mse-axis">
        {ticks.map((tick) => (
          <span key={tick}>10^{tick}</span>
        ))}
      </div>
    </div>
  );
}

function RewardChart({ history }) {
  const points = [...(history || [])]
    .map((item, index) => ({
      round: Number(item.round ?? index + 1),
      reward: Number(item.mean_team_reward),
    }))
    .filter(
      (item) =>
        Number.isFinite(item.round) &&
        Number.isFinite(item.reward)
    )
    .sort((a, b) => a.round - b.round);

  if (points.length === 0) {
    return (
      <p className="mi-empty">
        No federated PPO training history found.
      </p>
    );
  }

  const width = 680;
  const height = 245;
  const left = 54;
  const right = 24;
  const top = 25;
  const bottom = 45;

  const rewards = points.map((item) => item.reward);
  const minReward = Math.min(...rewards);
  const maxReward = Math.max(...rewards);
  const padding = Math.max(
    2,
    (maxReward - minReward) * 0.2
  );

  const lower = minReward - padding;
  const upper = maxReward + padding;

  const chartWidth = width - left - right;
  const chartHeight = height - top - bottom;

  function x(index) {
    return (
      left +
      (points.length === 1
        ? chartWidth / 2
        : (index / (points.length - 1)) *
          chartWidth)
    );
  }

  function y(reward) {
    return (
      top +
      ((upper - reward) / (upper - lower)) *
        chartHeight
    );
  }

  const line = points
    .map(
      (item, index) =>
        `${index === 0 ? "M" : "L"} ${x(index)} ${y(
          item.reward
        )}`
    )
    .join(" ");

  const area = `${line} L ${x(
    points.length - 1
  )} ${height - bottom} L ${x(0)} ${
    height - bottom
  } Z`;

  const yTicks = Array.from(
    { length: 4 },
    (_, i) => lower + ((upper - lower) * i) / 3
  );

  return (
    <div className="mi-reward-chart">
      <div className="mi-chart-help">
        Mean team reward by federated PPO round
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Federated PPO mean team reward across training rounds"
      >
        <defs>
          <linearGradient
            id="miRewardArea"
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop
              offset="0%"
              stopColor="#34caff"
              stopOpacity="0.36"
            />
            <stop
              offset="100%"
              stopColor="#34caff"
              stopOpacity="0.01"
            />
          </linearGradient>
        </defs>

        {yTicks.map((tick) => (
          <g key={tick}>
            <line
              x1={left}
              x2={width - right}
              y1={y(tick)}
              y2={y(tick)}
              stroke="#345777"
              strokeDasharray="5 7"
            />
            <text
              x={left - 12}
              y={y(tick) + 4}
              textAnchor="end"
              fill="#ffffff"
              fontSize="12"
            >
              {tick.toFixed(1)}
            </text>
          </g>
        ))}

        <path d={area} fill="url(#miRewardArea)" />

        <path
          d={line}
          fill="none"
          stroke="#43d6ff"
          strokeWidth="3.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {points.map((point, index) => (
          <g key={`${point.round}-${index}`}>
            <circle
              cx={x(index)}
              cy={y(point.reward)}
              r="6"
              fill="#061c35"
              stroke="#54ddff"
              strokeWidth="3"
            >
              <title>
                Round {point.round}:{" "}
                {numeric(point.reward, 2)}
              </title>
            </circle>

            <text
              x={x(index)}
              y={height - 16}
              fill="#ffffff"
              fontSize="12"
              textAnchor="middle"
            >
              R{point.round}
            </text>
          </g>
        ))}
      </svg>

      <div className="mi-reward-values">
        {points.map((item) => (
          <div key={item.round}>
            <span>Round {item.round}</span>
            <strong>{numeric(item.reward, 2)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function FederatedPanel({ data }) {
  const evaluation = data.evaluation || {};
  const mse = evaluation.test_mse || {};
  const history = data.training_history || [];

  return (
    <section
      className="mi-panel mi-federated"
      id="federated-insights"
    >
      <PanelTitle
        eyebrow="DECENTRALIZED INTELLIGENCE"
        title="Federated Learning Performance"
        right={evaluation.aggregation_method || "FedAvg"}
      />

      <div className="mi-federated-stats">
        <Metric
          label="Participating centres"
          value={
            evaluation.participating_centres?.length ?? "—"
          }
        />

        <Metric
          label="Training rounds"
          value={evaluation.training_rounds ?? "—"}
        />

        <Metric
          label="Federated test MSE"
          value={numeric(mse.federated, 6)}
        />
      </div>

      <div className="mi-federated-grid">
        <div className="mi-chart-panel">
          <h3>Model Error Comparison</h3>
          <MSEComparison mse={mse} />
        </div>

        <div className="mi-chart-panel">
          <h3>Training Reward</h3>
          <RewardChart history={history} />
        </div>
      </div>

      <div className="mi-inline-note">
        Test MSE and federated PPO rewards come from
        separate experiments.
      </div>
    </section>
  );
}

export default function AIInsights() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadData() {
      try {
        const paths = [
          "/api/ai-insights/gat",
          "/api/ai-insights/nlp",
          "/api/ai-insights/federated",
        ];

        const responses = await Promise.all(
          paths.map((path) =>
            fetch(`${API}${path}`, {
              signal: controller.signal,
            })
          )
        );

        for (const response of responses) {
          if (!response.ok) {
            throw new Error(
              `API returned ${response.status}`
            );
          }
        }

        const [gat, nlp, federated] = await Promise.all(
          responses.map((response) => response.json())
        );

        if (!controller.signal.aborted) {
          setData({ gat, nlp, federated });
          setError("");
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    loadData();

    return () => controller.abort();
  }, []);

  return (
    <section className="mi-page" id="ai-insights">
      {loading && (
        <div className="mi-panel mi-message">
          Loading model analytics...
        </div>
      )}

      {error && (
        <div className="mi-panel mi-error" role="alert">
          Unable to load analytics: {error}
        </div>
      )}

{data && (
  <>
    <div className="mi-main-grid">
      <GATPanel data={data.gat} />
      <NLPPanel data={data.nlp} />
    </div>

    <FederatedPanel data={data.federated} />

    <FederatedResearch />
  </>
)}
    </section>
  );
}

import "./VisualAnalytics.css";

import { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const RESOURCE_LABELS = {
  water_litres: "Water",
  food_meals: "Food",
  rescue_cases: "Rescue",
  shelter_households: "Shelter",
  medical_cases: "Medical",
};

const RESOURCE_UNITS = {
  water_litres: "L",
  food_meals: "meals",
  rescue_cases: "cases",
  shelter_households: "households",
  medical_cases: "cases",
};

const formatNumber = (value) =>
  Number(value || 0).toLocaleString("en-IN");

const formatPercent = (value) =>
  value === null || value === undefined
    ? "N/A"
    : `${Number(value).toFixed(2)}%`;

function ProgressBar({ value, color = "blue" }) {
  const percent = Math.max(
    0,
    Math.min(100, Number(value) || 0)
  );

  return (
    <div className="progress-track">
      <div
        className={`progress-fill ${color}`}
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}

function MetricCard({ label, value, description }) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <span className="metric-description">{description}</span>
    </div>
  );
}

function ResourceComparison({ ppo, baseline }) {
  return <div className="viz-comparison" aria-label="Resource satisfaction comparison">
    <div className="viz-legend"><span><i className="viz-dot viz-ppo"/> Trained PPO</span><span><i className="viz-dot viz-baseline"/> Fairness baseline</span></div>
    {baseline.resources.map(resource => {
      const ppoItem = ppo.resources.find(r => r.resource === resource.resource);
      const ppoPct = Math.max(0,Math.min(100,Number(ppoItem?.satisfaction_percent)||0));
      const basePct = Math.max(0,Math.min(100,Number(resource.satisfaction_percent)||0));
      return <div className="viz-comparison-row" key={resource.resource}>
        <div className="viz-row-title"><strong>{RESOURCE_LABELS[resource.resource]}</strong><span>Demand: {formatNumber(resource.demand)} {RESOURCE_UNITS[resource.resource]}</span></div>
        <div className="viz-pair">
          <div className="viz-bar-line"><span>PPO</span><div className="viz-track"><div className="viz-fill viz-fill-ppo" style={{width:`${ppoPct}%`}}/></div><strong>{ppoPct.toFixed(1)}%</strong></div>
          <div className="viz-bar-line"><span>Baseline</span><div className="viz-track"><div className="viz-fill viz-fill-baseline" style={{width:`${basePct}%`}}/></div><strong>{basePct.toFixed(1)}%</strong></div>
        </div><div className="viz-quantities">Delivered: PPO {formatNumber(ppoItem?.delivered)} · Baseline {formatNumber(resource.delivered)} {RESOURCE_UNITS[resource.resource]}</div>
      </div>;
    })}
  </div>;
}

function WardAnalysis({ result, resource }) {
  return <div className="viz-wards">{result.wards.map(ward => {
    const item = ward.resources.find(entry => entry.resource === resource);
    const pct = Math.max(0,Math.min(100,Number(item?.satisfaction_percent)||0));
    return <div className="viz-ward" key={ward.ward_no}>
      <div className="viz-ward-label"><strong>Ward {ward.ward_no}</strong><span>Vulnerability weight {Number(ward.vulnerability_weight).toFixed(2)}</span></div>
      <div className="viz-ward-bar"><div className="viz-track"><div className="viz-fill viz-fill-baseline" style={{width:`${pct}%`}}/></div><strong>{pct.toFixed(1)}%</strong></div>
      <div className="viz-ward-detail">{formatNumber(item?.delivered)} / {formatNumber(item?.demand)} {RESOURCE_UNITS[resource]} · {formatNumber(item?.unmet)} unmet</div>
    </div>;
  })}</div>;
}

export default function AllocationDashboard() {
  const [dynamic, setDynamic] = useState(true);
  const [algorithm, setAlgorithm] = useState("baseline");
  const [selectedResource, setSelectedResource] =
    useState("water_litres");

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();

    async function loadComparison() {
      setLoading(true);
      setError("");

      try {
        const response = await fetch(
          `${API_BASE}/api/multi-resource/compare?dynamic=${dynamic}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(
            `Multi-resource API returned ${response.status}`
          );
        }

        const result = await response.json();

        if (!result.ppo || !result.baseline) {
          throw new Error("Unexpected API response format");
        }

        setData(result);
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message);
          setData(null);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    loadComparison();

    return () => controller.abort();
  }, [dynamic, refreshKey]);

  const activeResult = data?.[algorithm];

  const totalDemand = useMemo(() => {
    if (!activeResult) return 0;

    return activeResult.resources.reduce(
      (sum, resource) => sum + resource.demand,
      0
    );
  }, [activeResult]);

  const totalDelivered = useMemo(() => {
    if (!activeResult) return 0;

    return activeResult.resources.reduce(
      (sum, resource) => sum + resource.delivered,
      0
    );
  }, [activeResult]);

  const water = activeResult?.resources.find(
    (resource) => resource.resource === "water_litres"
  );

  return (
    <section className="allocation-section" id="allocation">
      <div className="section-heading">
        <div>
          <span className="eyebrow">COOPERATIVE ALLOCATION</span>
          <h2>Resource allocation intelligence</h2>
          <p>
            Compare the trained multi-resource PPO policy with
            a fairness-aware heuristic baseline.
          </p>
        </div>
      </div>

      <div className="panel control-panel">
        <div className="control-group">
          <label htmlFor="scenario-select">
            Disaster scenario
          </label>

          <select
            id="scenario-select"
            value={dynamic ? "dynamic" : "static"}
            onChange={(event) =>
              setDynamic(event.target.value === "dynamic")
            }
          >
            <option value="dynamic">
              Dynamic — Ward 9 demand surge
            </option>
            <option value="static">
              Static — Initial demand
            </option>
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="algorithm-select">
            Allocation strategy
          </label>

          <select
            id="algorithm-select"
            value={algorithm}
            onChange={(event) =>
              setAlgorithm(event.target.value)
            }
          >
            <option value="baseline">
              Fairness-aware baseline
            </option>
            <option value="ppo">
              Masked cooperative PPO
            </option>
          </select>
        </div>

        <button
          className="refresh-button"
          onClick={() => setRefreshKey((value) => value + 1)}
          disabled={loading}
        >
          {loading ? "Running..." : "Rerun simulation"}
        </button>
      </div>

      {loading && (
        <div className="empty-state">
          Running resource allocation simulations...
        </div>
      )}

      {error && (
        <div className="error-state">
          <strong>Unable to load allocation results.</strong>
          <p>{error}</p>
          <p>
            Check that FastAPI is running on port 8000.
          </p>
        </div>
      )}

      {activeResult && !loading && (
        <>
          <div className="metrics-grid">
            <MetricCard
              label="Water satisfaction"
              value={formatPercent(
                water?.satisfaction_percent
              )}
              description="Share of water demand fulfilled"
            />

            <MetricCard
              label="Fairness gap"
              value={Number(
                activeResult.fairness_gap
              ).toFixed(4)}
              description="Lower indicates more equal satisfaction"
            />

            <MetricCard
              label="Unmet vulnerability"
              value={Number(
                activeResult.vulnerability_weighted_unmet_ratio
              ).toFixed(4)}
              description="Weighted unmet-demand ratio"
            />

            <MetricCard
              label="Simulation steps"
              value={activeResult.simulation_steps}
              description={
                `${activeResult.successful_actions} ` +
                "successful allocation actions"
              }
            />
          </div>

          <div className="panel">
            <div className="section-heading">
              <div>
                <span className="eyebrow">
                  MODEL PERFORMANCE
                </span>
                <h3>PPO vs fairness baseline</h3>
                <p>
                  Both strategies are evaluated on the same
                  simulated disaster scenario.
                </p>
              </div>
            </div>

            <ResourceComparison
              ppo={data.ppo}
              baseline={data.baseline}
            />

            <div className="comparison-footer">
              <span>
                PPO fairness gap:
                <strong>
                  {" "}{Number(data.ppo.fairness_gap).toFixed(4)}
                </strong>
              </span>

              <span>
                Baseline fairness gap:
                <strong>
                  {" "}{Number(
                    data.baseline.fairness_gap
                  ).toFixed(4)}
                </strong>
              </span>
            </div>
          </div>

          <div className="panel" id="ward-analysis">
            <div className="section-heading">
              <div>
                <span className="eyebrow">
                  EQUITABLE DISTRIBUTION
                </span>
                <h3>Ward-level allocation analysis</h3>
                <p>
                  Viewing results from:
                  {" "}<strong>{activeResult.algorithm}</strong>
                </p>
              </div>

              <div className="control-group compact">
                <label htmlFor="resource-select">
                  Resource
                </label>

                <select
                  id="resource-select"
                  value={selectedResource}
                  onChange={(event) =>
                    setSelectedResource(event.target.value)
                  }
                >
                  {Object.entries(RESOURCE_LABELS).map(
                    ([key, label]) => (
                      <option key={key} value={key}>
                        {label}
                      </option>
                    )
                  )}
                </select>
              </div>
            </div>

            <WardAnalysis
              result={activeResult}
              resource={selectedResource}
            />

            <p className="helper-text">
              Vulnerability weights are illustrative research
              parameters, not official government assessments.
            </p>
          </div>

          <div className="panel">
            <div className="section-heading">
              <div>
                <span className="eyebrow">
                  ALLOCATION SUMMARY
                </span>
                <h3>Delivered and unmet resources</h3>
              </div>
            </div>

            <div className="resource-card-grid">
              {activeResult.resources.map((resource) => (
                <div
                  className="resource-summary-card"
                  key={resource.resource}
                >
                  <span className="metric-label">
                    {RESOURCE_LABELS[resource.resource]}
                  </span>

                  <strong>
                    {formatNumber(resource.delivered)}
                    <small>
                      {" "}{RESOURCE_UNITS[resource.resource]}
                    </small>
                  </strong>

                  <ProgressBar
                    value={resource.satisfaction_percent}
                    color="green"
                  />

                  <span className="resource-detail">
                    {formatNumber(resource.unmet)} unmet
                    {" · "}
                    {formatPercent(
                      resource.satisfaction_percent
                    )} satisfied
                  </span>
                </div>
              ))}
            </div>

            <p className="helper-text">
              Total across all resource categories:
              {" "}{formatNumber(totalDelivered)} delivered
              against {formatNumber(totalDemand)} demanded.
              These are mixed units and must not be interpreted
              as one physical quantity.
            </p>
          </div>

          <div className="panel" id="disaster-events">
            <div className="section-heading">
              <div>
                <span className="eyebrow">
                  ADAPTIVE RESPONSE
                </span>
                <h3>Disaster event timeline</h3>
                <p>
                  Demand changes recorded during the simulation.
                </p>
              </div>
            </div>

            {activeResult.events.length === 0 ? (
              <div className="empty-state">
                No additional disaster events in this scenario.
              </div>
            ) : (
              <div className="event-list">
                {activeResult.events.map((event, index) => (
                  <div
                    className="event-card"
                    key={`${event.step}-${index}`}
                  >
                    <div className="event-step">
                      Step {event.step}
                    </div>

                    <div>
                      <strong>
                        New disaster demand — Ward {event.ward_no}
                      </strong>

                      <p>
                        The simulation introduced additional
                        resource requirements.
                      </p>

                      <div className="event-tags">
                        {Object.entries(
                          event.additional_demand || {}
                        ).map(([resource, amount]) => (
                          <span key={resource}>
                            {RESOURCE_LABELS[resource] ||
                              resource}: +{formatNumber(amount)}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>


        </>
      )}
    </section>
  );
}

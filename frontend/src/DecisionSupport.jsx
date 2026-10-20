import "./VisualAnalytics.css";

import { useEffect, useState } from "react";
import "./DecisionSupport.css";

const API = "http://127.0.0.1:8000";

const RESOURCE_LABELS = {
  water_litres: "Drinking water",
  food_meals: "Food",
  rescue_cases: "Rescue",
  shelter_households: "Shelter",
  medical_cases: "Medical assistance",
};

const RESOURCE_UNITS = {
  water_litres: "litres",
  food_meals: "meals",
  rescue_cases: "cases",
  shelter_households: "households",
  medical_cases: "cases",
};

function formatQuantity(resource, value) {
  return `${Number(value || 0).toLocaleString()} ${
    RESOURCE_UNITS[resource] || "units"
  }`;
}

function ReportSelector({ reports, selected, onChange }) {
  return (
    <div className="ds-panel">
      <div className="ds-panel-heading">
        <div>
          <span className="ds-kicker">INPUT</span>
          <h3>Select a disaster report</h3>
          <p>
            Choose one of the saved synthetic reports
            to run the decision-support workflow.
          </p>
        </div>
        <span className="ds-count">{reports.length} reports</span>
      </div>

      <label className="ds-select-label" htmlFor="ds-report-select">
        Incident report
      </label>

      <select
        id="ds-report-select"
        className="ds-select"
        value={selected}
        onChange={(event) => onChange(event.target.value)}
      >
        {reports.map((report) => (
          <option key={report.report_id} value={report.report_id}>
            {report.report_id} — Ward {report.ward_no}
            {report.ppo_supported ? " · PPO supported" : ""}
          </option>
        ))}
      </select>

      <div className="ds-report-preview">
        {reports
          .filter((report) => report.report_id === selected)
          .map((report) => (
            <div key={report.report_id}>
              <span className="ds-report-identifier">
                {report.report_id}
              </span>
              <p>{report.text}</p>
              <div className="ds-chips">
                {(report.requested_resources || []).map(
                  (resource) => (
                    <span key={resource}>{resource}</span>
                  )
                )}
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}

function SummaryCards({ data }) {
  const summary = data.recommendation.resource_summary;

  const requested = summary.filter(
    (item) => item.requested > 0
  );

  const totalRequestedTypes = requested.length;
  const fulfilledTypes = requested.filter(
    (item) => item.unmet === 0
  ).length;

  const centreCount = new Set(
    data.recommendation.allocations.map(
      (item) => item.centre_id
    )
  ).size;

  return (
    <div className="ds-summary-grid">
      <div className="ds-summary-card">
        <span>Affected ward</span>
        <strong>{data.ward.ward_no}</strong>
        <small>{data.ward.ward_name}</small>
      </div>

      <div className="ds-summary-card">
        <span>Flood priority score</span>
        <strong>
          {Number(
            data.ward.flood_priority_score
          ).toFixed(4)}
        </strong>
        <small>Calculated research score</small>
      </div>

      <div className="ds-summary-card">
        <span>Resource types fulfilled</span>
        <strong>
          {fulfilledTypes}/{totalRequestedTypes}
        </strong>
        <small>For this isolated report</small>
      </div>

      <div className="ds-summary-card">
        <span>Centres contributing</span>
        <strong>{centreCount}</strong>
        <small>Simulated relief centres</small>
      </div>
    </div>
  );
}

function DemandTable({ data }) {
  const requested = data.recommendation.resource_summary.filter(item => item.requested > 0);
  return <div className="ds-panel">
    <div className="ds-panel-heading"><div><span className="ds-kicker">DEMAND ESTIMATION</span><h3>Requested vs allocated</h3></div></div>
    {requested.length === 0 ? <p className="ds-muted">No quantified resources in this report.</p> :
      <div className="viz-demand">{requested.map(item => {
        const pct = Math.max(0,Math.min(100,Number(item.satisfaction_percent)||0));
        return <div className="viz-demand-row" key={item.resource}>
          <div className="viz-demand-heading"><strong>{RESOURCE_LABELS[item.resource]||item.resource}</strong><span>{pct.toFixed(1)}% fulfilled</span></div>
          <div className="viz-demand-track" aria-label={`${RESOURCE_LABELS[item.resource]} ${pct.toFixed(1)} percent fulfilled`}><div className="viz-demand-allocated" style={{width:`${pct}%`}}/></div>
          <div className="viz-demand-figures"><span>Allocated <strong>{formatQuantity(item.resource,item.allocated)}</strong></span><span>Requested <strong>{formatQuantity(item.resource,item.requested)}</strong></span><span>Unmet <strong>{formatQuantity(item.resource,item.unmet)}</strong></span></div>
        </div>;
      })}</div>}
    {data.unquantified_resources?.length > 0 && <p className="ds-warning-text">Unquantified requests: {data.unquantified_resources.join(', ')}</p>}
  </div>;
}

function AllocationPanel({ data }) {
  const allocations = data.recommendation.allocations;

  return (
    <div className="ds-panel">
      <div className="ds-panel-heading">
        <div>
          <span className="ds-kicker">
            ALLOCATION RECOMMENDATION
          </span>
          <h3>Recommended relief centres</h3>
          <p>
            Graph-informed, stock-constrained
            heuristic allocation.
          </p>
        </div>

        <span className="ds-count">
          {allocations.length} allocations
        </span>
      </div>

      {allocations.length === 0 ? (
        <p className="ds-muted">
          No allocation is available for this report.
        </p>
      ) : (
        <div className="ds-allocation-list">
          {allocations.map((item, index) => (
            <article
              key={`${item.centre_id}-${item.resource}-${index}`}
              className="ds-allocation-item"
            >
              <div className="ds-allocation-top">
                <div>
                  <span className="ds-centre-icon">
                    RC
                  </span>
                  <strong>{item.centre_id}</strong>
                  <span className="ds-to">→</span>
                  <strong>Ward {item.target_ward}</strong>
                </div>

                <span className="ds-hop-tag">
                  {item.adjacency_hops} adjacency hops
                </span>
              </div>

              <div className="ds-allocation-quantity">
                <strong>
                  {formatQuantity(
                    item.resource,
                    item.quantity
                  )}
                </strong>
                <span>{item.resource_label}</span>
              </div>

              <p>{item.reason}</p>
            </article>
          ))}
        </div>
      )}

      <div className="ds-explain-note">
        <strong>How this recommendation is generated</strong>
        <p>{data.recommendation.explanation}</p>
      </div>
    </div>
  );
}

function PPOComparison({ comparison }) {
  if (!comparison.available) {
    return (
      <div className="ds-panel">
        <span className="ds-kicker">
          REINFORCEMENT LEARNING
        </span>
        <h3>PPO comparison unavailable</h3>
        <p className="ds-muted">{comparison.reason}</p>
        <p className="ds-muted">
          The graph-informed heuristic remains available
          for this ward.
        </p>
      </div>
    );
  }

  const ppo = comparison.ppo;
  const baseline = comparison.baseline;

  const rows = ppo.resources.filter(
    (item) => item.requested > 0
  );

  return (
    <div className="ds-panel">
      <div className="ds-panel-heading">
        <div>
          <span className="ds-kicker">
            MODEL EVALUATION
          </span>
          <h3>PPO vs fairness-aware baseline</h3>
          <p>
            Isolated evaluation using the saved
            trained model.
          </p>
        </div>
        <span className="ds-count">
          Ward {comparison.ward_no}
        </span>
      </div>

      <div className="ds-compare-metrics">
        <div>
          <span>Masked Cooperative PPO</span>
          <strong>{ppo.steps} steps</strong>
          <small>
            {ppo.successful_actions} successful actions
          </small>
        </div>

        <div>
          <span>Fairness-aware baseline</span>
          <strong>{baseline.steps} steps</strong>
          <small>
            {baseline.successful_actions} successful actions
          </small>
        </div>
      </div>

      <div className="ds-table-scroll">
        <table className="ds-table">
          <thead>
            <tr>
              <th>Resource</th>
              <th>Demand</th>
              <th>PPO allocated</th>
              <th>Baseline allocated</th>
            </tr>
          </thead>

          <tbody>
            {rows.map((item) => {
              const other = baseline.resources.find(
                (row) => row.resource === item.resource
              );

              return (
                <tr key={item.resource}>
                  <td>{RESOURCE_LABELS[item.resource]}</td>
                  <td>
                    {formatQuantity(
                      item.resource,
                      item.requested
                    )}
                  </td>
                  <td>
                    {formatQuantity(
                      item.resource,
                      item.allocated
                    )}
                  </td>
                  <td>
                    {formatQuantity(
                      item.resource,
                      other?.allocated ?? 0
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="ds-footnote">{comparison.note}</p>
    </div>
  );
}

function ExplanationPanel({ data }) {
  const explanation = data.explanation;

  const items = [
    ["Report interpretation", explanation.report_reason],
    ["Demand calculation", explanation.demand_reason],
    ["Ward priority", explanation.ward_reason],
    ["Allocation logic", explanation.allocation_reason],
    ["GAT integration", explanation.gat_note],
  ];

  return (
    <div className="ds-panel">
      <div className="ds-panel-heading">
        <div>
          <span className="ds-kicker">
            EXPLAINABLE DECISION SUPPORT
          </span>
          <h3>Why was this decision recommended?</h3>
          <p>
            Rule-based reasoning trace for the
            simulated allocation.
          </p>
        </div>
      </div>

      <div className="ds-reason-list">
        {items.map(([title, description], index) => (
          <div className="ds-reason" key={title}>
            <div className="ds-reason-number">
              {index + 1}
            </div>

            <div>
              <strong>{title}</strong>
              <p>{description}</p>
            </div>
          </div>
        ))}
      </div>



      <p className="ds-footnote">
        {data.disclaimer}
      </p>
    </div>
  );
}

export default function DecisionSupport() {
  const [reports, setReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState("");
  const [result, setResult] = useState(null);
  const [loadingReports, setLoadingReports] = useState(true);
  const [loadingResult, setLoadingResult] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function fetchReports() {
      try {
        const response = await fetch(
          `${API}/api/decision-support/reports`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(
            `Report API returned ${response.status}`
          );
        }

        const data = await response.json();

        if (!controller.signal.aborted) {
          setReports(data.reports || []);
          setSelectedReport(
            data.reports?.[0]?.report_id || ""
          );
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoadingReports(false);
        }
      }
    }

    fetchReports();

    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedReport) return;

    const controller = new AbortController();

    async function fetchRecommendation() {
      setLoadingResult(true);
      setResult(null);
      setError("");

      try {
        const response = await fetch(
          `${API}/api/decision-support/recommend/${encodeURIComponent(
            selectedReport
          )}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          const body = await response.json().catch(
            () => ({})
          );

          throw new Error(
            body.detail ||
              `Recommendation API returned ${response.status}`
          );
        }

        const data = await response.json();

        if (!controller.signal.aborted) {
          setResult(data);
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoadingResult(false);
        }
      }
    }

    fetchRecommendation();

    return () => controller.abort();
  }, [selectedReport]);

  return (
    <section
      className="ds-section"
      id="decision-support"
    >
      <div className="ds-section-heading">
        <span className="ds-kicker">
          END-TO-END DECISION SUPPORT
        </span>

        <h2>Explainable Resource Recommendations</h2>

        <p>
          Follow a synthetic disaster report from
          extracted information to quantified demand,
          allocation recommendations and explanations.
        </p>
      </div>

      {loadingReports && (
        <div className="ds-panel">
          Loading available disaster reports...
        </div>
      )}

      {error && (
        <div className="ds-error">
          Unable to load decision support: {error}
        </div>
      )}

      {!loadingReports && reports.length > 0 && (
        <ReportSelector
          reports={reports}
          selected={selectedReport}
          onChange={setSelectedReport}
        />
      )}

      {loadingResult && (
        <div className="ds-panel">
          Evaluating report and allocation strategies...
        </div>
      )}

      {result && !loadingResult && (
        <>
          <SummaryCards data={result} />

          <DemandTable data={result} />

          <div className="ds-two-columns">
            <AllocationPanel data={result} />

            <PPOComparison
              comparison={result.ppo_comparison}
            />
          </div>

          <ExplanationPanel data={result} />
        </>
      )}
    </section>
  );
}

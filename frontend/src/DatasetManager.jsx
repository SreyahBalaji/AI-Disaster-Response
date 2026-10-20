
import { useState } from "react";
import "./DatasetManager.css";

const API = "http://127.0.0.1:8000";
const MAX_SIZE = 1_000_000;

const LABELS = {
  water_litres: "Water (L)",
  food_meals: "Food (meals)",
  rescue_cases: "Rescue (cases)",
  shelter_households: "Shelter (households)",
  medical_cases: "Medical (cases)",
};

function format(value) {
  return Number(value ?? 0).toLocaleString("en-IN");
}

async function postJSON(endpoint, payload) {
  const response = await fetch(`${API}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const result = await response.json().catch(() => null);

  if (!response.ok) {
    const details = result?.detail;
    const message =
      typeof details === "string"
        ? details
        : Array.isArray(details)
          ? details.map((item) => item.msg).join("; ")
          : `Request failed (HTTP ${response.status})`;

    throw new Error(message);
  }

  return result;
}

function CoverageCard({ name, result, recommended }) {
  if (!result) return null;

  return (
    <div className={`dm-method ${recommended ? "dm-selected" : ""}`}>
      <div className="dm-method-head">
        <strong>{name}</strong>

        {recommended && (
          <span className="dm-selected-label">RECOMMENDED</span>
        )}
      </div>

      <div className="dm-coverage-number">
        {Number(result.minimum_coverage_percent).toFixed(1)}%
      </div>

      <span className="dm-small">
        Worst-served ward-resource coverage
      </span>

      <div className="dm-track">
        <div
          className="dm-track-fill"
          style={{
            width: `${Math.min(
              100,
              Math.max(
                0,
                Number(result.minimum_coverage_percent) || 0
              )
            )}%`,
          }}
        />
      </div>

      <div className="dm-mini-row">
        <span>Average coverage</span>
        <strong>
          {Number(result.average_coverage_percent).toFixed(1)}%
        </strong>
      </div>

      <div className="dm-mini-row">
        <span>Unmet ward-resource entries</span>
        <strong>
          {
            result.ward_results.filter(
              (item) => item.unmet > 0
            ).length
          }
        </strong>
      </div>
    </div>
  );
}

export default function DatasetManager() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [content, setContent] = useState("");
  const [validation, setValidation] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  async function handleFile(event) {
    const file = event.target.files?.[0];

    setSelectedFile(null);
    setContent("");
    setValidation(null);
    setAnalysis(null);
    setError("");

    if (!file) return;

    if (!/\.(csv|json)$/i.test(file.name)) {
      setError("Choose a .csv or .json file.");
      event.target.value = "";
      return;
    }

    if (file.size > MAX_SIZE) {
      setError(
        "File is too large. Maximum allowed size is 1 MB."
      );
      event.target.value = "";
      return;
    }

    try {
      const text = await file.text();

      if (text.length > MAX_SIZE || !text.trim()) {
        throw new Error(
          "The file is empty or exceeds the character limit."
        );
      }

      setSelectedFile(file);
      setContent(text);
    } catch (err) {
      setError(err.message);
    }
  }

  async function validate() {
    if (!selectedFile || busy) return;

    setBusy("Validating");
    setError("");
    setValidation(null);
    setAnalysis(null);

    try {
      const result = await postJSON(
        "/api/datasets/validate",
        {
          filename: selectedFile.name,
          content,
        }
      );

      setValidation(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function analyze() {
    if (!selectedFile || !validation || busy) return;

    setBusy("Analyzing");
    setError("");
    setAnalysis(null);

    try {
      const result = await postJSON(
        "/api/datasets/analyze",
        {
          filename: selectedFile.name,
          content,
        }
      );

      setAnalysis(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  const review = analysis?.review;
  const recommended = review?.recommended_method;

  const methodResult = recommended
    ? review?.comparison?.[recommended]
    : null;

  return (
    <div className="dm-page">
      <div className="dm-intro">
        <div>
          <span className="dm-kicker">
            INCOMING DATA / THREE-WARD PILOT
          </span>
          <h2>Dataset Intake & AI Analysis</h2>
          <p>
            Validate new disaster reports, run the existing
            allocation models, and review their recommendations.
          </p>
        </div>

        <span className="dm-mode">SIMULATION MODE</span>
      </div>

      <div className="dm-top-grid">
        <section className="dm-panel">
          <div className="dm-panel-top">
            <span className="dm-step">01</span>
            <h3>Upload disaster dataset</h3>
          </div>

          <label
            className="dm-dropzone"
            htmlFor="dm-dataset-file"
          >
            <span
              className="dm-upload-symbol"
              aria-hidden="true"
            >
              ↑
            </span>

            <strong>
              {selectedFile
                ? selectedFile.name
                : "Choose CSV or JSON dataset"}
            </strong>

            <span>
              {selectedFile
                ? `${format(selectedFile.size)} bytes selected`
                : "Click here to browse files · maximum 1 MB"}
            </span>

            <input
              id="dm-dataset-file"
              type="file"
              accept=".csv,.json"
              onChange={handleFile}
              disabled={Boolean(busy)}
            />
          </label>

          <p className="dm-help">
            Required columns:{" "}
            <strong>
              report_id, detected_ward_no,
              requested_resources
            </strong>.
            Provide affected_families and/or
            affected_patients to estimate quantities.
            Supported wards: 3, 5 and 9.
          </p>

          <div className="dm-actions">
            <button
              type="button"
              onClick={validate}
              disabled={!selectedFile || Boolean(busy)}
            >
              {busy === "Validating"
                ? "Validating…"
                : "Validate Dataset"}
            </button>

            <button
              type="button"
              className="dm-primary"
              onClick={analyze}
              disabled={!validation || Boolean(busy)}
            >
              {busy === "Analyzing"
                ? "Analyzing…"
                : "Run AI Analysis"}
            </button>
          </div>

          {error && (
            <p className="dm-error" role="alert">
              {error}
            </p>
          )}

          {validation && (
            <p className="dm-success" role="status">
              ✓ Dataset is valid: {validation.report_count}{" "}
              reports accepted. You can now run the analysis.
            </p>
          )}
        </section>

        <section className="dm-panel">
          <div className="dm-panel-top">
            <span className="dm-step">02</span>
            <h3>Dataset status</h3>
          </div>

          <div className="dm-status-list">
            <div>
              <span>Selected file</span>
              <strong>
                {selectedFile?.name ?? "None"}
              </strong>
            </div>

            <div>
              <span>Validation</span>
              <strong>
                {validation ? "Passed" : "Waiting"}
              </strong>
            </div>

            <div>
              <span>Reports accepted</span>
              <strong>
                {validation?.report_count ?? "—"}
              </strong>
            </div>

            <div>
              <span>Model training</span>
              <strong>Not started</strong>
            </div>

            <div>
              <span>Inventory state</span>
              <strong>Reset for each analysis</strong>
            </div>
          </div>

          <div className="dm-note">
            Uploaded data is analyzed in memory.
            This version does not save reports, retrain
            models, update live stock, or dispatch supplies.
          </div>
        </section>
      </div>

      {validation && (
        <section className="dm-panel">
          <div className="dm-panel-top">
            <span className="dm-step">03</span>
            <h3>Demand preview</h3>
          </div>

          <div className="dm-table-wrap">
            <table className="dm-table">
              <thead>
                <tr>
                  <th>Ward</th>

                  {Object.values(LABELS).map((label) => (
                    <th key={label}>{label}</th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {[3, 5, 9].map((ward) => (
                  <tr key={ward}>
                    <td>
                      <strong>Ward {ward}</strong>
                    </td>

                    {Object.keys(LABELS).map((resource) => (
                      <td key={resource}>
                        {format(
                          validation.combined_demand?.[ward]?.[
                            resource
                          ]
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {review && (
        <>
          <section className="dm-panel">
            <div className="dm-panel-top">
              <span className="dm-step">04</span>
              <h3>Allocation safety comparison</h3>
            </div>

            <div className="dm-method-grid">
              <CoverageCard
                name="Fairness baseline"
                result={review.comparison?.baseline}
                recommended={recommended === "baseline"}
              />

              <CoverageCard
                name="Cooperative PPO"
                result={review.comparison?.ppo}
                recommended={recommended === "ppo"}
              />
            </div>

            <div className="dm-recommendation">
              <span>RECOMMENDED METHOD</span>

              <strong>
                {recommended === "baseline"
                  ? "Fairness baseline"
                  : "Cooperative PPO"}
              </strong>

              <p>{review.recommendation_reason}</p>
            </div>

            {(review.warnings || []).map(
              (warning, index) => (
                <p
                  key={index}
                  className="dm-warning"
                >
                  ⚠ {warning}
                </p>
              )
            )}
          </section>

          <section className="dm-panel">
            <div className="dm-panel-top">
              <span className="dm-step">05</span>
              <h3>Recommended ward coverage</h3>
            </div>

            <div className="dm-table-wrap">
              <table className="dm-table">
                <thead>
                  <tr>
                    <th>Ward</th>
                    <th>Resource</th>
                    <th>Requested</th>
                    <th>Delivered</th>
                    <th>Unmet</th>
                    <th>Coverage</th>
                  </tr>
                </thead>

                <tbody>
                  {(methodResult?.ward_results || []).map(
                    (item) => (
                      <tr
                        key={`${item.ward_no}-${item.resource}`}
                      >
                        <td>Ward {item.ward_no}</td>

                        <td>
                          {LABELS[item.resource] ||
                            item.resource}
                        </td>

                        <td>{format(item.requested)}</td>
                        <td>{format(item.delivered)}</td>

                        <td
                          className={
                            item.unmet > 0
                              ? "dm-alert-cell"
                              : ""
                          }
                        >
                          {format(item.unmet)}
                        </td>

                        <td>
                          {Number(
                            item.coverage_percent
                          ).toFixed(1)}
                          %
                        </td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <section className="dm-panel">
            <div className="dm-panel-top">
              <span className="dm-step">06</span>
              <h3>Recommended relief-centre deliveries</h3>
            </div>

            <div className="dm-table-wrap">
              <table className="dm-table">
                <thead>
                  <tr>
                    <th>Relief centre</th>
                    <th>Destination</th>
                    <th>Resource</th>
                    <th>Quantity</th>
                  </tr>
                </thead>

                <tbody>
                  {(review.recommended_allocations || []).map(
                    (item, index) => (
                      <tr
                        key={`${item.centre_id}-${item.ward_no}-${item.resource}-${index}`}
                      >
                        <td>{item.centre_id}</td>
                        <td>Ward {item.ward_no}</td>

                        <td>
                          {LABELS[item.resource] ||
                            item.resource}
                        </td>

                        <td>{format(item.quantity)}</td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>

            <p className="dm-human">
              Human approval required · Prototype
              recommendations only · No automatic dispatch
            </p>
          </section>
        </>
      )}
    </div>
  );
}

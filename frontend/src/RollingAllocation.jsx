import { useEffect, useState } from "react";
import "./RollingAllocation.css";

const API = "http://127.0.0.1:8000";

const RESOURCES = [
  ["water", "Water", "water_litres", "L"],
  ["food", "Food", "food_meals", "meals"],
  ["rescue", "Rescue", "rescue_cases", "cases"],
  ["shelter", "Shelter", "shelter_households", "households"],
  ["medicine", "Medical", "medical_cases", "cases"],
];

const initialForm = {
  report_id: "",
  ward: "9",
  families: "10",
  patients: "",
  resources: ["water", "rescue"],
  text: "",
};

const fmt = (value) =>
  Number(value ?? 0).toLocaleString("en-IN");

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = data.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message ||
          (Array.isArray(detail)
            ? detail.map((item) => item.msg).join("; ")
            : `Request failed (${response.status})`);

    throw new Error(message);
  }

  return data;
}

function post(path, payload, approvalKey = "") {
  return api(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(approvalKey
        ? { "X-Approval-Key": approvalKey }
        : {}),
    },
    body: JSON.stringify(payload),
  });
}

function SectionHeader({ label, title, description }) {
  return (
    <div className="ra-section-heading">
      <div>
        <span className="ra-eyebrow">{label}</span>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
    </div>
  );
}

function Metric({ label, value, description }) {
  return (
    <div className="ra-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{description}</small>
    </div>
  );
}

function InventoryTable({ centreId, stock }) {
  return (
    <div className="ra-inventory-card">
      <div className="ra-inventory-title">
        <span className="ra-centre-dot" />
        <h3>{centreId}</h3>
        <small>Simulated centre</small>
      </div>

      <div className="ra-inventory-lines">
        {RESOURCES.map(([, label, key, unit]) => (
          <div className="ra-inventory-line" key={key}>
            <span>{label}</span>
            <strong>
              {fmt(stock?.[key])} <small>{unit}</small>
            </strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function ApprovalHistory({ approvals }) {
  if (!approvals?.length) {
    return (
      <p className="ra-muted">
        No simulated reservations have been approved.
      </p>
    );
  }

  return (
    <div className="ra-history">
      {[...approvals].reverse().map((approval) => (
        <div
          className="ra-history-item"
          key={approval.review_id}
        >
          <span className="ra-history-symbol">✓</span>
          <div>
            <strong>Reservation approved</strong>
            <p>
              Review {approval.review_id.slice(0, 8)} ·{" "}
              {approval.allocations?.length || 0} allocations
            </p>
            <small>
              {new Date(
                approval.approved_at
              ).toLocaleString("en-IN")}
            </small>
          </div>
          <span className="ra-history-tag">SIMULATED</span>
        </div>
      ))}
    </div>
  );
}

export default function RollingAllocation() {
  const [state, setState] = useState(null);
  const [approvalKey, setApprovalKey] = useState("");
  const [review, setReview] = useState(null);
  const [samples, setSamples] = useState([]);
  const [form, setForm] = useState(initialForm);

  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  async function refreshState() {
    const latest = await api("/api/rolling/state");
    setState(latest);
    setReview(latest.pending_review || null);
    return latest;
  }

  useEffect(() => {
    let active = true;

    async function initialize() {
      try {
        const [latest, sampleData] = await Promise.all([
          api("/api/rolling/state"),
          api("/api/pipeline/sample"),
        ]);

        if (!active) return;

        setState(latest);
        setReview(latest.pending_review || null);
        setSamples(sampleData.reports || []);
      } catch (err) {
        if (active) setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    }

    initialize();

    return () => {
      active = false;
    };
  }, []);

  async function execute(action, successMessage) {
    if (busy) return false;

    setBusy(true);
    setError("");
    setNotice("");

    try {
      await action();
      await refreshState();
      setNotice(successMessage);
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    } finally {
      setBusy(false);
    }
  }

  function toggleResource(resource) {
    setForm((current) => ({
      ...current,
      resources: current.resources.includes(resource)
        ? current.resources.filter((r) => r !== resource)
        : [...current.resources, resource],
    }));
  }

  function validForm() {
    if (!form.report_id.trim()) {
      return "Enter a report ID.";
    }

    if (state?.report_ids?.includes(form.report_id.trim())) {
      return "This report ID has already been submitted.";
    }

    if (form.resources.length === 0) {
      return "Select at least one requested resource.";
    }

    const needsFamilies = form.resources.some((resource) =>
      ["water", "food", "rescue", "shelter"].includes(resource)
    );

    if (
      needsFamilies &&
      (form.families === "" ||
        !Number.isInteger(Number(form.families)) ||
        Number(form.families) < 0)
    ) {
      return "Enter a valid number of affected families.";
    }

    if (
      form.resources.includes("medicine") &&
      form.patients === "" &&
      form.families === ""
    ) {
      return "Enter affected patients or families for medical demand.";
    }

    if (
      form.patients !== "" &&
      (!Number.isInteger(Number(form.patients)) ||
        Number(form.patients) < 0)
    ) {
      return "Affected patients must be a non-negative integer.";
    }

    return "";
  }

  async function handleReport(event) {
    event.preventDefault();

    const validationError = validForm();

    if (validationError) {
      setError(validationError);
      setNotice("");
      return;
    }

    const report = {
      report_id: form.report_id.trim(),
      detected_ward_no: Number(form.ward),
      affected_families:
        form.families === "" ? null : Number(form.families),
      affected_patients:
        form.patients === "" ? null : Number(form.patients),
      requested_resources: form.resources,
      text: form.text.trim(),
    };

    const succeeded = await execute(
      () => post("/api/rolling/reports", { reports: [report] }),
      `Report ${report.report_id} recorded. Generate a new review.`
    );

    if (succeeded) {
      setForm((current) => ({
        ...initialForm,
        ward: current.ward,
      }));
    }
  }

  const recordedIds = new Set(state?.report_ids || []);

  const nextSample = samples.find(
    (report) =>
      [3, 5, 9].includes(report.detected_ward_no) &&
      !recordedIds.has(report.report_id)
  );

  async function addSample() {
    if (!nextSample) return;

    await execute(
      () =>
        post("/api/rolling/reports", {
          reports: [nextSample],
        }),
      `Sample ${nextSample.report_id} recorded. Generate a new review.`
    );
  }

  async function generateReview() {
    await execute(
      () => post("/api/rolling/review"),
      "Allocation proposal calculated. No stock has been reserved."
    );
  }

 async function approveReview() {
  if (!review?.contains_allocations || busy) return;

  if (!approvalKey.trim()) {
    setError("Enter the approval passphrase.");
    setNotice("");
    return;
  }

  const confirmed = window.confirm(
    "Approve these SIMULATED reservations?\n\n" +
    "This will deduct quantities from simulated inventory. " +
    "No physical resources will be dispatched."
  );

  if (!confirmed) return;

  const succeeded = await execute(
    () =>
      post(
        "/api/rolling/approve",
        {
          review_id: review.review_id,
          expected_version: review.expected_version,
        },
        approvalKey
      ),
    "Simulated reservation approved and inventory updated."
  );

  if (succeeded) {
    setApprovalKey("");
  }
}
  const demandRows = [];

  if (state) {
    for (const ward of [3, 5, 9]) {
      for (const [, label, resource, unit] of RESOURCES) {
        const demand = Number(state.demand?.[ward]?.[resource] || 0);
        const reserved = Number(
          state.reserved?.[ward]?.[resource] || 0
        );

        if (demand > 0) {
          demandRows.push({
            ward,
            label,
            unit,
            demand,
            reserved,
            unmet: Math.max(0, demand - reserved),
          });
        }
      }
    }
  }

  const unmetRows = demandRows.filter(
    (row) => row.unmet > 0
  ).length;

  if (loading) {
    return (
      <div className="ra-page">
        <div className="ra-panel">
          Loading persistent allocation state...
        </div>
      </div>
    );
  }

  return (
    <div className="ra-page">
      <div className="ra-banner">
        <div>
          <span className="ra-eyebrow">
            PHASE 25 / RESOURCE OPERATIONS
          </span>
          <h2>Rolling allocation control</h2>
          <p>
            Track incoming reports, review new allocations and
            reserve resources with explicit approval.
          </p>
        </div>

        <span className="ra-simulation-tag">
          ● SIMULATION MODE
        </span>
      </div>

      {error && (
        <div className="ra-alert ra-error" role="alert">
          {error}
        </div>
      )}

      {notice && (
        <div className="ra-alert ra-success" role="status">
          {notice}
        </div>
      )}

      {!state ? (
        <div className="ra-panel">
          <p>Unable to retrieve the rolling allocation state.</p>
        </div>
      ) : (
        <>
          <div className="ra-metrics">
            <Metric
              label="Reports processed"
              value={fmt(state.report_ids?.length)}
              description="Unique incident reports"
            />

            <Metric
              label="Approved reviews"
              value={fmt(state.approval_history?.length)}
              description="Simulated reservations"
            />

            <Metric
              label="Unmet resource needs"
              value={fmt(unmetRows)}
              description="Ward-resource categories"
            />

            <Metric
              label="State version"
              value={`v${state.version}`}
              description="Persistent ledger revision"
            />
          </div>

          <div className="ra-grid">
            <section className="ra-panel">
              <SectionHeader
                label="01 / NEW INFORMATION"
                title="Record disaster report"
                description="Add new demand without removing previous approvals."
              />

              <form
                className="ra-form"
                onSubmit={handleReport}
              >
                <div className="ra-form-grid">
                  <label>
                    Report ID
                    <input
                      maxLength={80}
                      placeholder="e.g. NEW-003"
                      value={form.report_id}
                      onChange={(event) =>
                        setForm({
                          ...form,
                          report_id: event.target.value,
                        })
                      }
                      required
                    />
                  </label>

                  <label>
                    Affected ward
                    <select
                      value={form.ward}
                      onChange={(event) =>
                        setForm({
                          ...form,
                          ward: event.target.value,
                        })
                      }
                    >
                      <option value="3">Ward 3</option>
                      <option value="5">Ward 5</option>
                      <option value="9">Ward 9</option>
                    </select>
                  </label>

                  <label>
                    Affected families
                    <input
                      type="number"
                      min="0"
                      max="100000"
                      value={form.families}
                      onChange={(event) =>
                        setForm({
                          ...form,
                          families: event.target.value,
                        })
                      }
                    />
                  </label>

                  <label>
                    Affected patients
                    <input
                      type="number"
                      min="0"
                      max="100000"
                      placeholder="Optional"
                      value={form.patients}
                      onChange={(event) =>
                        setForm({
                          ...form,
                          patients: event.target.value,
                        })
                      }
                    />
                  </label>
                </div>

                <span className="ra-field-label">
                  Requested resources
                </span>

                <div className="ra-resource-options">
                  {RESOURCES.map(([key, label]) => (
                    <label
                      className="ra-resource-option"
                      key={key}
                    >
                      <input
                        type="checkbox"
                        checked={form.resources.includes(key)}
                        onChange={() => toggleResource(key)}
                      />
                      {label}
                    </label>
                  ))}
                </div>

                <label>
                  Incident notes
                  <textarea
                    rows={3}
                    maxLength={3000}
                    placeholder="Describe the new disaster report..."
                    value={form.text}
                    onChange={(event) =>
                      setForm({
                        ...form,
                        text: event.target.value,
                      })
                    }
                  />
                </label>

                <button
                  className="ra-btn ra-btn-primary"
                  disabled={busy}
                  type="submit"
                >
                  {busy ? "Processing..." : "Record new report"}
                </button>
              </form>

              <div className="ra-sample-area">
                <div>
                  <strong>Existing sample reports</strong>
                  <p>
                    {nextSample
                      ? `${nextSample.report_id} is available for the next demo.`
                      : "All available sample reports are already recorded."}
                  </p>
                </div>

                <button
                  type="button"
                  className="ra-btn ra-btn-outline"
                  disabled={busy || !nextSample}
                  onClick={addSample}
                >
                  Add next sample
                </button>
              </div>
            </section>

            <section className="ra-panel">
              <SectionHeader
                label="02 / HUMAN REVIEW"
                title="Allocation approval"
                description="A review proposes allocations but does not reserve inventory."
              />

              <div className="ra-review-status">
                <span>Current proposal</span>
                <strong>
                  {review
                    ? review.status
                        .replaceAll("_", " ")
                        .toUpperCase()
                    : "NO PENDING REVIEW"}
                </strong>
              </div>

              <button
                type="button"
                className="ra-btn ra-btn-outline"
                disabled={busy || unmetRows === 0}
                onClick={generateReview}
              >
                Calculate allocation proposal
              </button>

              {review && (
                <div className="ra-review-results">
                  <div className="ra-coverage-grid">
                    <Metric
                      label="Minimum coverage"
                      value={`${fmt(
                        review.minimum_coverage_percent
                      )}%`}
                      description="If proposal is approved"
                    />

                    <Metric
                      label="Average coverage"
                      value={`${fmt(
                        review.average_coverage_percent
                      )}%`}
                      description="If proposal is approved"
                    />
                  </div>

                  <div className="ra-review-meta">
                    <p>
                      <strong>Review:</strong>{" "}
                      {review.review_id.slice(0, 8)}
                    </p>
                    <p>
                      <strong>Method:</strong> {review.method}
                    </p>
                    <p>
                      <strong>New allocations:</strong>{" "}
                      {review.allocations?.length || 0}
                    </p>
                    <p>
                      <strong>Based on version:</strong>{" "}
                      {review.expected_version}
                    </p>
                  </div>

                  {review.contains_allocations ? (
                    <>
                      <div className="ra-table-scroll">
                        <table className="ra-table">
                          <thead>
                            <tr>
                              <th>Centre</th>
                              <th>Ward</th>
                              <th>Resource</th>
                              <th>Quantity</th>
                            </tr>
                          </thead>
                          <tbody>
                            {review.allocations.map(
                              (allocation, index) => {
                                const resource = RESOURCES.find(
                                  (item) =>
                                    item[2] === allocation.resource
                                );

                                return (
                                  <tr key={index}>
                                    <td>{allocation.centre_id}</td>
                                    <td>{allocation.ward_no}</td>
                                    <td>
                                      {resource?.[1] ||
                                        allocation.resource}
                                    </td>
                                    <td>
                                      {fmt(allocation.quantity)}{" "}
                                      {resource?.[3] || ""}
                                    </td>
                                  </tr>
                                );
                              }
                            )}
                          </tbody>
                        </table>
                      </div>

                      <label
                        className="ra-approval-key-field"
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          gap: "8px",
                          marginTop: "18px",
                          marginBottom: "12px",
                          fontWeight: 600,
                          fontSize: "13px",
                        }}
                      >
                        Approval passphrase
                        <input
                          type="password"
                          value={approvalKey}
                          onChange={(event) =>
                            setApprovalKey(event.target.value)
                          }
                          placeholder="Enter authorized passphrase"
                          autoComplete="off"
                          aria-label="Approval passphrase"
                          style={{
                            padding: "12px 14px",
                            background: "#102138",
                            border: "1px solid #415974",
                            borderRadius: "9px",
                            color: "#fff",
                            fontSize: "14px",
                            width: "100%",
                            boxSizing: "border-box",
                          }}
                        />
                      </label>

                      <button
                        type="button"
                        className="ra-btn ra-btn-approve"
                        disabled={busy || !approvalKey.trim()}
                        onClick={approveReview}
                      >
                        Approve simulated reservation
                      </button>
                    </>
                  ) : (
                    <p className="ra-muted">
                      No additional resources can be proposed
                      with the current simulated inventory.
                    </p>
                  )}
                </div>
              )}

              <p className="ra-safety-note">
                Human approval is required for each simulated
                reservation. This prototype does not perform
                physical dispatch. Approval uses a shared
                prototype passphrase, not individual user accounts.
              </p>
            </section>
          </div>

          <section className="ra-panel">
            <SectionHeader
              label="03 / INVENTORY"
              title="Remaining relief-centre resources"
              description="Saved inventory after approved simulated reservations."
            />

            <div className="ra-inventory-grid">
              {Object.entries(
                state.remaining_stock || {}
              ).map(([centreId, stock]) => (
                <InventoryTable
                  centreId={centreId}
                  stock={stock}
                  key={centreId}
                />
              ))}
            </div>
          </section>

          <div className="ra-grid">
            <section className="ra-panel">
              <SectionHeader
                label="04 / CURRENT DEMAND"
                title="Ward resource requirements"
                description="Reserved means committed in the simulation, not physically delivered."
              />

              <div className="ra-table-scroll">
                <table className="ra-table">
                  <thead>
                    <tr>
                      <th>Ward</th>
                      <th>Resource</th>
                      <th>Demand</th>
                      <th>Reserved</th>
                      <th>Unmet</th>
                    </tr>
                  </thead>

                  <tbody>
                    {demandRows.length === 0 ? (
                      <tr>
                        <td colSpan={5}>
                          No disaster demand recorded yet.
                        </td>
                      </tr>
                    ) : (
                      demandRows.map((row) => (
                        <tr
                          key={`${row.ward}-${row.label}`}
                        >
                          <td>{row.ward}</td>
                          <td>{row.label}</td>
                          <td>
                            {fmt(row.demand)} {row.unit}
                          </td>
                          <td>
                            {fmt(row.reserved)} {row.unit}
                          </td>
                          <td
                            className={
                              row.unmet > 0
                                ? "ra-unmet"
                                : "ra-met"
                            }
                          >
                            {fmt(row.unmet)} {row.unit}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="ra-panel">
              <SectionHeader
                label="05 / AUDIT HISTORY"
                title="Approved reservations"
                description="A record of approvals made within this simulation."
              />

              <ApprovalHistory
                approvals={state.approval_history}
              />

              <div className="ra-report-summary">
                <strong>Recorded reports</strong>
                <p>
                  {state.report_ids?.length
                    ? state.report_ids.join(", ")
                    : "No reports submitted"}
                </p>
              </div>
            </section>
          </div>

          <div className="ra-footer">
            <span>RESPONDA · ALUVA PILOT</span>
            <button
              type="button"
              className="ra-btn ra-btn-outline"
              disabled={busy}
              onClick={() =>
                execute(
                  refreshState,
                  "Latest simulation state loaded."
                )
              }
            >
              Refresh state
            </button>
          </div>
        </>
      )}
    </div>
  );
}

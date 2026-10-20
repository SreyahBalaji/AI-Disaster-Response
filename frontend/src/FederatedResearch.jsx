
import { useEffect, useMemo, useState } from "react";
import "./FederatedResearch.css";

const API = "http://127.0.0.1:8000";

const CENTER_COLORS = [
  "#47c9ff",
  "#38dabc",
  "#a995ff",
];

function format(value, digits = 2) {
  const number = Number(value);

  if (!Number.isFinite(number)) return "—";

  return number.toFixed(digits);
}

function MetricCard({ label, value, note }) {
  return (
    <div className="fr-metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

function Bar({
  label,
  value,
  color,
  suffix = "%",
}) {
  const safeValue = Math.max(
    0,
    Math.min(100, Number(value) || 0)
  );

  return (
    <div className="fr-bar-row">
      <div className="fr-bar-heading">
        <span>{label}</span>
        <strong>
          {format(value)}
          {suffix}
        </strong>
      </div>

      <div className="fr-bar-track">
        <div
          className="fr-bar-fill"
          style={{
            width: `${safeValue}%`,
            background: color,
          }}
        />
      </div>
    </div>
  );
}

function TrainingFlow({ participants }) {
  return (
    <div className="fr-flow">
      <div className="fr-flow-heading">
        <h3>How the centres learn together</h3>
        <p>
          Each simulated relief centre trains a local
          copy of the allocation model.
        </p>
      </div>

      <div className="fr-centres">
        {participants.map((centre, index) => (
          <div
            className="fr-centre"
            key={centre}
            style={{
              "--fr-centre-color":
                CENTER_COLORS[
                  index % CENTER_COLORS.length
                ],
            }}
          >
            <div className="fr-centre-icon">
              <svg
                width="25"
                height="25"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                aria-hidden="true"
              >
                <rect
                  x="4"
                  y="3"
                  width="16"
                  height="18"
                  rx="2"
                />
                <path d="M8 7h2M14 7h2M8 11h2M14 11h2M8 15h2M14 15h2M10 21v-4h4v4" />
              </svg>
            </div>

            <strong>{centre}</strong>
            <span>Local PPO training</span>

            <div className="fr-centre-success">
              ✓ Participated
            </div>
          </div>
        ))}
      </div>

      <div className="fr-flow-connection">
        <div className="fr-flow-line" />
        <span>Model weights aggregated</span>
        <div className="fr-flow-arrow">↓</div>
      </div>

      <div className="fr-global">
        <div className="fr-global-symbol">◎</div>
        <div>
          <strong>Global Federated GAT-PPO</strong>
          <p>
            FedAvg combines the updated model
            parameters from all three centres.
          </p>
        </div>
        <span className="fr-global-tag">
          Experimental
        </span>
      </div>

      <div className="fr-flow-explain">
        In this experiment, participants share model
        parameters for averaging instead of collecting
        all their simulated training scenarios into a
        single central training dataset.
      </div>
    </div>
  );
}

function TrainingRounds({
  history,
  selectedRound,
  setSelectedRound,
}) {
  const chosen =
    history.find(
      (item) => item.round === selectedRound
    ) || history[history.length - 1];

  const results = Object.entries(
    chosen?.participants || {}
  );

  const maximum = Math.max(
    1,
    ...results.map(([, item]) =>
      Math.max(0, Number(item.mean_reward) || 0)
    )
  );

  return (
    <div className="fr-section">
      <div className="fr-section-heading">
        <div>
          <span className="fr-eyebrow">
            TRAINING PROGRESS
          </span>
          <h3>Explore the federated rounds</h3>
          <p>
            Select a round to see the training
            reward at each relief centre.
          </p>
        </div>
      </div>

      <div
        className="fr-round-tabs"
        role="group"
        aria-label="Select training round"
      >
        {history.map((item) => (
          <button
            key={item.round}
            type="button"
            className={
              selectedRound === item.round
                ? "fr-round-tab active"
                : "fr-round-tab"
            }
            onClick={() =>
              setSelectedRound(item.round)
            }
          >
            Round {item.round}
          </button>
        ))}
      </div>

      {chosen ? (
        <div className="fr-round-details">
          <div className="fr-round-summary">
            <strong>
              Round {chosen.round}
            </strong>
            <span>
              {chosen.aggregation_verified
                ? "✓ Model aggregation verified"
                : "Aggregation status unavailable"}
            </span>
          </div>

          <div className="fr-round-bars">
            {results.map(
              ([centre, result], index) => {
                const reward = Number(
                  result.mean_reward
                );

                return (
                  <div
                    className="fr-reward-row"
                    key={centre}
                  >
                    <div className="fr-reward-name">
                      <strong>{centre}</strong>
                      <span>
                        Reward {format(reward)}
                      </span>
                    </div>

                    <div className="fr-bar-track">
                      <div
                        className="fr-bar-fill"
                        style={{
                          width: `${
                            Math.max(
                              0,
                              reward
                            ) / maximum * 100
                          }%`,
                          background:
                            CENTER_COLORS[
                              index %
                                CENTER_COLORS.length
                            ],
                        }}
                      />
                    </div>
                  </div>
                );
              }
            )}
          </div>

          <p className="fr-explanation">
            A training reward measures how the PPO
            model performed in its local simulated
            environment. It is not the percentage
            of resources successfully allocated.
            Rewards from different centres are not
            directly equivalent to fairness scores.
          </p>
        </div>
      ) : (
        <p className="fr-muted">
          No training round history available.
        </p>
      )}
    </div>
  );
}

function PerformanceComparison({ evaluation }) {
  const [metric, setMetric] = useState("average");

  const baseline = evaluation.baseline || {};
  const federated =
    evaluation.fedavg_gat_ppo || {};

  const metricKey =
    metric === "average"
      ? "mean_average_coverage"
      : "mean_minimum_coverage";

  const baselineValue = Number(
    baseline[metricKey] || 0
  );

  const federatedValue = Number(
    federated[metricKey] || 0
  );

  const difference =
    baselineValue - federatedValue;

  const zeroBaseline = Number(
    baseline.mean_zero_coverage_items || 0
  );

  const zeroFederated = Number(
    federated.mean_zero_coverage_items || 0
  );

  return (
    <div className="fr-section">
      <div className="fr-section-heading">
        <div>
          <span className="fr-eyebrow">
            ALLOCATION EFFECTIVENESS
          </span>
          <h3>Which method serves people better?</h3>
          <p>
            Comparison on the same simulated dynamic
            disaster scenarios. Higher coverage
            is better.
          </p>
        </div>
      </div>

      <div className="fr-metric-switch">
        <button
          type="button"
          className={
            metric === "average"
              ? "active"
              : ""
          }
          onClick={() => setMetric("average")}
        >
          Average coverage
        </button>

        <button
          type="button"
          className={
            metric === "minimum"
              ? "active"
              : ""
          }
          onClick={() => setMetric("minimum")}
        >
          Minimum coverage
        </button>
      </div>

      <div className="fr-comparison-chart">
        <Bar
          label="Fairness baseline"
          value={baselineValue}
          color="#37dabb"
        />

        <Bar
          label="Federated GAT-PPO"
          value={federatedValue}
          color="#47a9ff"
        />

        <div className="fr-comparison-message">
          <strong>
            {format(Math.abs(difference))} percentage
            points
          </strong>

          <span>
            {difference > 0
              ? "The fairness baseline performs better on this metric."
              : difference < 0
                ? "The federated model performs better on this metric."
                : "Both methods have equal coverage on this metric."}
          </span>
        </div>
      </div>

      <div className="fr-comparison-footer">
        <div>
          <strong>
            {format(zeroBaseline, 2)}
          </strong>
          <span>
            Average completely unserved categories
            with the fairness baseline
          </span>
        </div>

        <div>
          <strong>
            {format(zeroFederated, 2)}
          </strong>
          <span>
            Average completely unserved categories
            with federated PPO
          </span>
        </div>
      </div>

      <p className="fr-explanation">
        Average coverage measures how much demand
        is satisfied across resource categories.
        Minimum coverage measures the least-served
        category, making it especially useful for
        identifying unfair outcomes.
      </p>
    </div>
  );
}

function ResearchConclusion({
  evaluation,
  operational,
}) {
  const baseline =
    evaluation.baseline || {};

  const federated =
    evaluation.fedavg_gat_ppo || {};

  const baselineMin = Number(
    baseline.mean_minimum_coverage
  );

  const federatedMin = Number(
    federated.mean_minimum_coverage
  );

  const baselineAvg = Number(
    baseline.mean_average_coverage
  );

  const federatedAvg = Number(
    federated.mean_average_coverage
  );

  const baselinePreferred =
    baselineMin >= federatedMin &&
    baselineAvg >= federatedAvg;

  return (
    <div className="fr-conclusion">
      <span className="fr-eyebrow">
        RESEARCH INTERPRETATION
      </span>

      <h3>
        {baselinePreferred
          ? "Federated learning works, but fairness still matters"
          : "Research results require further evaluation"}
      </h3>

      <p>
        {baselinePreferred
          ? "Three simulated participants successfully trained and aggregated a global PPO model. However, the fairness-based allocation method achieved stronger coverage in this experiment."
          : "The models have different strengths across the reported evaluation metrics. Broader testing is needed before choosing an operational policy."}
      </p>

      <div className="fr-decision">
        <span>Selected allocation method</span>
        <strong>
          {operational
            ? "Federated PPO"
            : "Coverage-first fairness baseline"}
        </strong>
        <small>
          {operational
            ? "Reported as active by the backend."
            : "The federated checkpoint is research-only and does not control rolling reservations."}
        </small>
      </div>
    </div>
  );
}

export default function FederatedResearch() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedRound, setSelectedRound] =
    useState(null);

  useEffect(() => {
    const controller = new AbortController();

    async function load() {
      try {
        const response = await fetch(
          `${API}/api/federated-research/results`,
          {
            signal: controller.signal,
          }
        );

        if (!response.ok) {
          const payload = await response
            .json()
            .catch(() => ({}));

          throw new Error(
            typeof payload.detail === "string"
              ? payload.detail
              : `Research API error ${response.status}`
          );
        }

        const result = await response.json();

        if (!controller.signal.aborted) {
          setData(result);
          const rounds =
            result.experiment?.round_history || [];

          setSelectedRound(
            rounds.length
              ? rounds[rounds.length - 1].round
              : null
          );
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

    load();

    return () => controller.abort();
  }, []);

  const experiment = data?.experiment;

  const totalEpisodes = useMemo(() => {
    if (!experiment) return 0;

    return (
      (experiment.participants?.length || 0) *
      Number(experiment.rounds || 0) *
      Number(experiment.episodes_per_centre || 0)
    );
  }, [experiment]);

  return (
    <section
      className="fr-root"
      id="federated-research"
    >
      <div className="fr-header">
        <div>
          <span className="fr-eyebrow">
            FEDERATED LEARNING / PHASE 27
          </span>

          <h2>
            Collaborative AI for disaster response
          </h2>

          <p>
            Understand how three simulated relief
            centres train a shared allocation model,
            and what the results reveal.
          </p>
        </div>

        <span className="fr-research-badge">
          RESEARCH EXPERIMENT
        </span>
      </div>

      {loading && (
        <div className="fr-loading">
          Loading federated training results...
        </div>
      )}

      {error && (
        <div
          className="fr-error"
          role="alert"
        >
          Could not load saved federated results:
          {" "}{error}
        </div>
      )}

      {experiment && (
        <>
          <div className="fr-metrics">
            <MetricCard
              label="Relief centres"
              value={
                experiment.participants?.length || 0
              }
              note="Simulated participants"
            />

            <MetricCard
              label="Federated rounds"
              value={experiment.rounds}
              note="Completed aggregations"
            />

            <MetricCard
              label="Local training episodes"
              value={totalEpisodes}
              note="Across all participants"
            />

            <MetricCard
              label="Resources"
              value="5"
              note="Water, food, rescue, shelter, medical"
            />
          </div>

          <div className="fr-panel">
            <TrainingFlow
              participants={
                experiment.participants || []
              }
            />
          </div>

          <div className="fr-panel">
            <TrainingRounds
              history={
                experiment.round_history || []
              }
              selectedRound={selectedRound}
              setSelectedRound={setSelectedRound}
            />
          </div>

          <div className="fr-panel">
            <PerformanceComparison
              evaluation={
                experiment.evaluation || {}
              }
            />
          </div>

          <ResearchConclusion
            evaluation={
              experiment.evaluation || {}
            }
            operational={
              data.federated_model_operational
            }
          />

          <div className="fr-disclaimer">
            <strong>About this experiment:</strong>{" "}
            Training was simulated locally on one
            machine using different disaster scenarios.
            The GAT features provide model context,
            while PPO parameters are aggregated with
            FedAvg. This does not establish a formal
            privacy guarantee or demonstrate real-world
            deployment.
          </div>
        </>
      )}
    </section>
  );
}

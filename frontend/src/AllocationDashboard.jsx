
import { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

export default function AllocationDashboard() {
  const [dynamic, setDynamic] = useState(true);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadAllocation() {
      setLoading(true);
      setError("");

      try {
        const response = await fetch(
          `${API_BASE}/api/allocation/simulate?dynamic=${dynamic}`
        );

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        if (!cancelled) {
          setData(result);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setData(null);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadAllocation();

    return () => {
      cancelled = true;
    };
  }, [dynamic]);

  const cardStyle = {
    background: "#202938",
    padding: "20px",
    borderRadius: "12px",
    flex: "1 1 180px",
    textAlign: "center",
  };

  return (
    <section
      style={{
        marginTop: "35px",
        padding: "24px",
        background: "#151b26",
        borderRadius: "16px",
        color: "#f1f5f9",
      }}
    >
      <h2>AI Resource Allocation Dashboard</h2>

      <p style={{ color: "#aab8ca" }}>
        Simulated allocation results from the Federated Cooperative PPO API
      </p>

      <label>
        Disaster Scenario:{" "}
        <select
          value={dynamic ? "dynamic" : "static"}
          onChange={(event) =>
            setDynamic(event.target.value === "dynamic")
          }
          style={{
            padding: "10px",
            borderRadius: "8px",
            marginBottom: "20px",
          }}
        >
          <option value="dynamic">Dynamic Ward 9 Surge</option>
          <option value="static">Static Demand</option>
        </select>
      </label>

      {loading && <p>Loading allocation results...</p>}

      {error && (
        <p style={{ color: "#ff6b6b" }}>
          Unable to load allocation results: {error}
        </p>
      )}

      {data && !loading && (
        <>
          <div
            style={{
              display: "flex",
              gap: "15px",
              flexWrap: "wrap",
              marginTop: "15px",
            }}
          >
            <div style={cardStyle}>
              <h3>Total Demand</h3>
              <h2>{data.total_demand_litres} L</h2>
            </div>

            <div style={cardStyle}>
              <h3>Resources Delivered</h3>
              <h2>{data.total_delivered_litres} L</h2>
            </div>

            <div style={cardStyle}>
              <h3>Demand Satisfaction</h3>
              <h2>{data.demand_satisfaction_percent}%</h2>
            </div>

            <div style={cardStyle}>
              <h3>Fairness Gap</h3>
              <h2>{Number(data.fairness_gap).toFixed(4)}</h2>
            </div>
          </div>

          <h3 style={{ marginTop: "30px" }}>
            Ward-wise Water Allocation
          </h3>

          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                textAlign: "left",
              }}
            >
              <thead>
                <tr>
                  {[
                    "Ward",
                    "Demand (L)",
                    "Delivered (L)",
                    "Unmet (L)",
                    "Satisfaction",
                  ].map((heading) => (
                    <th
                      key={heading}
                      style={{
                        padding: "12px",
                        borderBottom: "1px solid #475569",
                      }}
                    >
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {data.ward_allocations.map((ward) => (
                  <tr key={ward.ward_no}>
                    <td style={{ padding: "12px" }}>
                      Ward {ward.ward_no}
                    </td>
                    <td>{ward.demand_litres}</td>
                    <td>{ward.delivered_litres}</td>
                    <td>{ward.unmet_litres}</td>
                    <td>{ward.satisfaction_percent}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p
            style={{
              marginTop: "20px",
              fontSize: "13px",
              color: "#aab8ca",
            }}
          >
            Model: {data.model} | Scenario: {data.scenario}
          </p>

          <p style={{ color: "#fbbf24", fontSize: "13px" }}>
            {data.note}
          </p>
        </>
      )}
    </section>
  );
}

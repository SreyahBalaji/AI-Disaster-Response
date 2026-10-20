
import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import AllocationDashboard from "./AllocationDashboard";
import AIInsights from "./AIInsights";
import GATNetwork from "./GATNetwork";
import DecisionSupport from "./DecisionSupport";
import DatasetManager from "./DatasetManager";
import RollingAllocation from "./RollingAllocation";
import "./App.css";
import "./CommandCenter.css";

const API = "http://127.0.0.1:8000";

const pages = [
  ["overview", "Overview", "grid"],
  ["gis", "GIS Intelligence", "map"],
  ["network", "GAT Network", "network"],
  ["decision", "Decision Support", "shield"],
  ["allocation", "Resource Allocation", "layers"],
  ["rolling", "Rolling Allocation", "layers"],
  ["insights", "Model Analytics", "chart"],
  ["datasets", "Data & Training", "database"],
];

const headings = {
  overview: [
    "OPERATIONS / OVERVIEW",
    "Aluva Command Centre",
    "Flood intelligence at a glance",
  ],
  rolling: [
  "LIVE RESOURCE OPERATIONS",
  "Rolling Allocation",
  "Review incoming reports, simulated inventory and approvals",
],
  gis: [
    "GEOSPATIAL INTELLIGENCE",
    "Flood Priority Map",
    "Explore all 26 municipal wards",
  ],
  network: [
    "GRAPH INTELLIGENCE",
    "Ward & Relief Network",
    "Explore connections, priorities and relief-centre inventory",
  ],
  decision: [
    "DECISION INTELLIGENCE",
    "Response Recommendations",
    "Turn incident reports into resource allocation decisions",
  ],
  allocation: [
    "RESOURCE OPERATIONS",
    "Allocation Simulation",
    "Compare cooperative PPO and the fairness baseline",
  ],
  insights: [
    "RESEARCH ANALYTICS",
    "AI Model Performance",
    "Graph embeddings, incident extraction and federated results",
  ],
  datasets: [
    "DATA OPERATIONS",
    "Data & Training",
    "Upload, validate and analyze new disaster reports",
  ],
};

function Glyph({ name, size = 20 }) {
  const paths = {
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </>
    ),
    map: (
      <>
        <path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3z" />
        <path d="M9 3v15M15 6v15" />
      </>
    ),
    network: (
      <>
        <circle cx="12" cy="4" r="2.5" />
        <circle cx="5" cy="19" r="2.5" />
        <circle cx="19" cy="19" r="2.5" />
        <path d="m11 6-5 10M13 6l5 10M7.5 19h9" />
      </>
    ),
    shield: (
      <>
        <path d="M12 2 4 6v6c0 5 3.4 8 8 10 4.6-2 8-5 8-10V6z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),
    layers: (
      <>
        <path d="m12 2 9 5-9 5-9-5z" />
        <path d="m3 12 9 5 9-5M3 17l9 5 9-5" />
      </>
    ),
    chart: (
      <>
        <path d="M4 20v-7M10 20V5M16 20v-9M21 20V8M2 20h20" />
      </>
    ),
    database: (
      <>
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5M3 12c0 1.7 4 3 9 3s9-1.3 9-3" />
      </>
    ),
    arrow: (
      <>
        <path d="M5 12h14m-6-6 6 6-6 6" />
      </>
    ),
    pulse: (
      <>
        <path d="M2 12h4l3-7 5 14 3-7h5" />
      </>
    ),
  };

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth="1.8"
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}

function FitMap({ data }) {
  const map = useMap();

  useEffect(() => {
    if (!data?.features?.length) return;

    const bounds = L.geoJSON(data).getBounds();

    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        padding: [24, 24],
        maxZoom: 14,
      });
    }
  }, [data, map]);

  return null;
}

function FloodMap({ compact = false }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${API}/api/gis/priority-map`, {
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) {
          throw Error(`Map service: ${response.status}`);
        }

        return response.json();
      })
      .then(setData)
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError(err.message);
        }
      });

    return () => controller.abort();
  }, []);

  const colors = {
    high: "#ff626d",
    moderate: "#ffb454",
    low: "#25c6a0",
  };

  return (
    <section
      className={`cc-map-frame ${compact ? "compact" : ""}`}
    >
      <div className="cc-card-top">
        <div>
          <span className="cc-label">
            GEOSPATIAL VIEW
          </span>
          <h2>Flood priority map</h2>
        </div>

        <span className="cc-tag">
          26 wards mapped
        </span>
      </div>

      <div className="cc-map-legend">
        <span>
          <i style={{ background: colors.high }} />
          High
        </span>

        <span>
          <i style={{ background: colors.moderate }} />
          Moderate
        </span>

        <span>
          <i style={{ background: colors.low }} />
          Low
        </span>
      </div>

      {error && (
        <div className="cc-error">
          {error} — start the FastAPI backend.
        </div>
      )}

      {!data && !error && (
        <div className="cc-map-placeholder">
          Loading ward boundaries…
        </div>
      )}

      {data && (
        <div className="cc-map-surface">
          <MapContainer
            center={[10.11, 76.35]}
            zoom={12}
            scrollWheelZoom={false}
            style={{
              width: "100%",
              height: "100%",
            }}
          >
            <TileLayer
              attribution="&copy; OpenStreetMap contributors"
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            <FitMap data={data} />

            <GeoJSON
              data={data}
              style={(feature) => ({
                color: "#ffffff",
                weight: 1.5,
                fillColor:
                  colors[
                    String(
                      feature.properties?.priority_category || ""
                    ).toLowerCase()
                  ] || "#718aaa",
                fillOpacity: 0.79,
              })}
              onEachFeature={(feature, layer) => {
                const properties = feature.properties || {};

                const div = document.createElement("div");
                const name = document.createElement("strong");
                const detail = document.createElement("p");

                name.textContent =
                  properties.ward_name ||
                  `Ward ${properties.ward_no}`;

                detail.style.margin = "5px 0 0";
                detail.textContent =
                  `Ward ${properties.ward_no} · Priority ` +
                  Number(
                    properties.flood_priority_score || 0
                  ).toFixed(4);

                div.append(name, detail);
                layer.bindPopup(div);
              }}
            />
          </MapContainer>
        </div>
      )}
    </section>
  );
}

function Stat({
  icon,
  label,
  value,
  sub,
  tone = "blue",
}) {
  return (
    <div className="cc-stat">
      <div className={`cc-stat-icon ${tone}`}>
        <Glyph name={icon} size={21} />
      </div>

      <div className="cc-stat-text">
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{sub}</small>
      </div>
    </div>
  );
}

function Overview({ go }) {
  const [metrics, setMetrics] = useState({
    wards: "—",
    centres: "—",
    edges: "—",
    reports: "—",
  });

  const [reports, setReports] = useState([]);

  useEffect(() => {
    const controller = new AbortController();

    Promise.allSettled([
      fetch(`${API}/api/ai-insights/graph`, {
        signal: controller.signal,
      }).then((response) =>
        response.ok
          ? response.json()
          : Promise.reject()
      ),

      fetch(`${API}/api/decision-support/reports`, {
        signal: controller.signal,
      }).then((response) =>
        response.ok
          ? response.json()
          : Promise.reject()
      ),
    ]).then(([graph, incidents]) => {
      if (controller.signal.aborted) return;

      setMetrics({
        wards:
          graph.status === "fulfilled"
            ? graph.value.node_count
            : "—",
        centres:
          graph.status === "fulfilled"
            ? graph.value.centre_count
            : "—",
        edges:
          graph.status === "fulfilled"
            ? graph.value.edge_count
            : "—",
        reports:
          incidents.status === "fulfilled"
            ? incidents.value.count
            : "—",
      });

      if (incidents.status === "fulfilled") {
        setReports(incidents.value.reports || []);
      }
    });

    return () => controller.abort();
  }, []);

  return (
    <>
      <div className="cc-stats">
        <Stat
          icon="map"
          label="Mapped wards"
          value={metrics.wards}
          sub="Aluva region"
        />

        <Stat
          icon="shield"
          label="Relief centres"
          value={metrics.centres}
          sub="Simulated locations"
          tone="mint"
        />

        <Stat
          icon="pulse"
          label="Incident reports"
          value={metrics.reports}
          sub="Synthetic reports"
          tone="coral"
        />

        <Stat
          icon="network"
          label="Ward connections"
          value={metrics.edges}
          sub="Shared boundaries"
          tone="violet"
        />
      </div>

      <div className="cc-overview-grid">
        <FloodMap compact />

        <div className="cc-feed">
          <div className="cc-card-top">
            <div>
              <span className="cc-label">
                INCIDENT FEED
              </span>
              <h2>Disaster reports</h2>
            </div>

            <button
              className="cc-text-btn"
              onClick={() => go("decision")}
            >
              Explore
              <Glyph name="arrow" size={15} />
            </button>
          </div>

          <div className="cc-feed-list">
            {reports.length === 0 && (
              <p className="cc-soft">
                Loading reports…
              </p>
            )}

            {reports.slice(0, 6).map((report, index) => (
              <button
                key={report.report_id}
                className="cc-feed-item"
                onClick={() => go("decision")}
              >
                <div
                  className={`cc-feed-icon ${
                    index % 3 === 0 ? "alert" : ""
                  }`}
                >
                  <Glyph name="pulse" size={17} />
                </div>

                <div>
                  <strong>
                    {report.report_id}
                    <span> Ward {report.ward_no}</span>
                  </strong>

                  <p>{report.text}</p>
                </div>

                <Glyph name="arrow" size={16} />
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="cc-actions">
        <button onClick={() => go("network")}>
          <span>
            <Glyph name="network" />
            Graph intelligence
          </span>
          <Glyph name="arrow" />
        </button>

        <button onClick={() => go("decision")}>
          <span>
            <Glyph name="shield" />
            Get a recommendation
          </span>
          <Glyph name="arrow" />
        </button>

        <button onClick={() => go("datasets")}>
          <span>
            <Glyph name="database" />
            Analyze new dataset
          </span>
          <Glyph name="arrow" />
        </button>
      </div>
    </>
  );
}

export default function App() {
  const readPage = () => {
    const hash = window.location.hash.replace("#", "");

    return pages.some(
      (page) => page[0] === hash
    )
      ? hash
      : "overview";
  };

  const [page, setPage] = useState(readPage);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const handler = () => setPage(readPage());

    window.addEventListener("hashchange", handler);

    return () =>
      window.removeEventListener("hashchange", handler);
  }, []);

  const go = (id) => {
    setPage(id);
    setMobileOpen(false);

    window.history.replaceState(
      null,
      "",
      `#${id}`
    );

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  const [eyebrow, title, subtitle] = headings[page];

  return (
    <div className="cc-app">
      <aside
        className={`cc-sidebar ${
          mobileOpen ? "open" : ""
        }`}
      >
        <div className="cc-brand">
          <div className="cc-brand-icon">
            <Glyph name="pulse" size={27} />
          </div>

          <div>
            <strong>
              RESPONDA
              <span className="cc-brand-point">
                .
              </span>
            </strong>

            <small>DISASTER INTELLIGENCE</small>
          </div>
        </div>

        <div className="cc-sidebar-caption">
          WORKSPACE
        </div>

        <nav aria-label="Main navigation">
          {pages.map(([id, label, icon]) => (
            <button
              type="button"
              key={id}
              className={`cc-nav-item ${
                id === page ? "active" : ""
              }`}
              onClick={() => go(id)}
            >
              <Glyph name={icon} />
              {label}

              {id === page && <i />}
            </button>
          ))}
        </nav>

        <div className="cc-sidebar-foot">
          <span className="cc-location-icon">
            ⌖
          </span>

          <div>
            <strong>Aluva, Kerala</strong>
            <small>Pilot region</small>
          </div>
        </div>
      </aside>

      {mobileOpen && (
        <button
          className="cc-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <div className="cc-main">
        <header className="cc-header">
          <div className="cc-heading">
            <button
              className="cc-menu"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation"
            >
              ☰
            </button>

            <div>
              <span className="cc-label">
                {eyebrow}
              </span>

              <h1>{title}</h1>
              <p>{subtitle}</p>
            </div>
          </div>

          <div className="cc-header-right">
            <span className="cc-region">
              <i />
              ALUVA, KERALA
            </span>

            <span className="cc-avatar">
              DR
            </span>
          </div>
        </header>

        <main
          className="cc-body"
          key={page}
        >
          {page === "overview" && (
            <Overview go={go} />
          )}

          {page === "gis" && <FloodMap />}

          {page === "network" && (
            <GATNetwork />
          )}

          {page === "decision" && (
            <DecisionSupport />
          )}

          {page === "allocation" && (
            <AllocationDashboard />
          )}
          {page === "rolling" && (
  <RollingAllocation />
)}

          {page === "insights" && (
            <AIInsights />
          )}

          {page === "datasets" && (
            <DatasetManager />
          )}
        </main>
      </div>
    </div>
  );
}

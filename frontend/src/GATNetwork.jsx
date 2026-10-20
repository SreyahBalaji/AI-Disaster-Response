import { useEffect, useMemo, useRef, useState } from "react";
import "./GATNetwork.css";

const API = "http://127.0.0.1:8000";
const CANVAS = { width: 1100, height: 760 };
const MARGIN = 70;
const RESOURCE_LABELS = {
  water_litres: ["Water", "L"],
  food_meals: ["Food", "meals"],
  rescue_cases: ["Rescue", "cases"],
  shelter_households: ["Shelter", "households"],
  medical_cases: ["Medical", "cases"],
};

function wardColor(priority) {
  if (priority >= 0.5) return "#ff647b";
  if (priority >= 0.3) return "#ffbd60";
  return "#4ca6ff";
}

function priorityLabel(priority) {
  if (priority >= 0.5) return "High";
  if (priority >= 0.3) return "Moderate";
  return "Lower";
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

// Deterministic force layout: all connections come from the API, while
// these positions are illustrative and are NOT geographic coordinates.
function makeLayout(graph) {
  const sorted = [...graph.nodes].sort((a, b) => a.id - b.id);
  const count = sorted.length;
  const placed = new Map();
  const centreX = CANVAS.width / 2;
  const centreY = CANVAS.height / 2;

  sorted.forEach((node, index) => {
    const angle = (2 * Math.PI * index) / Math.max(1, count) - Math.PI / 2;
    const radius = 235 + ((index * 7) % 4) * 24;
    placed.set(node.id, {
      x: centreX + Math.cos(angle) * radius,
      y: centreY + Math.sin(angle) * radius,
    });
  });

  const edges = graph.edges.filter(
    (edge) => placed.has(edge.source) && placed.has(edge.target)
  );

  for (let iteration = 0; iteration < 320; iteration += 1) {
    const forces = new Map(sorted.map((node) => [node.id, { x: 0, y: 0 }]));

    for (let i = 0; i < sorted.length; i += 1) {
      for (let j = i + 1; j < sorted.length; j += 1) {
        const one = sorted[i].id;
        const two = sorted[j].id;
        const a = placed.get(one);
        const b = placed.get(two);
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let d = Math.hypot(dx, dy);
        if (d < 0.5) {
          dx = 1;
          dy = 0;
          d = 1;
        }
        // Mild global repulsion plus a stronger overlap-removal force.
        const overlap = Math.max(0, 88 - d);
        const amount = 1250 / (d * d) + overlap * 0.055;
        const fx = (dx / d) * amount;
        const fy = (dy / d) * amount;
        forces.get(one).x += fx;
        forces.get(one).y += fy;
        forces.get(two).x -= fx;
        forces.get(two).y -= fy;
      }
    }

    for (const edge of edges) {
      const a = placed.get(edge.source);
      const b = placed.get(edge.target);
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.max(1, Math.hypot(dx, dy));
      const magnitude = (d - 178) * 0.009;
      const fx = (dx / d) * magnitude;
      const fy = (dy / d) * magnitude;
      forces.get(edge.source).x += fx;
      forces.get(edge.source).y += fy;
      forces.get(edge.target).x -= fx;
      forces.get(edge.target).y -= fy;
    }

    for (const node of sorted) {
      const point = placed.get(node.id);
      const force = forces.get(node.id);
      const attraction = 0.003;
      const dx = (centreX - point.x) * attraction;
      const dy = (centreY - point.y) * attraction;
      const speed = 0.95;
      point.x = clamp(point.x + clamp(force.x + dx, -7, 7) * speed, 110, 990);
      point.y = clamp(point.y + clamp(force.y + dy, -7, 7) * speed, 92, 660);
    }
  }

  const layout = Object.fromEntries(placed.entries());
  // Keep centre markers away from wards and from each other. The link still
  // terminates at the correct host ward regardless of the display position.
  (graph.relief_centres || []).forEach((centre, index) => {
    const host = layout[centre.ward_no];
    if (!host) return;
    let best = null;
    let bestScore = -Infinity;
    for (let candidate = 0; candidate < 24; candidate += 1) {
      const angle = (candidate / 24) * Math.PI * 2 + index * .37;
      const distance = candidate % 3 === 0 ? 105 : candidate % 3 === 1 ? 122 : 142;
      const x = host.x + Math.cos(angle) * distance;
      const y = host.y + Math.sin(angle) * distance;
      if (x < 62 || x > CANVAS.width - 62 || y < 85 || y > CANVAS.height - 65) continue;
      const proximity = Object.entries(layout)
        .filter(([id]) => String(id) !== String(centre.ward_no))
        .reduce((min, [, point]) => Math.min(min, Math.hypot(point.x - x, point.y - y)), Infinity);
      const score = proximity - distance * .11;
      if (score > bestScore) { bestScore = score; best = { x, y }; }
    }
    layout[centre.id] = best || { x: clamp(host.x + 96, MARGIN, CANVAS.width - MARGIN), y: clamp(host.y - 85, MARGIN, CANVAS.height - MARGIN) };
  });
  return layout;
}

function HouseIcon() {
  return (
    <>
      <path d="M-14 -1 0 -14 14 -1" fill="none" stroke="#ffffff" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M-11 -1V12H11V-1" fill="white" stroke="white" strokeWidth="1.5" strokeLinejoin="round" />
      <rect x="-3" y="4" width="6" height="8" rx="1" fill="#0c66c8" />
    </>
  );
}

function FeatureBars({ features }) {
  const entries = Object.entries(features || {}).sort(([a], [b]) =>
    (Number(a.split("_").pop()) || 0) - (Number(b.split("_").pop()) || 0)
  );
  const maximum = Math.max(1, ...entries.map(([, value]) => Math.abs(Number(value) || 0)));
  if (entries.length === 0) return <p className="gat-empty">No saved embedding values.</p>;

  return (
    <div className="gat-feature-bars">
      {entries.map(([key, value], index) => {
        const n = Number(value) || 0;
        return (
          <div className="gat-feature-item" key={key}>
            <span>F{index + 1}</span>
            <div className="gat-feature-rail" aria-label={`Feature ${index + 1}, ${n.toFixed(3)}`}>
              <div className={n >= 0 ? "gat-feature-fill" : "gat-feature-fill negative"} style={{ width: `${Math.abs(n) / maximum * 100}%` }} />
            </div>
            <strong>{n.toFixed(2)}</strong>
          </div>
        );
      })}
    </div>
  );
}

export default function GATNetwork() {
  const [graph, setGraph] = useState(null);
  const [error, setError] = useState("");
  const [selection, setSelection] = useState({ type: "ward", id: 3 });
  const [positions, setPositions] = useState({});
  const [zoom, setZoom] = useState(1);
  const [showLabels, setShowLabels] = useState(true);
  const [focusPriority, setFocusPriority] = useState("all");
  const svgRef = useRef(null);
  const dragging = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API}/api/ai-insights/graph`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Graph API returned ${response.status}`);
        return response.json();
      })
      .then((payload) => {
        if (controller.signal.aborted) return;
        const resolved = {
          ...payload,
          nodes: payload.nodes || [],
          edges: payload.edges || [],
          relief_centres: payload.relief_centres || [],
          centre_links: payload.centre_links || [],
        };
        setGraph(resolved);
        setPositions(makeLayout(resolved));
        setSelection({ type: "ward", id: resolved.nodes.some((n) => n.id === 3) ? 3 : resolved.nodes[0]?.id });
      })
      .catch((err) => {
        if (err.name !== "AbortError") setError(err.message);
      });
    return () => controller.abort();
  }, []);

  const activeWard = selection.type === "ward" ? selection.id : null;
  const activeCentre = selection.type === "centre" ? selection.id : null;
  const ward = graph?.nodes.find((item) => item.id === activeWard);
  const centre = graph?.relief_centres.find((item) => item.id === activeCentre);
  const adjacent = useMemo(() => {
    const result = new Set();
    if (!graph || activeWard == null) return result;
    graph.edges.forEach(({ source, target }) => {
      if (source === activeWard) result.add(target);
      if (target === activeWard) result.add(source);
    });
    return result;
  }, [graph, activeWard]);
  const hosting = graph?.relief_centres.filter((item) => item.ward_no === activeWard) || [];

  function toGraphPoint(event) {
    const svg = svgRef.current;
    const matrix = svg?.getScreenCTM();
    if (!svg || !matrix) return null;
    const p = svg.createSVGPoint();
    p.x = event.clientX;
    p.y = event.clientY;
    const local = p.matrixTransform(matrix.inverse());
    return {
      x: CANVAS.width / 2 + (local.x - CANVAS.width / 2) / zoom,
      y: CANVAS.height / 2 + (local.y - CANVAS.height / 2) / zoom,
    };
  }

  function startDrag(event, type, id) {
    if (event.button !== 0 && event.pointerType !== "touch") return;
    setSelection({ type, id });
    dragging.current = id;
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function drag(event) {
    const id = dragging.current;
    if (id == null) return;
    const p = toGraphPoint(event);
    if (!p) return;
    setPositions((old) => ({
      ...old,
      [id]: {
        x: clamp(p.x, MARGIN, CANVAS.width - MARGIN),
        y: clamp(p.y, MARGIN, CANVAS.height - MARGIN),
      },
    }));
  }

  function stopDrag() {
    dragging.current = null;
  }

  function reset() {
    if (!graph) return;
    setPositions(makeLayout(graph));
    setZoom(1);
    setFocusPriority("all");
    setShowLabels(true);
  }

  function isMatch(priority) {
    return focusPriority === "all" || priorityLabel(priority).toLowerCase() === focusPriority;
  }

  return (
    <section className="gat-network-section" id="gat-network">
      <div className="gat-network-heading">
        <div>
          <span className="gat-eyebrow">WARD INTELLIGENCE</span>
          <h2>Connectivity Network</h2>
        </div>
        {graph && (
          <div className="gat-counts">
            <span><strong>{graph.node_count}</strong> wards</span>
            <span><strong>{graph.edge_count}</strong> connections</span>
            <span><strong>{graph.relief_centres.length}</strong> centres</span>
          </div>
        )}
      </div>

      {error && <div className="gat-error" role="alert">Unable to load network: {error}</div>}
      {!graph && !error && <div className="gat-loading">Loading Aluva's ward connections…</div>}

      {graph && (
        <div className="gat-network-layout">
          <div className="gat-canvas-card">
            <div className="gat-toolbar">
              <div className="gat-legend">
                <span><i className="gat-legend-dot high" />High priority</span>
                <span><i className="gat-legend-dot moderate" />Moderate</span>
                <span><i className="gat-legend-dot low" />Lower</span>
                <span><i className="gat-legend-dot centre" />Relief centre</span>
              </div>
              <div className="gat-controls">
                <select
                  className="gat-ward-select"
                  aria-label="Jump to ward or centre"
                  value={`${selection.type}:${selection.id}`}
                  onChange={(event) => {
                    const [type, id] = event.target.value.split(":");
                    setSelection({ type, id: type === "ward" ? Number(id) : id });
                  }}
                >
                  {graph.nodes.map((item) => <option key={`w${item.id}`} value={`ward:${item.id}`}>Ward {item.id}</option>)}
                  {graph.relief_centres.map((item) => <option key={item.id} value={`centre:${item.id}`}>{item.id}</option>)}
                </select>
                <button type="button" onClick={() => setShowLabels((old) => !old)} title="Toggle ward labels" aria-pressed={showLabels}>{showLabels ? "Labels on" : "Labels off"}</button>
                <button type="button" onClick={() => setZoom((old) => clamp(old - .2, .75, 2.4))} aria-label="Zoom out">−</button>
                <button type="button" onClick={() => setZoom((old) => clamp(old + .2, .75, 2.4))} aria-label="Zoom in">+</button>
                <button type="button" onClick={reset}>Reset</button>
              </div>
            </div>

            <div className="gat-focus-row" role="group" aria-label="Highlight priority group">
              <span>Highlight</span>
              {[["all", "All 26 wards"], ["high", "High"], ["moderate", "Moderate"], ["lower", "Lower"]].map(([id, label]) => (
                <button className={focusPriority === id ? "active" : ""} type="button" key={id} aria-pressed={focusPriority === id} onClick={() => setFocusPriority(id)}>{label}</button>
              ))}
            </div>

            <div className="gat-viewport">
              <svg
                ref={svgRef}
                className="gat-svg"
                viewBox={`0 0 ${CANVAS.width} ${CANVAS.height}`}
                role="img"
                aria-label={`Ward adjacency network showing ${graph.node_count} labeled ward nodes and ${graph.relief_centres.length} simulated relief centre nodes`}
              >
                <defs>
                  <radialGradient id="gatCanvasGlow"><stop offset="0%" stopColor="#174c8b" stopOpacity=".55"/><stop offset="100%" stopColor="#03112a" stopOpacity="0"/></radialGradient>
                  <pattern id="gatDotGrid" width="28" height="28" patternUnits="userSpaceOnUse"><circle cx="1.4" cy="1.4" r="1" fill="#589bd6" fillOpacity=".16" /></pattern>
                  <filter id="gatSelectedGlow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="9" /></filter>
                </defs>
                <rect width={CANVAS.width} height={CANVAS.height} fill="url(#gatDotGrid)" />
                <ellipse cx="550" cy="370" rx="480" ry="320" fill="url(#gatCanvasGlow)" />
                <g transform={`translate(${CANVAS.width / 2} ${CANVAS.height / 2}) scale(${zoom}) translate(${-CANVAS.width / 2} ${-CANVAS.height / 2})`}>
                  {graph.edges.map((edge, index) => {
                    const a = positions[edge.source];
                    const b = positions[edge.target];
                    if (!a || !b) return null;
                    const emphasized = activeWard === edge.source || activeWard === edge.target ||
                      (centre && (centre.ward_no === edge.source || centre.ward_no === edge.target));
                    return (
                      <line
                        key={`e-${index}`}
                        className={emphasized ? "gat-edge active" : "gat-edge"}
                        x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                        stroke={emphasized ? "#50c9ff" : "#54759e"}
                        strokeWidth={emphasized ? 3.2 : 1.6}
                        opacity={emphasized ? .93 : .43}
                      >
                        <title>Wards {edge.source}–{edge.target}: {Number(edge.shared_boundary_m).toFixed(2)} m shared boundary</title>
                      </line>
                    );
                  })}
                  {graph.centre_links.map((link) => {
                    const a = positions[link.source];
                    const b = positions[link.target];
                    if (!a || !b) return null;
                    const highlighted = activeCentre === link.source || activeWard === link.target;
                    return <line key={link.source} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#31decb" strokeWidth={highlighted ? 4 : 2.7} strokeDasharray="9 7" opacity={highlighted ? 1 : .76} />;
                  })}
                  {graph.nodes.map((node) => {
                    const at = positions[node.id];
                    if (!at) return null;
                    const picked = activeWard === node.id;
                    const isAdjacent = adjacent.has(node.id);
                    const centreHost = centre?.ward_no === node.id;
                    const related = picked || isAdjacent || centreHost;
                    const muted = !isMatch(node.priority) && !related;
                    const color = wardColor(node.priority);
                    return (
                      <g
                        className={`gat-node gat-ward-node ${picked ? "selected" : ""} ${muted ? "muted" : ""}`}
                        key={`ward-${node.id}`}
                        transform={`translate(${at.x}, ${at.y})`}
                        role="button"
                        tabIndex={0}
                        aria-label={`Ward ${node.id}, ${node.name}, ${priorityLabel(node.priority)} priority`}
                        onPointerDown={(event) => startDrag(event, "ward", node.id)}
                        onPointerMove={drag}
                        onPointerUp={stopDrag}
                        onPointerCancel={stopDrag}
                        onLostPointerCapture={stopDrag}
                        onClick={() => setSelection({ type: "ward", id: node.id })}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelection({ type: "ward", id: node.id }); }
                        }}
                      >
                        {related && <circle r="32" fill={picked ? "#26d4ff" : "#62adff"} opacity=".47" filter="url(#gatSelectedGlow)" />}
                        <circle r={picked ? 27 : 24} fill="#06172c" stroke={picked ? "#aff4ff" : "#75b7f1"} strokeWidth={picked ? 3.3 : 1.6} />
                        <circle r={picked ? 22 : 19} fill={color} stroke="rgba(255,255,255,.4)" strokeWidth="1" />
                        <text className="gat-ward-number" textAnchor="middle" dominantBaseline="central">{node.id}</text>
                        {showLabels && (
                          <g className="gat-node-label" aria-hidden="true">
                            <rect x="-22" y="-48" width="44" height="20" rx="7" />
                            <text className="gat-label-text" x="0" y="-34" textAnchor="middle">W{node.id}</text>
                          </g>
                        )}
                        <title>{`Ward ${node.id}: ${node.name}; priority ${Number(node.priority).toFixed(4)}`}</title>
                      </g>
                    );
                  })}
                  {graph.relief_centres.map((item) => {
                    const at = positions[item.id];
                    if (!at) return null;
                    const picked = activeCentre === item.id;
                    const hostSelected = activeWard === item.ward_no;
                    return (
                      <g
                        key={item.id}
                        className={`gat-node gat-centre-node ${picked ? "selected" : ""}`}
                        transform={`translate(${at.x}, ${at.y})`}
                        role="button"
                        tabIndex={0}
                        aria-label={`${item.id}, simulated relief centre, Ward ${item.ward_no}`}
                        onPointerDown={(event) => startDrag(event, "centre", item.id)}
                        onPointerMove={drag}
                        onPointerUp={stopDrag}
                        onPointerCancel={stopDrag}
                        onLostPointerCapture={stopDrag}
                        onClick={() => setSelection({ type: "centre", id: item.id })}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelection({ type: "centre", id: item.id }); }
                        }}
                      >
                        {(picked || hostSelected) && <circle r="41" fill="#17f0cb" opacity=".33" filter="url(#gatSelectedGlow)" />}
                        <circle r="32" fill="#09243e" stroke="#59ffe4" strokeWidth="3" />
                        <circle r="27" fill="#0c92b8" />
                        <HouseIcon />
                        {showLabels && (
                          <g className="gat-node-label">
                            <rect x="-29" y="-59" width="58" height="22" rx="7" />
                            <text className="gat-label-text centre" textAnchor="middle" y="-43">{item.id}</text>
                          </g>
                        )}
                        <title>{item.name} — simulated in Ward {item.ward_no}</title>
                      </g>
                    );
                  })}
                </g>
              </svg>
            </div>
            <div className="gat-canvas-note"><span><b>Solid</b> = shared ward boundary</span><span><b>Dashed teal</b> = relief centre's host ward</span><span>Click or drag a node to explore</span></div>
          </div>

          <aside className="gat-details">
            {ward && (
              <>
                <span className="gat-eyebrow">WARD PROFILE</span>
                <div className="gat-profile-top">
                  <div><h3>Ward {ward.id}</h3><p>{ward.name}</p></div>
                  <span className="gat-priority" style={{ background: `${wardColor(ward.priority)}22`, color: wardColor(ward.priority), borderColor: `${wardColor(ward.priority)}66` }}>{priorityLabel(ward.priority)} priority</span>
                </div>
                <div className="gat-score-card"><span>Flood priority</span><strong>{Number(ward.priority).toFixed(4)}</strong><div className="gat-score-track"><div style={{ width: `${clamp(Number(ward.priority) * 100, 0, 100)}%`, background: wardColor(ward.priority) }}/></div></div>
                <div className="gat-quick-stats"><div><span>Flood exposure</span><strong>{Number(ward.flood_exposure).toFixed(4)}</strong></div><div><span>Population score</span><strong>{Number(ward.population_score).toFixed(4)}</strong></div></div>
                <div className="gat-details-heading">Adjacent wards <span>{adjacent.size}</span></div>
                <div className="gat-neighbors">{[...adjacent].sort((a, b) => a - b).map((id) => <button key={id} type="button" onClick={() => setSelection({ type: "ward", id })}>W{id}</button>)}</div>
                {hosting.length > 0 && <><div className="gat-details-heading">Relief centres</div><div className="gat-neighbors">{hosting.map((item) => <button className="gat-centre-pill" key={item.id} type="button" onClick={() => setSelection({ type: "centre", id: item.id })}>{item.id} →</button>)}</div></>}
                <div className="gat-details-heading">GAT embedding <span>8 features</span></div>
                <FeatureBars features={ward.features} />
              </>
            )}
            {centre && (
              <>
                <span className="gat-eyebrow">RELIEF CENTRE</span>
                <div className="gat-profile-top"><div><h3>{centre.id}</h3><p>{centre.name}</p></div><span className="gat-priority centre">Simulated</span></div>
                <button className="gat-host-link" onClick={() => setSelection({ type: "ward", id: centre.ward_no })} type="button">Located in Ward {centre.ward_no} <span>→</span></button>
                <div className="gat-details-heading">Initial inventory</div>
                <div className="gat-inventory-list">
                  {Object.entries(centre.resources || {}).map(([key, count]) => {
                    const [label, unit] = RESOURCE_LABELS[key] || [key, "units"];
                    return <div className="gat-inventory-row" key={key}><span>{label}</span><strong>{Number(count).toLocaleString()} <small>{unit}</small></strong></div>;
                  })}
                </div>
                <p className="gat-centre-context">Initial simulated stock. No live dispatch or road routing is represented.</p>
              </>
            )}
          </aside>
        </div>
      )}
    </section>
  );
}

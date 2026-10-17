
import { useEffect, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import AllocationDashboard from "./AllocationDashboard";
<AllocationDashboard />
const API_URL = "http://127.0.0.1:8000/api/gis/priority-map";

const priorityColors = {
  high: "#e74c3c",
  moderate: "#f39c12",
  low: "#27ae60",
};
function FitWardBounds({ data }) {
  const map = useMap();

  useEffect(() => {
    const bounds = L.geoJSON(data).getBounds();

    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [25, 25] });
    }
  }, [data, map]);

  return null;
}
function App() {
  const [wardData, setWardData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(API_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Unable to fetch ward data");
        }
        return response.json();
      })
      .then((data) => setWardData(data))
      .catch((err) => setError(err.message));
  }, []);

  const wardStyle = (feature) => ({
    fillColor:
      priorityColors[feature.properties.priority_category] || "#95a5a6",
    weight: 1.5,
    color: "#ffffff",
    fillOpacity: 0.7,
  });

  const onEachWard = (feature, layer) => {
    const ward = feature.properties;

    layer.bindPopup(`
      <strong>${ward.ward_name}</strong><br/>
      Ward Number: ${ward.ward_no}<br/>
      Flood Priority: ${Number(ward.flood_priority_score).toFixed(3)}<br/>
      Category: ${ward.priority_category}
    `);
  };

  return (
    <div style={{ padding: "24px", fontFamily: "Arial, sans-serif" }}>
      <h1>AI-Driven Disaster Response System</h1>
      <p>Aluva Ward Flood-Priority Visualization</p>

      <div style={{ marginBottom: "15px" }}>
        🔴 High Priority &nbsp;&nbsp;
        🟠 Moderate Priority &nbsp;&nbsp;
        🟢 Low Priority
      </div>

      {error && <p style={{ color: "red" }}>Error: {error}</p>}

      {!wardData && !error && <p>Loading ward data...</p>}

      {wardData && (
        <>
          <p>Total Wards: {wardData.features.length}</p>

          <MapContainer
             center={[10.11, 76.35]}
              zoom={12}
              style={{ height: "600px", width: "100%" }}
          >
  <FitWardBounds data={wardData} />
            <TileLayer
              attribution='&copy; OpenStreetMap contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            <GeoJSON
              data={wardData}
              style={wardStyle}
              onEachFeature={onEachWard}
            />
          </MapContainer>
          <AllocationDashboard />
        </>
        
      )}

      <p style={{ marginTop: "15px", fontSize: "13px", color: "#666" }}>
        Research prototype. Priority scores are calculated using flood
        exposure and population proxies, not official emergency warnings.
      </p>
    </div>
  );
}

export default App;

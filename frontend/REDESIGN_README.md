# Responda — Frontend Redesign

## Install

1. Back up the existing `frontend` folder (or commit your current changes).
2. Extract this ZIP **into your existing `frontend` folder**, keeping the `src/` directory structure; choose to replace matching files.
3. From `D:\AI-Disaster-Response\frontend` run:

   ```powershell
   npm install
   npm run dev
   ```

4. Keep the existing FastAPI server running at `http://127.0.0.1:8000`.
5. Open `http://localhost:5173`.
6. Optional compile check: `npm run build`.

## Changes

- New premium, dark-blue command-centre interface with separate Overview, GIS Intelligence, GAT Network, Decision Support, Resource Allocation, and Model Analytics views.
- Uses the existing backend endpoints and original working functional components.
- Overview stats come from `/api/ai-insights/graph` and `/api/decision-support/reports`.
- Removed the repetitive research-status notices and overview banner from the UI.
- Existing `App.css` and component CSS files are included; `CommandCenter.css` applies the revised theme.

## Scientific limits (retain in documentation)

- Incident reports and relief centres are simulated; the UI is not connected to emergency response operations.
- Ward adjacency edges are not road connections or travel times.
- The GAT embedding visualization is not a learned attention edge-weight display and does not drive the current heuristic.
- The trained multi-resource PPO works on wards 3, 5, 9; the broader report-to-allocation support is a heuristic.
- Saved federated evaluation is not directly integrated into the displayed multi-resource allocation policy.
- Allocation comparisons must be reported honestly; baseline performance can exceed PPO.

## Validation

JSX syntax was parsed successfully for all the delivered JSX files. A full Vite dependency install/build could not be run in this environment. Please run `npm run build` on your machine and report errors if any.

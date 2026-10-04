# AI-Driven Disaster Response — Data Schema

## 1. Affected Areas

- area_id
- area_name
- district
- latitude
- longitude
- population
- flood_hazard
- disaster_severity
- vulnerability_score

## 2. Relief Centres

- centre_id
- centre_name
- latitude
- longitude
- capacity
- food_stock
- water_stock
- medicine_stock
- rescue_teams

## 3. Resource Demand

- timestamp
- area_id
- food_demand
- water_demand
- medicine_demand
- rescue_demand

## 4. Road Network

- road_id
- source_id
- destination_id
- distance
- estimated_travel_time
- road_status

## 5. Crisis Reports

- report_id
- timestamp
- report_text
- area_id
- extracted_resource
- urgency
- humanitarian_category

## 6. Allocation Results

- allocation_id
- timestamp
- centre_id
- area_id
- resource_type
- quantity
- estimated_delivery_time
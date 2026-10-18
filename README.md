# EV Fleet Asset Valuation & Lifecycle Model

## 📌 Project Overview
This project models the depreciation, operational lifecycle, and battery degradation of an 8,100+ unit Electric Vehicle (EV) fleet. The goal was to build a data-driven system to identify the optimal resale window for assets and automatically flag vehicles for second-life repurposing (e.g., grid storage) once battery health falls below 80%.

## 🛠️ Technology Stack
* **Data Processing & ML:** Python (Pandas, Scikit-Learn)
* **Database Analysis:** SQL (SQLite, CTEs, Window Functions)
* **Business Intelligence:** Power BI, DAX

## 🧠 Methodology & Machine Learning
1. **Unsupervised Segmentation (K-Means):** Grouped the fleet into three distinct lifecycle tiers:
   * *Premium Health (Resale Focus)*
   * *Mid-Life (Monitoring)*
   * *Second-Life Target (Recycling/Repurposing)*
2. **Predictive Modeling (Random Forest):** Trained a regression model to identify primary drivers of battery capacity loss, proving that vehicle age (52%) and total mileage (38%) far outweigh fast-charging frequency as degradation drivers.

## 📊 Power BI Dashboard
*<img width="1125" height="633" alt="Executive Overview PowerBI" src="https://github.com/user-attachments/assets/3d76ec5f-6094-47a7-84b4-c8e4ea3c0f64" />
<img width="1125" height="634" alt="Degradation Drivers   Machine Learning Insights PoweBI" src="https://github.com/user-attachments/assets/d741fe70-371d-4cc0-9a63-7b71dd551d40" />
<img width="1125" height="620" alt="Battery Degradation   Second-Life Segmentation PowerBI" src="https://github.com/user-attachments/assets/b21d0b2f-a7a3-40fb-899e-ad454ea65775" />
*
* **Page 1: Executive Valuation Overview** - Tracks total fleet value and brand depreciation.
* **Page 2: Second-Life Target Matrix** - Filters high-mileage, degraded assets.
* **Page 3: ML Driver Insights** - Translates Random Forest outputs into strategic fleet actions.

## 💼 Strategic Recommendations (Asset Management)
* **Hold vs. Sell:** Exit vehicles from primary operations at Year 5 (approx. 85-88% battery health) to maximize capital recovery before steep linear depreciation occurs.
* **Capital Reallocation:** Vehicles crossing the 118,000 km threshold should be evaluated for secondary market liquidation or internal power-backup repurposing.

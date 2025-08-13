# Industrial Spare Parts Demand Forecasting

## 📌 Overview
An interactive **Streamlit-based dashboard** for **forecasting and inventory optimization** in industrial spare parts management.  
The app supports **synthetic data generation** and **real CSV uploads**, processes demand history and part metadata, and applies **KMeans clustering** to group similar parts.  
For forecasting, it uses **LightGBM regression with quantile estimates** to:
- Predict future demand.
- Provide confidence intervals.
- Break down contributions from **trend, seasonality, and variability**.

Additional features include:
- **Inventory policy recommendations** (reorder points, suggested order quantities).
- **Delivery delay analysis** with root cause visualization.

---

## 🎯 Live Demo
Try the app here: [**Industrial Spare Parts Forecasting – Streamlit Cloud**](https://your-app-name.streamlit.app)  

---

## 🚀 Features
- **Synthetic Data Generation** – Create realistic datasets for 100+ parts over multiple years.
- **Custom Data Upload** – Upload real demand and metadata CSV files.
- **Clustering with KMeans** – Group similar parts by demand patterns and attributes.
- **LightGBM Forecasting** – Predict demand and generate confidence intervals.
- **Feature Engineering** – Lags, rolling stats, seasonality, trend detection.
- **Interactive Visualizations** – Cluster plots, demand forecasts, delivery performance charts.
- **Inventory Recommendations** – Suggested order quantities, reorder triggers, safety stock.
- **Delay Analysis** – Delivery performance trends and root cause breakdown.

---

## 🛠 Tech Stack
- **UI & Visualization**: Streamlit, Plotly
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: LightGBM, scikit-learn (KMeans, StandardScaler)
- **Utilities**: Faker (synthetic data), datetime, random

---

## 📂 Workflow
1. **Data Input**
   - Generate synthetic data OR upload CSVs.
2. **Preprocessing**
   - Aggregate monthly demand, calculate statistics.
3. **Clustering**
   - Group parts based on selected features using KMeans.
4. **Forecasting**
   - Train LightGBM models (median, quantile, regular).
   - Predict future demand for selected clusters.
5. **Post-Forecast Analysis**
   - Confidence intervals, component breakdown.
6. **Recommendations**
   - Reorder points, suggested quantities, safety stock.
7. **Delay Analysis**
   - Delivery performance metrics and cause analysis.

---

## 📥 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/industrial-spare-parts-forecasting.git
cd industrial-spare-parts-forecasting

# Run the app
streamlit run app.py

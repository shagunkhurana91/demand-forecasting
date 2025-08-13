
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from faker import Faker
import random
from datetime import datetime, timedelta


st.set_page_config(page_title="Industrial Spare Parts Forecasting", layout="wide")
st.title("Industrial Spare Parts Demand Forecasting with ML Clustering")


def generate_synthetic_data(years=3):
    fake = Faker()
    parts = [f"PART-{str(i).zfill(5)}" for i in range(1, 101)]
    categories = ["Engine", "Electrical", "Hydraulic", "Pneumatic", "Structural"]
    suppliers = ["Supplier A", "Supplier B", "Supplier C", "Supplier D"]
    lead_times = [9, 14, 21, 30, 45, 60]
    costs = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    festivals = [True,False]
    # Generate part metadata
    part_metadata = pd.DataFrame({
        'part_number': parts,
        'category': np.random.choice(categories, size=len(parts)),
        'supplier': np.random.choice(suppliers, size=len(parts)),
        'lead_time_days': np.random.choice(lead_times, size=len(parts)),
        'unit_cost': np.random.choice(costs, size=len(parts)),
        'criticality': np.random.choice(["Low", "Medium", "High"], size=len(parts)),
        'safety_stock_days': np.random.randint(1, 10, size=len(parts)),
        'festivals': np.random.choice(festivals, size=len(parts)),
        'product_lifecycle_days': np.random.randint(30, 365, size=len(parts)),
    })
    
    start_date = datetime.now() - timedelta(days=365*years)
    dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
    
    demand_data = []
    for part in parts:
      
        base_level = max(1, int(np.random.normal(loc=5, scale=3)))
        trend = random.choice([-0.5, -0.2, 0, 0, 0.2, 0.5])  
        
        for i, date in enumerate(dates):
            
            trend_effect = 1 + (i/len(dates)) * trend
            
            month_factor = 1 + 0.5 * np.sin(2 * np.pi * date.month/12)
            
            day_factor = 1 + 0.2 * np.sin(2 * np.pi * date.weekday()/7)
            
            spike = 10 if random.random() < 0.05 else 1
            
            
            calculated_demand = base_level * trend_effect * month_factor * day_factor * spike * random.uniform(0.7, 1.3)
            
            demand = max(0, int(calculated_demand))
            
            if demand > 0 or random.random() < 0.3:  
                demand_data.append({
                    'date': date,
                    'part_number': part,
                    'demand': demand,
                    'price': part_metadata.loc[part_metadata['part_number'] == part, 'unit_cost'].values[0] * random.uniform(0.8, 1.2),
                })
    
    demand_df = pd.DataFrame(demand_data)
    return demand_df, part_metadata

st.sidebar.header("Data Options")
if st.sidebar.button("Generate Synthetic Data"):
    with st.spinner("Generating realistic industrial dataset..."):
        demand_df, part_metadata = generate_synthetic_data()
        st.session_state.demand_df = demand_df
        st.session_state.part_metadata = part_metadata
    st.sidebar.success("Generated synthetic data with 100 parts and 3 years of demand!")

uploaded_file = st.sidebar.file_uploader("Or upload your own demand data CSV", type=["csv"])

if uploaded_file:
    demand_df = pd.read_csv(uploaded_file, parse_dates=["date"])
    st.session_state.demand_df = demand_df
    st.sidebar.success("✅ Data uploaded successfully!")
    
    
    metadata_file = st.sidebar.file_uploader("Upload part metadata CSV (optional)", type=["csv"])
    if metadata_file:
        part_metadata = pd.read_csv(metadata_file)
        st.session_state.part_metadata = part_metadata
        st.sidebar.success("✅ Metadata uploaded successfully!")

if 'demand_df' in st.session_state:
    demand_df = st.session_state.demand_df
    
    
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Parts", len(demand_df['part_number'].unique()))
    col2.metric("Date Range", f"{demand_df['date'].min().date()} to {demand_df['date'].max().date()}")
    col3.metric("Total Demand Records", len(demand_df))
    
    
    with st.expander("View Sample Data"):
        st.dataframe(demand_df.sample(5))
    
    
    demand_df['month'] = demand_df['date'].dt.to_period('M').dt.to_timestamp()
    monthly_pivot = demand_df.pivot_table(index='month', columns='part_number', values='demand', aggfunc='sum').fillna(0)
    
    
    if 'part_metadata' in st.session_state:
        part_metadata = st.session_state.part_metadata
        st.subheader("Part Metadata")
        st.dataframe(part_metadata)
        
       
        demand_stats = demand_df.groupby('part_number').agg(
            total_demand=('demand', 'sum'),
            demand_variability=('demand', 'std'),
            avg_price=('price', 'mean'),
            demand_frequency=('demand', lambda x: (x > 0).mean())  # For intermittent demand
        ).fillna(0)
        
        demand_stats.rename(columns={'demand_variability': 'demand_variability (%)'}, inplace=True)
        demand_stats.rename(columns={'total_demand': 'total_demand (units)'}, inplace=True)
        part_features = part_metadata.merge(demand_stats, on='part_number', how='left')
        
       
        st.subheader("Advanced Part Clustering")
        cluster_features = st.multiselect(
            "Select features for clustering",
            options=part_features.select_dtypes(include=['number']).columns.tolist(),
            default=['total_demand (units)', 'demand_variability (%)', 'lead_time_days', 'unit_cost', 'demand_frequency']
        )
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(part_features[cluster_features])
        
        k = st.slider("Number of Clusters", 2, 8, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        
        part_features['cluster'] = labels
        st.dataframe(part_features.sort_values("cluster"))
        
       
        st.subheader("Cluster Characteristics")
        cluster_summary = part_features.groupby('cluster')[cluster_features].mean()
        st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))
        
        
        if len(cluster_features) >= 2:
            fig = go.Figure()
            
            for cluster in sorted(part_features['cluster'].unique()):
                cluster_data = part_features[part_features['cluster'] == cluster]
                fig.add_trace(go.Scatter(
                    x=cluster_data[cluster_features[0]],
                    y=cluster_data[cluster_features[1]],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    text=cluster_data['part_number'],
                    marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey'))
                ))
            
            fig.update_layout(
                title=f"2D Cluster Visualization ({cluster_features[0]} vs {cluster_features[1]})",
                xaxis_title=cluster_features[0],
                yaxis_title=cluster_features[1],
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
       
        st.warning("No part metadata found. Using basic demand patterns for clustering.")
        scaler = StandardScaler()
        pivot_scaled = scaler.fit_transform(monthly_pivot.T)
        
        k = st.slider("Number of Clusters", 2, 6, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pivot_scaled)
        
        cluster_df = pd.DataFrame({'part_number': monthly_pivot.columns, 'cluster': labels})
        st.subheader("Part Clusters")
        st.dataframe(cluster_df.sort_values("cluster"))
        part_features = cluster_df
    
    
    st.subheader("Variable Demand Forecasting")
    selected_cluster = st.selectbox("Select Cluster to Forecast", sorted(part_features["cluster"].unique()))
    selected_parts = part_features[part_features['cluster'] == selected_cluster]['part_number'].tolist()
    
    forecast_months = st.slider("Forecast Months Ahead", 1, 12, 6)
    confidence_level = st.slider("Estimated Range(%)", 70, 95, 80) / 100
    include_seasonality = st.checkbox("Include Seasonality Effects", value=True)
    include_trend = st.checkbox("Include Trend Detection", value=True)
    variability_level = st.slider("Forecast Variability(%)", 0.1, 1.0, 0.5)

    for part in selected_parts[:5]:  # Limit to 5 parts for demo
        st.markdown(f"---\n### Forecast for Part `{part}`")
        
        
        part_df = demand_df[demand_df['part_number'] == part].groupby("month")["demand"].sum().reset_index()
        part_df["month_num"] = np.arange(len(part_df))
        
       
        part_df['month_sin'] = np.sin(2 * np.pi * part_df['month'].dt.month/12)
        part_df['month_cos'] = np.cos(2 * np.pi * part_df['month'].dt.month/12)
        
  
        if include_trend and len(part_df) > 6:
            z = np.polyfit(part_df['month_num'], part_df['demand'], 1)
            part_df['trend'] = z[0] * part_df['month_num'] + z[1]
        else:
            part_df['trend'] = 0
        
 
        for i in range(1, 5):
            part_df[f"lag_{i}"] = part_df["demand"].shift(i)

        part_df["rolling_avg_3"] = part_df["demand"].rolling(3).mean().shift(1)
        part_df["rolling_std_3"] = part_df["demand"].rolling(3).std().shift(1)
        part_df["ewm"] = part_df["demand"].ewm(span=3).mean().shift(1)
        
        part_df.dropna(inplace=True)
        
      
        models = {
            'lgbm_quantile': lgb.LGBMRegressor(objective='quantile', alpha=confidence_level),
            'lgbm_median': lgb.LGBMRegressor(objective='quantile', alpha=0.5),
            'lgbm_regular': lgb.LGBMRegressor()
        }
        
        
        features = ["month_num", "lag_1", "lag_2", "lag_3", "rolling_avg_3", "rolling_std_3", "ewm"]
        if include_seasonality:
            features.extend(["month_sin", "month_cos"])
        if include_trend:
            features.append("trend")
        
        X = part_df[features]
        y = part_df["demand"]
        
        # Train models
        for name, model in models.items():
            model.fit(X, y)
        
       
        future_df = part_df.copy()
        all_forecasts = []
        forecast_components = []
        forecast_dates = pd.date_range(
            start=part_df["month"].iloc[-1] + pd.DateOffset(months=1),
            periods=forecast_months, 
            freq="MS"
        )
        
        for i in range(forecast_months):
            
            new_row = {
                "month_num": future_df["month_num"].iloc[-1] + 1,
                "month_sin": np.sin(2 * np.pi * ((future_df['month'].iloc[-1].month + i) % 12)/12),
                "month_cos": np.cos(2 * np.pi * ((future_df['month'].iloc[-1].month + i) % 12)/12),
                "lag_1": future_df["demand"].iloc[-1],
                "lag_2": future_df["demand"].iloc[-2],
                "lag_3": future_df["demand"].iloc[-3],
                "rolling_avg_3": future_df["demand"].iloc[-3:].mean(),
                "rolling_std_3": future_df["demand"].iloc[-3:].std(),
                "ewm": future_df["demand"].ewm(span=3).mean().iloc[-1]
            }
            
            if include_trend:
                new_row["trend"] = future_df["trend"].iloc[-1] + z[0]  
            
            x_pred = pd.DataFrame([new_row])[features]
            
            
            preds = {
                'median': models['lgbm_median'].predict(x_pred)[0],
                'quantile': models['lgbm_quantile'].predict(x_pred)[0],
                'regular': models['lgbm_regular'].predict(x_pred)[0]
            }
            
          
            weights = {'median': 0.5, 'quantile': 0.3, 'regular': 0.2}
            combined_pred = sum(preds[m] * weights[m] for m in preds.keys())
            
    
            variability = future_df["demand"].std() * variability_level * np.random.normal()
            final_pred = max(0, combined_pred + variability)
            
     
            components = {
                'base': combined_pred,
                'variability': variability,
                'seasonality': new_row.get("month_sin", 0) + new_row.get("month_cos", 0),
                'trend': new_row.get("trend", 0)
            }
            
            new_row["demand"] = final_pred
            all_forecasts.append(final_pred)
            forecast_components.append(components)
            future_df = pd.concat([future_df, pd.DataFrame([new_row])], ignore_index=True)
        
  
        train_preds = models['lgbm_regular'].predict(X)
        residuals = y - train_preds
        std_residual = residuals.std()
        
      
        lower_bounds = []
        upper_bounds = []
        # for i, pred in enumerate(all_forecasts):
           
        #     horizon_factor = 1 + (i/forecast_months)
        #     lower = max(0, pred - horizon_factor * 1.5 * std_residual)
        #     upper = pred + horizon_factor * 1.5 * std_residual
        #     lower_bounds.append(lower)
        #     upper_bounds.append(upper)
        
      
        # full_dates = list(part_df["month"]) + list(forecast_dates)
        # actual_dates = full_dates[:len(part_df)]
        # forecast_dates_list = full_dates[len(part_df):]
        
        # fig = go.Figure()
        
    
        # fig.add_trace(go.Scatter(
        #     x=actual_dates,
        #     y=part_df["demand"],
        #     mode="lines+markers",
        #     name="Actual Demand",
        #     line=dict(color='royalblue', width=2),
        #     marker=dict(size=6)
        # ))
        
    
        # fig.add_trace(go.Scatter(
        #     x=forecast_dates_list,
        #     y=all_forecasts,
        #     mode="lines+markers",
        #     name="Forecast",
        #     line=dict(color='orange', width=2, dash='dash'),
        #     marker=dict(size=6)
        # ))
        
   
        # fig.add_trace(go.Scatter(
        #     x=forecast_dates_list + forecast_dates_list[::-1],
        #     y=upper_bounds + lower_bounds[::-1],
        #     fill='toself',
        #     fillcolor='rgba(255, 165, 0, 0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     name=f"{int(confidence_level*100)}% Confidence"
        # ))
        
        # fig.update_layout(
        #     title=f"Demand Forecast for {part} (cluster {selected_cluster})",
        #     xaxis_title="Date",
        #     yaxis_title="Demand Quantity",
        #     hovermode="x unified",
        #     showlegend=True
        # )
        
        # st.plotly_chart(fig, use_container_width=True)
        
        for i, pred in enumerate(all_forecasts):
            horizon_factor = 1 + (i/forecast_months)
            lower = max(0, pred - horizon_factor * 1.5 * std_residual)
            upper = (pred + horizon_factor * 1.5 * std_residual)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        full_dates = list(part_df["month"]) + list(forecast_dates)
        actual_dates = full_dates[:len(part_df)]
        forecast_dates_list = full_dates[len(part_df):]

        fig = go.Figure()

      
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=part_df["demand"],
            mode="lines+markers",
            name="Actual Demand",
            line=dict(color='royalblue', width=2),
            marker=dict(size=6)
        ))

  
        fig.add_trace(go.Scatter(
            x=forecast_dates_list,
            y=all_forecasts,
            mode="lines+markers",
            name="Forecast",
            line=dict(color='orange', width=2, dash='dash'),
            marker=dict(size=6)
        ))

       
        fig.add_trace(go.Scatter(
            x=forecast_dates_list + forecast_dates_list[::-1],
            y=upper_bounds + lower_bounds[::-1],
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f"{int(confidence_level*100)}% Confidence"
        ))

        fig.update_layout(
            title=f"Demand Forecast for {part} (cluster {selected_cluster})",
            xaxis_title="Date",
            yaxis_title="Demand Quantity",
            hovermode="x unified",
            showlegend=True,
            xaxis=dict(
                type='category', 
                tickmode='array',
                tickvals=full_dates,  
                ticktext=[date.strftime('%b %Y') for date in full_dates],  
                tickangle=45,  
                tickfont=dict(size=10)  
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Forecast Components Analysis"):
            components_df = pd.DataFrame(forecast_components)
            st.write("How each component contributed to the forecast:")
            st.dataframe(components_df)
            
            fig_components = go.Figure()
            for col in components_df.columns:
                fig_components.add_trace(go.Scatter(
                    x=forecast_dates_list,
                    y=components_df[col],
                    mode="lines+markers",
                    name=col
                ))
            fig_components.update_layout(title="Forecast Components Breakdown")
            st.plotly_chart(fig_components, use_container_width=True)
        
  
        if 'part_metadata' in st.session_state:
            part_meta = part_metadata[part_metadata['part_number'] == part].iloc[0]
            avg_lead_time = part_meta['lead_time_days']
            safety_stock = part_meta['safety_stock_days']
            
            st.subheader("Inventory Recommendations")
            
            avg_monthly_demand = part_df["demand"].mean()
            forecasted_next_quarter = sum(all_forecasts[:3])
            
            safety_factor = max(1.5, 3 * variability_level)  
            suggested_order = max(
                forecasted_next_quarter * (1 + variability_level),
                safety_stock * avg_monthly_demand * safety_factor
            )
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Forecasted Next Quarter", f"{forecasted_next_quarter:.0f} ± {np.std(all_forecasts[:3]):.0f}")
            col2.metric("Suggested Order Qty", f"{suggested_order:.0f}")
            col3.metric("Lead Time", f"{avg_lead_time} days")
            
            service_level = 0.95 
            z_score = 1.645  
            demand_std = part_df["demand"].std()
            reorder_point = (avg_monthly_demand/30 * avg_lead_time) + \
                          (z_score * demand_std * np.sqrt(avg_lead_time/30))
            
            st.info(f"""
            **Inventory Policy Recommendations:**
            - Reorder when inventory reaches **{int(reorder_point)} units**
            - Safety stock should cover **{safety_factor:.1f}×** normal variability
            - Consider ordering **{int(suggested_order)}** units next
            """)

       
        st.subheader("Delivery Delay Detection")
        
 
        delivery_data = pd.DataFrame({
            'order_date': pd.date_range(start=part_df['month'].min(), 
                                      periods=len(part_df),
                                      freq='M'),
            'expected_delivery_days': [part_meta['lead_time_days']] * len(part_df),
            'actual_delivery_days': [max(1, int(np.random.normal(part_meta['lead_time_days'], 
                                                               part_meta['lead_time_days']*0.3))) 
                                   for _ in range(len(part_df))]
        })
        
       
        delivery_data['delay_days'] = delivery_data['actual_delivery_days'] - delivery_data['expected_delivery_days']
        delivery_data['delay_ratio'] = delivery_data['delay_days'] / delivery_data['expected_delivery_days']
        delivery_data['is_delayed'] = delivery_data['delay_days'] > 0
        
    
        delay_rate = delivery_data['is_delayed'].mean() * 100
        avg_delay = delivery_data[delivery_data['is_delayed']]['delay_days'].mean()
        
        col1, col2 = st.columns(2)
        col1.metric("Delay Rate", f"{delay_rate:.1f}%")
        col2.metric("Average Delay (when delayed)", f"{avg_delay:.1f} days" if not np.isnan(avg_delay) else "0 days")
        
       
        fig_delay = go.Figure()
        
        fig_delay.add_trace(go.Scatter(
            x=delivery_data['order_date'],
            y=delivery_data['actual_delivery_days'],
            mode='lines+markers',
            name='Actual Delivery Days',
            line=dict(color='red', width=2)
        ))
        
        fig_delay.add_trace(go.Scatter(
            x=delivery_data['order_date'],
            y=delivery_data['expected_delivery_days'],
            mode='lines',
            name='Expected Lead Time',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        
        delayed_points = delivery_data[delivery_data['is_delayed']]
        fig_delay.add_trace(go.Scatter(
            x=delayed_points['order_date'],
            y=delayed_points['actual_delivery_days'],
            mode='markers',
            name='Delayed Delivery',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
        fig_delay.update_layout(
            title='Delivery Performance Over Time',
            xaxis_title='Order Date',
            yaxis_title='Delivery Time (days)',
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig_delay, use_container_width=True)
        
     
        delay_causes = pd.DataFrame({
            'cause': ['Supplier issues', 'Transportation delays', 'Quality inspections', 
                     'Documentation problems', 'Customs clearance'],
            'count': np.random.randint(1, 10, size=5),
            'impact': np.random.uniform(0.5, 2.0, size=5)
        }).sort_values('count', ascending=False)
        
        with st.expander("Delay Root Cause Analysis"):
            st.write("Common causes of delivery delays (simulated data):")
            st.dataframe(delay_causes)
            
            fig_causes = go.Figure()
            
            fig_causes.add_trace(go.Bar(
                x=delay_causes['cause'],
                y=delay_causes['count'],
                name='Frequency',
                marker_color='indianred'
            ))
            
            fig_causes.add_trace(go.Scatter(
                x=delay_causes['cause'],
                y=delay_causes['impact'],
                name='Impact (days)',
                mode='lines+markers',
                yaxis='y2',
                line=dict(color='royalblue', width=2)
            ))
            
            fig_causes.update_layout(
                title='Delay Causes Frequency and Impact',
                xaxis_title='Cause',
                yaxis_title='Frequency',
                yaxis2=dict(title='Average Impact (days)', overlaying='y', side='right'),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_causes, use_container_width=True)
        
  
        forecast_dates = pd.date_range(
            start=part_df["month"].iloc[-1] + pd.DateOffset(months=1),
            periods=forecast_months, 
            freq="MS"
        )
        
        forecast_table = pd.DataFrame({
            "Month": forecast_dates.strftime('%Y-%m'),
            "Forecast": all_forecasts,
            "Lower Bound": lower_bounds,
            "Upper Bound": upper_bounds,
            "Variability": [f"±{ub-lb:.1f}" for lb, ub in zip(lower_bounds, upper_bounds)]
        })
        
        st.dataframe(
            forecast_table.set_index("Month").style.format({
                "Forecast": "{:.1f}",
                "Lower Bound": "{:.1f}",
                "Upper Bound": "{:.1f}"
            }).background_gradient(subset=["Forecast"], cmap="Oranges")
        )

else:
    st.info("💡 Generate synthetic data or upload your own to get started.")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             caption="Industrial spare parts management")

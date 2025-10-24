# import pandas as pd
# import numpy as np
# import joblib
# import streamlit as st
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler # <-- Need this import for joblib
# from pulp import LpMaximize, LpProblem, LpVariable, value, LpStatus

# # ---------- 1. SET UP THE PAGE ----------
# st.set_page_config(page_title="Factory Optimization", layout="wide")
# st.title("Smart Factory: LSTM Forecast & PuLP Optimization")
# st.write("""
# This tool uses a trained LSTM model to forecast future demand, then runs a 
# Linear Programming model (PuLP) to find the most profitable production plan 
# based on your factory's real-world constraints.
# """)

# # ---------- 2. PARAMETERS & MODEL LOADING ----------
# # Use st.cache_resource to load models only once
# @st.cache_resource
# def load_models():
#     try:
#         model = load_model("forcasting_lstm_model.keras")
#         scaler = joblib.load("my_scaler.pkl")
#         return model, scaler
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         st.error("Please make sure 'forcasting_lstm_model.keras' and 'my_scaler.pkl' are in the same folder.")
#         return None, None

# model, scaler = load_models()
# if model is None:
#     st.stop()

# # Load the base data for seeding predictions
# @st.cache_data
# def load_seed_data():
#     try:
#         df = pd.read_csv("factory_demand.csv")
#         return df
#     except FileNotFoundError:
#         st.error("Error: 'factory_demand_with_patterns.csv' not found.")
#         st.stop()

# df_seed = load_seed_data()

# products = ['Gear', 'Shaft', 'Bearing']
# materials = ['Steel', 'Oil', 'Casing']
# n_steps = 60
# n_features = 3

# # ---------- 3. USER INPUT (The "Dashboard") ----------
# st.sidebar.header("Step 1: Get Forecast")
# target_date_input = st.sidebar.date_input("Select Target Forecast Date")
# run_forecast_btn = st.sidebar.button("Run Forecast")

# st.sidebar.header("Step 2: Define Factory (Daily Inputs)")

# with st.sidebar.expander("Economic Inputs (Prices & Costs)"):
#     selling_price_per_unit = {p: st.number_input(f"Selling Price for {p}", min_value=0.0, value=70.0, step=1.0) for p in products}
#     labor_cost_per_hour = st.number_input("Cost per Labor Hour", min_value=0.0, value=25.0, step=1.0)
#     material_cost = {m: st.number_input(f"Cost per unit of {m}", min_value=0.0, value=2.0, step=0.1) for m in materials}

# with st.sidebar.expander("Per-Unit Resource Needs (Factory Specs)"):
#     machine_hours = {p: st.number_input(f"{p} Machine Hrs/Unit", min_value=0.0, value=0.5, step=0.1) for p in products}
#     labor_hours = {p: st.number_input(f"{p} Labor Hrs/Unit", min_value=0.0, value=1.0, step=0.1) for p in products}
#     space_per_unit = {p: st.number_input(f"{p} Warehouse (m³)/Unit", min_value=0.0, value=0.5, step=0.1) for p in products}
#     material_needs = {p: {} for p in products}
#     for p in products:
#         st.subheader(f"Material Needs for {p}")
#         for m in materials:
#             material_needs[p][m] = st.number_input(f"{m} units for {p}", min_value=0.0, value=5.0, step=0.5, key=f"{p}_{m}")

# with st.sidebar.expander("Minimum Production & Resource Limits"):
#     low_bounds = {p: st.number_input(f"Min Production for {p}", min_value=0, value=5) for p in products}
#     max_machine_hours = st.number_input("Total Machine Hours Available", min_value=0, value=200)
#     max_labor_hours = st.number_input("Total Labor Hours Available", min_value=0, value=400)
#     available_materials = {m: st.number_input(f"Total {m} Available", min_value=0, value=2500) for m in materials}
#     max_warehouse_space = st.number_input("Total Warehouse Space (m³)", min_value=0, value=250)

# run_optimize_btn = st.sidebar.button("Run Production Optimization", type="primary")

# # ---------- 4. FORECASTING LOGIC ----------
# if run_forecast_btn:
#     st.session_state.forecast_ran = True
    
#     last_date = pd.to_datetime(df_seed['Date'].iloc[-1])
#     target_date = pd.to_datetime(target_date_input)
    
#     if target_date <= last_date:
#         st.error(f"Date must be after last historical date ({last_date.date()})")
#     else:
#         with st.spinner("Running LSTM model to forecast..."):
#             days_ahead = (target_date - last_date).days
            
#             last_known_data = df_seed[products].values[-n_steps:]
#             last_known_data_scaled = scaler.transform(last_known_data)
            
#             last_seq = last_known_data_scaled.copy()
#             for _ in range(days_ahead):
#                 input_seq = last_seq.reshape(1, n_steps, n_features)
#                 pred_scaled = model.predict(input_seq, verbose=0) 
#                 last_seq = np.vstack([last_seq[1:], pred_scaled]) # More efficient
            
#             final_pred_scaled = last_seq[-1].reshape(1, n_features)
#             final_pred_orig = scaler.inverse_transform(final_pred_scaled)
            
#             forecasted_demand = np.maximum(0, np.round(final_pred_orig)).astype(int)[0]
#             demand_dict = dict(zip(products, forecasted_demand))
            
#             # Store in session state to use in optimization
#             st.session_state.demand_dict = demand_dict
#             st.session_state.target_date = target_date
            
#             st.header(f"Forecast for {target_date.date()}")
#             st.table(pd.DataFrame.from_dict(demand_dict, orient='index', columns=['Forecasted Demand']))
#             st.info("Forecast complete. Now, adjust your factory inputs on the left and run the optimization.")

# # ---------- 5. OPTIMIZATION LOGIC ----------
# if run_optimize_btn:
#     if 'demand_dict' not in st.session_state:
#         st.error("Please run a forecast first (Step 1).")
#     else:
#         with st.spinner("Running PuLP optimizer..."):
#             demand_dict = st.session_state.demand_dict
#             target_date = st.session_state.target_date

#             # Decision variables:
#             x = {p: LpVariable(f"{p}_units", lowBound=low_bounds[p], upBound=demand_dict[p]) for p in products}
            
#             model_lp = LpProblem("Factory_Production_Plan", LpMaximize)
            
#             # Objective Function (from user inputs)
#             total_revenue = sum(selling_price_per_unit[p] * x[p] for p in products)
#             total_labor_cost = sum(labor_hours[p] * x[p] for p in products) * labor_cost_per_hour
#             total_material_cost = sum(
#                 material_needs[p][material] * x[p] * material_cost[material] 
#                 for p in products for material in materials
#             )
#             model_lp += (total_revenue - total_labor_cost - total_material_cost), "Total_Net_Profit"

#             # Constraints (from user inputs)
#             model_lp += sum(machine_hours[p]*x[p] for p in products) <= max_machine_hours, "Max_Machine_Hours"
#             model_lp += sum(labor_hours[p]*x[p] for p in products) <= max_labor_hours, "Max_Labor_Hours"
#             model_lp += sum(space_per_unit[p] * x[p] for p in products) <= max_warehouse_space, "Max_Warehouse_Space"
#             for material in materials:
#                 model_lp += sum(material_needs[p][material] * x[p] for p in products) <= available_materials[material], f"Max_{material}"

#             # Solve
#             model_lp.solve()
            
#             # --- 6. DISPLAY RESULTS (The Dashboard) ---
#             st.header(f"Optimized Production Plan for {target_date.date()}")
            
#             if LpStatus[model_lp.status] == 'Optimal':
#                 st.success(f"Optimal Solution Found! Maximum Net Profit = ${value(model_lp.objective):,.2f}")
                
#                 # --- Result 1: The Plan ---
#                 st.subheader("Production Plan")
#                 plan_data = {
#                     'Product': products,
#                     'Forecasted Demand': [demand_dict[p] for p in products],
#                     'Optimized Production': [int(x[p].value()) for p in products],
#                     'Min Production': [low_bounds[p] for p in products]
#                 }
#                 plan_df = pd.DataFrame(plan_data).set_index('Product')
#                 st.dataframe(plan_df)
                
#                 # --- Result 2: Visuals ---
#                 st.bar_chart(plan_df[['Forecasted Demand', 'Optimized Production']])
                
#                 # --- Result 3: Bottlenecks (Resource Utilization) ---
#                 st.subheader("Resource Utilization (Your Bottlenecks)")
#                 util_data = {
#                     'Resource': ['Machine Hours', 'Labor Hours', 'Warehouse Space'] + materials,
#                     'Used': [
#                         sum(machine_hours[p]*x[p].value() for p in products),
#                         sum(labor_hours[p]*x[p].value() for p in products),
#                         sum(space_per_unit[p]*x[p].value() for p in products),
#                     ] + [sum(material_needs[p][m]*x[p].value() for p in products) for m in materials],
#                     'Total': [max_machine_hours, max_labor_hours, max_warehouse_space] + [available_materials[m] for m in materials]
#                 }
#                 util_df = pd.DataFrame(util_data)
#                 util_df['Percentage'] = (util_df['Used'] / util_df['Total']).clip(0, 1) * 100
#                 st.dataframe(util_df.style.format({'Percentage': '{:.1f}%'}))
                
#                 # Plot utilization
#                 util_plot_df = util_df.set_index('Resource')
#                 st.bar_chart(util_plot_df['Percentage'])
                
#                 bottlenecks = util_df[util_df['Percentage'] == 100.0]['Resource'].tolist()
#                 if bottlenecks:
#                     st.warning(f"**Factory Bottlenecks Identified:** You ran out of {', '.join(bottlenecks)}!")
#                 else:
#                     st.info("No bottlenecks reached. You have surplus capacity.")
                    
#             elif LpStatus[model_lp.status] == 'Infeasible':
#                 st.error("Optimization Failed: Infeasible. This means it's impossible to meet your minimum production goals with your available resources. Try reducing the 'Min Production' or increasing resources.")
#             else:
#                 st.error(f"Optimization Failed. Status: {LpStatus[model_lp.status]}")

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler # <-- Need this import for joblib
from pulp import LpMaximize, LpProblem, LpVariable, value, LpStatus

# ---------- 1. SET UP THE PAGE ----------
st.set_page_config(page_title="Factory Optimization", layout="wide")
st.title("Smart Factory: LSTM Forecast & Simplex Optimization")
st.write("""
This tool uses a trained LSTM model to forecast future demand, then runs a 
Linear Programming model (PuLP) to find the most profitable production plan 
based on your factory's real-world constraints.
""")

# ---------- 2. PARAMETERS & MODEL LOADING ----------
# Use st.cache_resource to load models only once
@st.cache_resource
def load_models():
    try:
        model = load_model("forcasting_lstm_model.keras")
        scaler = joblib.load("my_scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please make sure 'forcasting_lstm_model.keras' and 'my_scaler.pkl' are in the same folder.")
        return None, None

model, scaler = load_models()
if model is None:
    st.stop()

# Load the base data for seeding predictions
@st.cache_data
def load_seed_data():
    try:
        df = pd.read_csv("factory_demand.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'factory_demand.csv' not found.")
        st.stop()

df_seed = load_seed_data()

products = ['Gear', 'Shaft', 'Bearing']
materials = ['Steel', 'Oil', 'Casing']
n_steps = 60
n_features = 3

# ---------- 3. USER INPUT (The "Dashboard") ----------
st.sidebar.header("Step 1: Get Forecast")
target_date_input = st.sidebar.date_input("Select Target Forecast Date")
run_forecast_btn = st.sidebar.button("Run Forecast")

st.sidebar.header("Step 2: Define Factory (Daily Inputs)")

with st.sidebar.expander("Economic Inputs (Prices & Costs)", expanded=True):
    selling_price_per_unit = {p: st.number_input(f"Selling Price for {p}", min_value=0.0, value=70.0, step=1.0) for p in products}
    labor_cost_per_hour = st.number_input("Cost per Labor Hour", min_value=0.0, value=25.0, step=1.0)
    material_cost = {m: st.number_input(f"Cost per unit of {m}", min_value=0.0, value=2.0, step=0.1) for m in materials}

with st.sidebar.expander("Per-Unit Resource Needs (Factory Specs)"):
    machine_hours = {p: st.number_input(f"{p} Machine Hrs/Unit", min_value=0.0, value=0.5, step=0.1) for p in products}
    labor_hours = {p: st.number_input(f"{p} Labor Hrs/Unit", min_value=0.0, value=1.0, step=0.1) for p in products}
    space_per_unit = {p: st.number_input(f"{p} Warehouse (m³)/Unit", min_value=0.0, value=0.5, step=0.1) for p in products}
    material_needs = {p: {} for p in products}
    for p in products:
        st.subheader(f"Material Needs for {p}")
        for m in materials:
            material_needs[p][m] = st.number_input(f"{m} units for {p}", min_value=0.0, value=5.0, step=0.5, key=f"{p}_{m}")

with st.sidebar.expander("Minimum Production & Resource Limits"):
    low_bounds = {p: st.number_input(f"Min Production for {p}", min_value=0, value=5) for p in products}
    max_machine_hours = st.number_input("Total Machine Hours Available", min_value=0, value=200)
    max_labor_hours = st.number_input("Total Labor Hours Available", min_value=0, value=400)
    available_materials = {m: st.number_input(f"Total {m} Available", min_value=0, value=2500) for m in materials}
    max_warehouse_space = st.number_input("Total Warehouse Space (m³)", min_value=0, value=250)

run_optimize_btn = st.sidebar.button("Run Production Optimization", type="primary")

# ---------- 4. FORECASTING LOGIC ----------
if run_forecast_btn:
    st.session_state.forecast_ran = True
    
    last_date = pd.to_datetime(df_seed['Date'].iloc[-1])
    target_date = pd.to_datetime(target_date_input)
    
    if target_date <= last_date:
        st.error(f"Date must be after last historical date ({last_date.date()})")
    else:
        with st.spinner("Running LSTM model to forecast..."):
            days_ahead = (target_date - last_date).days
            
            last_known_data = df_seed[products].values[-n_steps:]
            last_known_data_scaled = scaler.transform(last_known_data)
            
            last_seq = last_known_data_scaled.copy()
            for _ in range(days_ahead):
                input_seq = last_seq.reshape(1, n_steps, n_features)
                pred_scaled = model.predict(input_seq, verbose=0) 
                last_seq = np.vstack([last_seq[1:], pred_scaled]) # More efficient
            
            final_pred_scaled = last_seq[-1].reshape(1, n_features)
            final_pred_orig = scaler.inverse_transform(final_pred_scaled)
            
            forecasted_demand = np.maximum(0, np.round(final_pred_orig)).astype(int)[0]
            demand_dict = dict(zip(products, forecasted_demand))
            
            # Store in session state to use in optimization
            st.session_state.demand_dict = demand_dict
            st.session_state.target_date = target_date
            
            st.header(f"Forecast for {target_date.date()}")
            st.table(pd.DataFrame.from_dict(demand_dict, orient='index', columns=['Forecasted Demand']))
            st.info("Forecast complete. Now, adjust your factory inputs on the left and run the optimization.")

# ---------- 5. OPTIMIZATION LOGIC ----------
if run_optimize_btn:
    if 'demand_dict' not in st.session_state:
        st.error("Please run a forecast first (Step 1).")
    else:
        with st.spinner("Running PuLP optimizer..."):
            demand_dict = st.session_state.demand_dict
            target_date = st.session_state.target_date

            # Decision variables:
            x = {p: LpVariable(f"{p}_units", lowBound=low_bounds[p], upBound=demand_dict[p]) for p in products}
            
            model_lp = LpProblem("Factory_Production_Plan", LpMaximize)
            
            # Objective Function (from user inputs)
            total_revenue = sum(selling_price_per_unit[p] * x[p] for p in products)
            total_labor_cost = sum(labor_hours[p] * x[p] for p in products) * labor_cost_per_hour
            total_material_cost = sum(
                material_needs[p][material] * x[p] * material_cost[material] 
                for p in products for material in materials
            )
            model_lp += (total_revenue - total_labor_cost - total_material_cost), "Total_Net_Profit"

            # Constraints (from user inputs)
            model_lp += sum(machine_hours[p]*x[p] for p in products) <= max_machine_hours, "Max_Machine_Hours"
            model_lp += sum(labor_hours[p]*x[p] for p in products) <= max_labor_hours, "Max_Labor_Hours"
            model_lp += sum(space_per_unit[p] * x[p] for p in products) <= max_warehouse_space, "Max_Warehouse_Space"
            for material in materials:
                model_lp += sum(material_needs[p][material] * x[p] for p in products) <= available_materials[material], f"Max_{material}"

            # Solve
            model_lp.solve()
            
            # --- 6. DISPLAY RESULTS (The Dashboard) ---
            st.header(f"Optimized Production Plan for {target_date.date()}")
            
            if LpStatus[model_lp.status] == 'Optimal':
                st.success(f"Optimal Solution Found! Maximum Net Profit = ₹{value(model_lp.objective):,.2f}")
                
                # Split page into two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # --- Result 1: The Plan ---
                    st.subheader("Production Plan")
                    plan_data = {
                        'Product': products,
                        'Forecasted Demand': [demand_dict[p] for p in products],
                        'Optimized Production': [int(x[p].value()) for p in products],
                        'Min Production': [low_bounds[p]for p in products]
                    }
                    plan_df = pd.DataFrame(plan_data).set_index('Product')
                    st.dataframe(plan_df)
                    st.bar_chart(plan_df[['Forecasted Demand', 'Optimized Production']])
                
                with col2:
                    # --- Result 2: Bottlenecks (Resource Utilization) ---
                    st.subheader("Resource Utilization")
                    util_data = {
                        'Resource': ['Machine Hours', 'Labor Hours', 'Warehouse Space'] + materials,
                        'Used': [
                            sum(machine_hours[p]*x[p].value() for p in products),
                            sum(labor_hours[p]*x[p].value() for p in products),
                            sum(space_per_unit[p]*x[p].value() for p in products),
                        ] + [sum(material_needs[p][m]*x[p].value() for p in products) for m in materials],
                        'Total': [max_machine_hours, max_labor_hours, max_warehouse_space] + [available_materials[m] for m in materials]
                    }
                    util_df = pd.DataFrame(util_data)
                    util_df['Percentage'] = (util_df['Used'] / util_df['Total']).clip(0, 1) * 100
                    st.dataframe(util_df.style.format({'Percentage': '{:.1f}%'}))
                    
                    bottlenecks = util_df[util_df['Percentage'] == 100.0]['Resource'].tolist()
                    if bottlenecks:
                        st.warning(f"**Factory Bottlenecks Identified:** You ran out of {', '.join(bottlenecks)}!")
                    else:
                        st.info("No bottlenecks reached. You have surplus capacity.")
                
                
                # --- (NEW) Result 3: Actionable Insights (Shadow Prices) ---
                st.subheader("Actionable Insights (Sensitivity Analysis)")
                st.write("""
                This table shows the 'Shadow Price' for each resource. This is the **exact amount of extra profit** you will make if you get **one more unit** of that resource.
                """)
                
                insights = []
                
                # --- Labor Insight ---
                constraint_name = "Max_Labor_Hours"
                shadow_price_labor = model_lp.constraints[constraint_name].pi
                decision_labor = ""
                if shadow_price_labor > labor_cost_per_hour:
                    decision_labor = f"BUY/HIRE. Worth ${shadow_price_labor:.2f} (costs ${labor_cost_per_hour:.2f})"
                else:
                    decision_labor = f"DO NOT BUY. Worth ${shadow_price_labor:.2f} (costs ${labor_cost_per_hour:.2f})"
                
                insights.append({
                    'Resource': 'Labor Hour', 
                    'Actual Cost': labor_cost_per_hour, 
                    'Shadow Price (Value)': shadow_price_labor, 
                    'Decision': decision_labor
                })
                
                # --- Material Insights ---
                for m in materials:
                    constraint_name = f"Max_{m}"
                    shadow_price_mat = model_lp.constraints[constraint_name].pi
                    actual_cost_mat = material_cost[m]
                    decision_mat = ""
                    if shadow_price_mat > actual_cost_mat:
                        decision_mat = f"BUY MORE. Worth ₹{shadow_price_mat:.2f} (costs ₹{actual_cost_mat:.2f})"
                    else:
                        decision_mat = f"DO NOT BUY. Worth ₹{shadow_price_mat:.2f} (costs ₹{actual_cost_mat:.2f})"
                    
                    insights.append({
                        'Resource': m, 
                        'Actual Cost': actual_cost_mat, 
                        'Shadow Price (Value)': shadow_price_mat, 
                        'Decision': decision_mat
                    })
                
                # --- Other Resource Insights (cost is $0, so value is pure profit) ---
                shadow_price_machine = model_lp.constraints['Max_Machine_Hours'].pi
                insights.append({
                    'Resource': 'Machine Hour', 
                    'Actual Cost': 0.0, 
                    'Shadow Price (Value)': shadow_price_machine, 
                    'Decision': f"Worth ₹{shadow_price_machine:.2f} in profit"
                })
                
                shadow_price_space = model_lp.constraints['Max_Warehouse_Space'].pi
                insights.append({
                    'Resource': 'Warehouse Space (m³)', 
                    'Actual Cost': 0.0, 
                    'Shadow Price (Value)': shadow_price_space, 
                    'Decision': f"Worth ${shadow_price_space:.2f} in profit"
                })
                
                insights_df = pd.DataFrame(insights).set_index('Resource')
                st.dataframe(insights_df.style.format({'Actual Cost': '₹{:,.2f}', 'Shadow Price (Value)': '₹{:,.2f}'}))
                
                
            elif LpStatus[model_lp.status] == 'Infeasible':
                st.error("Optimization Failed: Infeasible. This means it's impossible to meet your minimum production goals with your available resources. Try reducing the 'Min Production' or increasing resources.")
            else:
                st.error(f"Optimization Failed. Status: {LpStatus[model_lp.status]}")




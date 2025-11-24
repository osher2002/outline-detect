import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import chi2
from scipy.linalg import pinv
from scipy.ndimage import gaussian_filter1d

# Page configuration
st.set_page_config(page_title="Outlier Detector", layout="wide")

st.title("Ultimate Outlier Detector")
st.markdown("Mahalanobis Analysis with Data Filtering, Adaptive Smoothing, and Custom Visualization.")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    with st.expander("Advanced Load Settings"):
        header_row_idx = st.number_input("Header Row Index", 0, 10, 0)
        encoding_option = st.selectbox("Encoding", ["Auto / UTF-8", "ISO-8859-1"])

    st.header("2. Analysis Config")

# --- Main Logic ---
if uploaded_file is not None:
    # 1. Load File
    try:
        enc = None if encoding_option == "Auto / UTF-8" else encoding_option.split()[0]
        df_raw = pd.read_csv(uploaded_file, encoding=enc, header=header_row_idx)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # --- DATA FILTERING SECTION ---
    st.markdown("---")
    with st.expander("Data Filtering and Preview", expanded=True):
        st.write("Filter your data before analysis:")
        
        # Column selection for filtering
        filter_cols = st.multiselect("Select columns to filter by:", df_raw.columns.tolist())
        
        df_filtered = df_raw.copy()
        
        # Dynamic Filters
        for col in filter_cols:
            # Numeric Filter
            if pd.api.types.is_numeric_dtype(df_filtered[col]):
                min_val = float(df_filtered[col].min())
                max_val = float(df_filtered[col].max())
                step = (max_val - min_val) / 100 if max_val != min_val else 1.0
                
                rng = st.slider(f"Range for: {col}", min_val, max_val, (min_val, max_val), step=step)
                df_filtered = df_filtered[df_filtered[col].between(rng[0], rng[1])]
            
            # Categorical Filter
            else:
                unique_vals = df_filtered[col].unique().tolist()
                selected_vals = st.multiselect(f"Values for: {col}", unique_vals, default=unique_vals)
                df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]
        
        st.info(f"Rows after filtering: {len(df_filtered)} (Original: {len(df_raw)})")
        st.dataframe(df_filtered.head(50), height=200)

    total_rows = len(df_filtered)
    if total_rows == 0:
        st.error("No data left after filtering.")
        st.stop()

    # --- Row Range Control ---
    st.sidebar.subheader("Select Row Range (Subset)")
    start_row, end_row = st.sidebar.slider(
        "Process rows:", 0, total_rows, (0, total_rows), 1
    )
    
    df = df_filtered.iloc[start_row:end_row].copy()
    st.sidebar.info(f"Active Rows: {len(df)}")

    # --- Parameters ---
    st.sidebar.subheader("Variables")
    all_cols = df.columns.tolist()
    potential = ['year', 'make', 'condition', 'odometer', 'mmr', 'sellingprice', 'mileage']
    default_cols = [c for c in all_cols if str(c).lower().strip() in potential]
    selected_cols = st.sidebar.multiselect("Select Model Variables (Min 2):", all_cols, default=default_cols)

    # Thresholds
    st.sidebar.markdown("---")
    st.sidebar.subheader("Threshold Settings")
    global_percent = st.slider("Global Sensitivity (%)", 80.0, 99.9, 95.0, 0.1)
    
    with st.expander("Smoothing Settings"):
        sigma_val = st.slider("Smoothness Factor (Sigma)", 1, 200, 50, 5, help="Higher = Smoother line.")
        quantile_target = st.slider("Target Quantile (%)", 80, 99, 95, 1)

    auto_clean = st.sidebar.checkbox("Auto-Clean Text", value=True)

    # --- Execution ---
    if st.sidebar.button("Run Analysis", type="primary"):
        if len(selected_cols) < 2:
            st.error("Select at least 2 variables.")
        else:
            with st.spinner('Calculating Statistics...'):
                # Prep
                df_work = df[selected_cols].copy()
                if auto_clean:
                    for c in df_work.select_dtypes(include=['object']).columns:
                        df_work[c] = df_work[c].astype(str).str.lower().str.strip()
                df_work.dropna(inplace=True)
                
                if len(df_work) < 10:
                    st.error("Not enough data points.")
                    st.stop()

                # Mahalanobis Calculation
                try:
                    df_encoded = pd.get_dummies(df_work, dtype=int)
                    data = df_encoded.values
                    mu = np.mean(data, axis=0)
                    cov = np.cov(data.T)
                    inv_cov = pinv(cov)
                    distances = df_encoded.apply(lambda row: distance.mahalanobis(row.values, mu, inv_cov), axis=1)
                except Exception as e:
                    st.error(f"Calculation Error: {e}")
                    st.stop()
                
                # Thresholds
                dims = df_encoded.shape[1]
                global_thresh = np.sqrt(chi2.ppf(global_percent/100.0, dims))
                
                # Results
                df_res = df.loc[df_work.index].copy()
                df_res['Mahalanobis_Dist'] = distances
                df_res['Status_Global'] = np.where(df_res['Mahalanobis_Dist'] > global_thresh, 'Outlier', 'Normal')

                # Save State
                numeric_cols = [c for c in selected_cols if pd.api.types.is_numeric_dtype(df_res[c])]
                st.session_state['results'] = df_res
                st.session_state['numeric_cols'] = numeric_cols
                st.session_state['global_thresh'] = global_thresh
                st.session_state['sigma_val'] = sigma_val 
                st.session_state['quantile_target'] = quantile_target

    # --- Visualization ---
    if 'results' in st.session_state:
        res = st.session_state['results']
        numeric_cols = st.session_state['numeric_cols']
        global_thresh = st.session_state['global_thresh']
        sigma_val = st.session_state['sigma_val']
        q_target = st.session_state['quantile_target'] / 100.0
        
        st.markdown("---")
        st.subheader("Visualization")
        
        # Visualization Menu
        plot_type = st.radio(
            "Select Visualization Mode:",
            [
                "Adaptive Boundary (Smooth Line)",
                "Custom Pair Scatter (X vs Y)", 
                "Distance vs. Variables (N-Graphs)", 
                "Pair Plot (NxN Matrix)"             
            ],
            horizontal=True
        )
        
        limit_plot = st.slider("Max Points to Plot:", 100, len(res), min(5000, len(res)), 100)
        
        if len(res) > limit_plot:
            df_viz = res.sample(limit_plot, random_state=42)
            st.caption(f"Showing sample of {limit_plot} points.")
        else:
            df_viz = res

        # 1. Adaptive Smoothing
        if plot_type == "Adaptive Boundary (Smooth Line)":
            if len(numeric_cols) > 0:
                col1, col2 = st.columns([1, 3])
                with col1:
                    sort_col = st.selectbox("Sort By (X-Axis):", numeric_cols, index=0)
                
                with col2:
                    df_sorted = res.sort_values(by=sort_col).copy()
                    
                    # Rolling & Gaussian Smoothing
                    rolling_window = max(10, int(len(df_sorted) * 0.05))
                    raw_line = df_sorted['Mahalanobis_Dist'].rolling(window=rolling_window, center=True).quantile(q_target)
                    raw_line = raw_line.fillna(method='bfill').fillna(method='ffill')
                    
                    try:
                        smoothed = gaussian_filter1d(raw_line.values, sigma=sigma_val)
                    except:
                        smoothed = raw_line.values
                    
                    df_sorted['Smooth_Threshold'] = smoothed
                    df_sorted['Status_Adaptive'] = np.where(df_sorted['Mahalanobis_Dist'] > df_sorted['Smooth_Threshold'], 'Outlier', 'Normal')

                    # Sample for plot
                    if len(df_sorted) > limit_plot:
                        df_viz_adapt = df_sorted.sample(limit_plot, random_state=42).sort_values(by=sort_col)
                    else:
                        df_viz_adapt = df_sorted

                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.scatterplot(data=df_viz_adapt, x=sort_col, y='Mahalanobis_Dist', hue='Status_Adaptive', 
                                    palette={'Normal': '#3498db', 'Outlier': '#e74c3c'}, alpha=0.5, s=20, ax=ax)
                    
                    ax.plot(df_sorted[sort_col].values, df_sorted['Smooth_Threshold'].values, 
                            color='black', linewidth=3, label='Adaptive Boundary')
                    
                    ax.set_title(f"Adaptive Detection: {sort_col}")
                    ax.legend()
                    st.pyplot(fig)
                    
                    csv = df_sorted.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Adaptive Results", csv, "adaptive_results.csv", "text/csv")
            else:
                st.warning("No numeric variables.")

        # 2. Custom Pair Scatter
        elif plot_type == "Custom Pair Scatter (X vs Y)":
            if len(numeric_cols) >= 2:
                c1, c2 = st.columns(2)
                x_var = c1.selectbox("Select X Axis:", numeric_cols, index=0)
                y_var = c2.selectbox("Select Y Axis:", numeric_cols, index=1)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df_viz, x=x_var, y=y_var, hue='Status_Global', 
                                palette={'Normal': '#3498db', 'Outlier': '#e74c3c'}, alpha=0.6, s=30, ax=ax)
                
                ax.set_title(f"Scatter: {x_var} vs {y_var}")
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric variables.")

        # 3. N Graphs
        elif plot_type == "Distance vs. Variables (N-Graphs)":
            if numeric_cols:
                fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4 * len(numeric_cols)), constrained_layout=True)
                if len(numeric_cols) == 1: axes = [axes]
                
                for i, col in enumerate(numeric_cols):
                    ax = axes[i]
                    sns.scatterplot(data=df_viz, x=col, y='Mahalanobis_Dist', hue='Status_Global', 
                                    palette={'Normal': '#3498db', 'Outlier': '#e74c3c'}, alpha=0.6, ax=ax)
                    ax.axhline(global_thresh, color='black', linestyle='--', label='Global Threshold')
                    ax.set_ylabel("Distance")
                    if i == 0: ax.legend()
                    else: ax.get_legend().remove()
                st.pyplot(fig)

        # 4. Pair Plot
        elif plot_type == "Pair Plot (NxN Matrix)":
            if numeric_cols:
                st.info("Generating Pair Plot...")
                plot_cols = numeric_cols + ['Mahalanobis_Dist']
                fig = sns.pairplot(df_viz, vars=plot_cols, hue='Status_Global',
                                   palette={'Normal': '#3498db', 'Outlier': '#e74c3c'},
                                   plot_kws={'alpha': 0.5, 's': 15})
                st.pyplot(fig)

        if plot_type != "Adaptive Boundary (Smooth Line)":
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button("Download Standard Results", csv, "standard_results.csv", "text/csv")

else:
    st.info("Upload CSV to start.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import chi2
from scipy.linalg import pinv
from scipy.ndimage import gaussian_filter1d
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Outlier Detector Pro", layout="wide")

st.title("Ultimate Outlier Detector")
st.markdown("Mahalanobis Analysis with Data Filtering, Adaptive Smoothing, Clustering, and Custom Visualization.")

# --- Helper Function for Grid (Matplotlib) ---
def add_fine_grid(ax):
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)

# --- Sidebar ---
with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    with st.expander("Advanced Load Settings"):
        header_row_idx = st.number_input("Header Row Index", 0, 10, 0)
        encoding_option = st.selectbox("Encoding", ["Auto / UTF-8", "ISO-8859-1", "cp1252"])

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
        
        filter_cols = st.multiselect("Select columns to filter by:", df_raw.columns.tolist())
        df_filtered = df_raw.copy()
        
        for col in filter_cols:
            if pd.api.types.is_numeric_dtype(df_filtered[col]):
                min_val = float(df_filtered[col].min())
                max_val = float(df_filtered[col].max())
                
                if min_val < max_val:
                    step = (max_val - min_val) / 100
                    rng = st.slider(f"Range for: {col}", min_val, max_val, (min_val, max_val), step=step)
                    df_filtered = df_filtered[df_filtered[col].between(rng[0], rng[1])]
            else:
                unique_vals = df_filtered[col].unique().tolist()
                selected_vals = st.multiselect(f"Values for: {col}", unique_vals, default=unique_vals)
                if selected_vals:
                    df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]
        
        st.info(f"Rows after filtering: {len(df_filtered)} (Original: {len(df_raw)})")
        st.dataframe(df_filtered.head(50), height=150)

    total_rows = len(df_filtered)
    if total_rows == 0:
        st.error("No data left after filtering.")
        st.stop()

    # --- Row Range Control ---
    st.sidebar.subheader("Select Row Range (Subset)")
    if total_rows > 1:
        start_row, end_row = st.sidebar.slider("Process rows:", 0, total_rows, (0, total_rows), 1)
    else:
        start_row, end_row = 0, total_rows
    
    df = df_filtered.iloc[start_row:end_row].copy()
    st.sidebar.info(f"Active Rows: {len(df)}")

    # --- Parameters ---
    st.sidebar.subheader("Variables")
    all_cols = df.columns.tolist()
    potential = ['year', 'make', 'condition', 'odometer', 'mmr', 'sellingprice', 'mileage', 'price', 'x1', 'x2', 'x3', 'x4']
    default_cols = [c for c in all_cols if str(c).lower().strip() in potential]
    selected_cols = st.sidebar.multiselect("Select Model Variables (Min 2):", all_cols, default=default_cols)

    # Thresholds
    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    global_percent = st.slider("Global Sensitivity (%)", 80.0, 99.9, 95.0, 0.1)
    
    with st.expander("Smoothing Settings"):
        sigma_val = st.slider("Smoothness Factor (Sigma)", 1, 200, 50, 5)
        quantile_target = st.slider("Target Quantile (%)", 80, 99, 95, 1)

    # --- FEATURES CONFIG ---
    st.sidebar.subheader("Advanced Features")
    
    # Missing Data
    missing_strategy = st.sidebar.selectbox("Handle Missing Values:", ["Drop Rows", "Impute Mean", "Impute Median"])
    
    # Log Transform
    use_log = st.sidebar.checkbox("Log-Transform (Skewed Data)", value=False)
    
    # Clustering
    use_clustering = st.sidebar.checkbox("Enable Clustering", value=False)
    cluster_algo = "K-Means"
    if use_clustering:
        cluster_algo = st.sidebar.radio("Algorithm:", ["K-Means", "DBSCAN"])
        if cluster_algo == "K-Means":
            k_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
        else:
            eps_val = st.sidebar.slider("Epsilon (Distance)", 0.1, 5.0, 0.5, 0.1)
            min_samples_val = st.sidebar.slider("Min Samples", 2, 20, 5)

    auto_clean = st.sidebar.checkbox("Auto-Clean Text", value=True)

    # --- EXECUTION ---
    if st.sidebar.button("Run Analysis", type="primary"):
        if len(selected_cols) < 2:
            st.error("Select at least 2 numeric variables.")
        else:
            with st.spinner('Calculating Statistics...'):
                # 1. Prep
                df_work = df[selected_cols].copy()
                
                # Auto Clean
                if auto_clean:
                    for c in df_work.select_dtypes(include=['object']).columns:
                        df_work[c] = df_work[c].astype(str).str.lower().str.strip()
                
                # Missing Data Handling
                if missing_strategy == "Drop Rows":
                    df_work.dropna(inplace=True)
                else:
                    strategy = "mean" if "Mean" in missing_strategy else "median"
                    num_cols_work = df_work.select_dtypes(include=np.number).columns
                    if len(num_cols_work) > 0:
                        imputer = SimpleImputer(strategy=strategy)
                        df_work[num_cols_work] = imputer.fit_transform(df_work[num_cols_work])
                    df_work.dropna(inplace=True)

                if len(df_work) < 5:
                    st.error("Not enough data points.")
                    st.stop()

                # Log Transform
                if use_log:
                    num_cols_log = df_work.select_dtypes(include=np.number).columns
                    for c in num_cols_log:
                        if (df_work[c] > 0).all():
                            df_work[c] = np.log1p(df_work[c])

                # Mahalanobis
                try:
                    df_encoded = pd.get_dummies(df_work, drop_first=True, dtype=int)
                    data = df_encoded.values
                    mu = np.mean(data, axis=0)
                    cov = np.cov(data.T)
                    inv_cov = pinv(cov)
                    distances = df_encoded.apply(lambda row: distance.mahalanobis(row.values, mu, inv_cov), axis=1)
                except Exception as e:
                    st.error(f"Calculation Error: {e}")
                    st.stop()
                
                dims = df_encoded.shape[1]
                global_thresh = np.sqrt(chi2.ppf(global_percent/100.0, dims))
                
                # Results DataFrame
                df_res = df.loc[df_work.index].copy()
                df_res['Mahalanobis_Dist'] = distances
                df_res['Status_Global'] = np.where(df_res['Mahalanobis_Dist'] > global_thresh, 'Outlier', 'Normal')
                
                # Clustering Logic
                if use_clustering:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df_encoded)
                    
                    if cluster_algo == "K-Means":
                        model = KMeans(n_clusters=k_clusters, random_state=42)
                        labels = model.fit_predict(scaled_data)
                    else: # DBSCAN
                        model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
                        labels = model.fit_predict(scaled_data)
                    
                    df_res['Cluster'] = labels.astype(str)
                    if cluster_algo == "DBSCAN":
                        df_res['Cluster'] = df_res['Cluster'].replace('-1', 'Noise')

                # Save Session State
                st.session_state['results'] = df_res
                st.session_state['numeric_cols'] = [c for c in selected_cols if pd.api.types.is_numeric_dtype(df_res[c])]
                st.session_state['cat_cols'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
                st.session_state['global_thresh'] = global_thresh
                st.session_state['sigma_val'] = sigma_val 
                st.session_state['quantile_target'] = quantile_target
                st.session_state['has_clusters'] = use_clustering

    # --- VISUALIZATION ---
    if 'results' in st.session_state:
        res = st.session_state['results']
        numeric_cols = st.session_state['numeric_cols']
        cat_cols = st.session_state['cat_cols']
        global_thresh = st.session_state['global_thresh']
        sigma_val = st.session_state['sigma_val']
        q_target = st.session_state['quantile_target'] / 100.0
        
        st.markdown("---")
        st.subheader("Visualization Console")

        c1, c2, c3 = st.columns(3)
        
        # Color Selection
        color_opts = ["Status_Global"]
        if 'Cluster' in res.columns: color_opts.insert(0, "Cluster")
        color_opts += cat_cols
        
        color_by = c1.selectbox("Color Points By:", color_opts)
        
        # Plot Type Selection
        plot_type = c2.radio(
            "Select Visualization Mode:",
            [
                "Group Analysis (K-Groups)",
                "Adaptive Boundary (Smooth Line)",
                "Custom Pair Scatter (X vs Y)", 
                "Distance vs. Variables (N-Graphs)", 
                "Pair Plot (NxN Matrix)",
                "Outlier Contribution Analysis",
                "3D Scatter View"
            ],
            horizontal=True
        )
        
        # Sampling Slider
        total_points = len(res)
        if total_points > 100:
            limit_plot = c3.slider("Max Points to Plot:", 100, total_points, min(5000, total_points), 100)
        else:
            limit_plot = total_points
        
        # Data Subset
        if len(res) > limit_plot:
            df_viz_base = res.sample(limit_plot, random_state=42)
            st.caption(f"Showing sample of {limit_plot} points.")
        else:
            df_viz_base = res
            st.caption(f"Showing all {total_points} points.")

        # Define Palette
        custom_palette = None
        if color_by == "Status_Global":
            custom_palette = {'Normal': 'blue', 'Outlier': 'red'}
        elif color_by == "Cluster" and 'Noise' in res['Cluster'].unique():
             unique_clusters = sorted(res['Cluster'].unique())
             pal = sns.color_palette("tab10", len(unique_clusters)).as_hex()
             custom_palette = {grp: col for grp, col in zip(unique_clusters, pal)}
             if 'Noise' in custom_palette:
                 custom_palette['Noise'] = '#000000'

        # -------------------------------------------------------
        # 1. GROUP ANALYSIS (K-GROUPS)
        # -------------------------------------------------------
        if plot_type == "Group Analysis (K-Groups)":
            default_grp = "Cluster" if 'Cluster' in res.columns else (cat_cols[0] if cat_cols else None)
            
            if not default_grp and not cat_cols:
                st.warning("No grouping columns found.")
            else:
                gc1, gc2, gc3 = st.columns(3)
                
                grp_list = []
                if 'Cluster' in res.columns: grp_list.append('Cluster')
                grp_list += cat_cols
                
                group_col = gc1.selectbox("Group By:", grp_list, index=0)
                
                uniqs = res[group_col].unique()
                sel_grps = gc2.multiselect("Select Groups:", uniqs, default=uniqs[:3] if len(uniqs)>0 else None)
                x_axis = gc3.selectbox("X Axis Variable:", numeric_cols, index=0)

                if sel_grps:
                    df_g = res[res[group_col].isin(sel_grps)]
                    if len(df_g) > limit_plot: df_g = df_g.sample(limit_plot, random_state=42)
                        
                    fig = px.scatter(df_g, x=x_axis, y='Mahalanobis_Dist', 
                                     color=group_col, symbol=group_col,
                                     title=f"Group Comparison: {group_col}")
                    
                    fig.add_hline(y=global_thresh, line_dash="dash", line_color="black", annotation_text="Threshold")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least one group.")

        # -------------------------------------------------------
        # 2. ADAPTIVE SMOOTHING
        # -------------------------------------------------------
        elif plot_type == "Adaptive Boundary (Smooth Line)":
            if len(numeric_cols) > 0:
                xc, yc = st.columns([1, 3])
                sort_col = xc.selectbox("Sort By (X-Axis):", numeric_cols, index=0)
                
                with yc:
                    df_sorted = res.sort_values(by=sort_col).copy()
                    
                    min_win = max(2, int(len(df_sorted) * 0.05))
                    rolling = max(5, min_win)
                    
                    raw_line = df_sorted['Mahalanobis_Dist'].rolling(window=rolling, center=True).quantile(q_target)
                    raw_line = raw_line.fillna(method='bfill').fillna(method='ffill')
                    
                    try:
                        smoothed = gaussian_filter1d(raw_line.values, sigma=sigma_val)
                    except:
                        smoothed = raw_line.values
                    
                    df_sorted['Smooth_Threshold'] = smoothed
                    df_sorted['Status_Adaptive'] = np.where(df_sorted['Mahalanobis_Dist'] > df_sorted['Smooth_Threshold'], 'Outlier', 'Normal')

                    if len(df_sorted) > limit_plot:
                        df_viz_ad = df_sorted.sample(limit_plot, random_state=42).sort_values(by=sort_col)
                    else:
                        df_viz_ad = df_sorted

                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    hue = 'Status_Adaptive' if color_by == "Status_Global" else color_by
                    pal = {'Normal': 'blue', 'Outlier': 'red'} if hue == 'Status_Adaptive' else custom_palette

                    sns.scatterplot(data=df_viz_ad, x=sort_col, y='Mahalanobis_Dist', 
                                    hue=hue, palette=pal, alpha=0.6, s=30, ax=ax)
                    
                    ax.plot(df_sorted[sort_col].values, df_sorted['Smooth_Threshold'].values, 
                            color='black', linewidth=3, label='Adaptive Boundary')
                    
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                    add_fine_grid(ax)
                    st.pyplot(fig)
            else:
                st.warning("No numeric variables.")

        # -------------------------------------------------------
        # 3. CUSTOM SCATTER
        # -------------------------------------------------------
        elif plot_type == "Custom Pair Scatter (X vs Y)":
            if len(numeric_cols) >= 2:
                c1, c2 = st.columns(2)
                x_var = c1.selectbox("X Axis:", numeric_cols, index=0)
                y_var = c2.selectbox("Y Axis:", numeric_cols, index=1)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df_viz_base, x=x_var, y=y_var, 
                                hue=color_by, palette=custom_palette, alpha=0.6, s=30, ax=ax)
                
                ax.set_title(f"Scatter: {x_var} vs {y_var}")
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                add_fine_grid(ax)
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric variables.")

        # -------------------------------------------------------
        # 4. N GRAPHS
        # -------------------------------------------------------
        elif plot_type == "Distance vs. Variables (N-Graphs)":
            if numeric_cols:
                fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4 * len(numeric_cols)), constrained_layout=True)
                if len(numeric_cols) == 1: axes = [axes]
                
                for i, col in enumerate(numeric_cols):
                    ax = axes[i]
                    sns.scatterplot(data=df_viz_base, x=col, y='Mahalanobis_Dist', 
                                    hue=color_by, palette=custom_palette, alpha=0.6, ax=ax)
                    ax.axhline(global_thresh, color='black', linestyle='--', label='Global Threshold')
                    ax.set_ylabel("Distance")
                    
                    if i == 0: ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                    else: 
                        if ax.get_legend(): ax.get_legend().remove()
                    add_fine_grid(ax)
                st.pyplot(fig)

        # -------------------------------------------------------
        # 5. PAIR PLOT
        # -------------------------------------------------------
        elif plot_type == "Pair Plot (NxN Matrix)":
            if numeric_cols:
                st.info("Generating Pair Plot...")
                plot_cols = numeric_cols + ['Mahalanobis_Dist']
                
                if len(df_viz_base) > 1000:
                    df_pair = df_viz_base.sample(1000, random_state=42)
                else:
                    df_pair = df_viz_base

                g = sns.pairplot(df_pair, vars=plot_cols, hue=color_by, palette=custom_palette, plot_kws={'alpha': 0.6, 's': 20})
                for ax in g.axes.flatten():
                    if ax is not None: add_fine_grid(ax)
                st.pyplot(g.fig)

        # -------------------------------------------------------
        # 6. CONTRIBUTION ANALYSIS
        # -------------------------------------------------------
        elif plot_type == "Outlier Contribution Analysis":
            outs = res[res['Status_Global'] == 'Outlier'].sort_values('Mahalanobis_Dist', ascending=False)
            if not outs.empty:
                st.write("Top Extreme Outliers:")
                st.dataframe(outs.head(5))
                
                top_row = outs.iloc[0]
                st.markdown(f"#### Driver Analysis for Index: {top_row.name}")
                
                z_scores = {}
                for c in numeric_cols:
                    m = res[c].mean()
                    s = res[c].std()
                    z_scores[c] = (top_row[c] - m) / s if s > 0 else 0
                
                df_z = pd.DataFrame(list(z_scores.items()), columns=['Var', 'Z'])
                df_z['Abs'] = df_z['Z'].abs()
                df_z = df_z.sort_values('Abs', ascending=False)
                
                fig = px.bar(df_z, x='Z', y='Var', orientation='h', color='Z', color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No outliers found.")

        # -------------------------------------------------------
        # 7. 3D SCATTER
        # -------------------------------------------------------
        elif plot_type == "3D Scatter View":
            if len(numeric_cols) >= 3:
                c3a, c3b, c3c = st.columns(3)
                x3 = c3a.selectbox("X", numeric_cols, index=0)
                y3 = c3b.selectbox("Y", numeric_cols, index=1)
                z3 = c3c.selectbox("Z", numeric_cols, index=2)
                
                # Plotly for 3D is best
                fig = px.scatter_3d(df_viz_base, x=x3, y=y3, z=z3, color=color_by, 
                                    color_discrete_map=custom_palette, opacity=0.7, title="3D View")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need 3+ numeric variables.")

        # --- DOWNLOAD ---
        st.markdown("---")
        csv = res.to_csv(index=False).encode('utf-8')
        st.download_button("Download Standard Results", csv, "standard_results.csv", "text/csv")

else:
    st.info("Upload CSV to start.")

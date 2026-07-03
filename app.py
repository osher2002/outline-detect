import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import (chi2, shapiro, kstest, anderson, jarque_bera,
                         levene, bartlett, f_oneway, ttest_ind, ttest_rel,
                         mannwhitneyu, kruskal, spearmanr, kendalltau,
                         zscore, probplot)
from scipy.linalg import pinv
from scipy.ndimage import gaussian_filter1d
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error, explained_variance_score)
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

def apply_theme_css():
    if st.session_state['theme'] == 'dark':
        st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        [data-testid="stSidebar"] { background-color: #1a1d21; }
        h1, h2, h3, h4, h5, h6, .stMarkdown, p, label, div { color: #fafafa !important; }
        [data-testid="stMetric"] { background-color: #1a1d21; border: 1px solid #333; border-radius: 5px; padding: 10px; }
        [data-testid="stMetricLabel"] { color: #a0a0a0 !important; }
        [data-testid="stMetricValue"] { color: #fafafa !important; }
        [data-testid="stMetricDelta"] { color: #fafafa !important; }
        .stButton > button { background-color: #2d333b; color: #fafafa; border: 1px solid #4a5568; }
        .stButton > button:hover { background-color: #4a5568; border-color: #718096; }
        .stSelectbox > div > div, .stMultiselect > div > div { background-color: #2d333b; color: #fafafa; }
        .stTextInput > div > div > input { background-color: #2d333b; color: #fafafa; }
        .streamlit-expanderHeader { background-color: #1a1d21; color: #fafafa; }
        .streamlit-expanderContent { background-color: #0e1117; color: #fafafa; }
        .stDataFrame { background-color: #1a1d21; }
        .stAlert { background-color: #1a1d21; color: #fafafa; border: 1px solid #333; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #ffffff; color: #262730; }
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
        [data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 5px; padding: 10px; }
        .stButton > button { background-color: #ffffff; color: #262730; border: 1px solid #d0d0d0; }
        .stButton > button:hover { background-color: #f0f2f6; }
        </style>
        """, unsafe_allow_html=True)

apply_theme_css()
st.set_page_config(page_title="Outlier Detector Pro", layout="wide", page_icon="🔍")

with st.sidebar:
    st.header("🎨 Appearance")
    is_dark = st.toggle("🌙 Dark Mode", value=(st.session_state['theme'] == 'dark'))
    if is_dark and st.session_state['theme'] != 'dark':
        st.session_state['theme'] = 'dark'
        st.rerun()
    elif not is_dark and st.session_state['theme'] != 'light':
        st.session_state['theme'] = 'light'
        st.rerun()
    st.divider()
    st.header("📂 1. Data Input")
    uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx", "xls"], help="Supports CSV and Excel formats")
    with st.expander("⚙️ Advanced Load Settings"):
        header_row_idx = st.number_input("Header Row Index", 0, 10, 0)
        encoding_option = st.selectbox("Encoding (CSV only)", ["Auto / UTF-8", "ISO-8859-1", "cp1252"])
        excel_sheet = st.text_input("Sheet Name (Excel)", value="", help="Leave empty for first sheet")
        skip_rows_excel = st.number_input("Skip Rows (Excel)", 0, 20, 0)
    st.divider()
    st.header("⚙️ 2. Analysis Configuration")

def get_theme_colors():
    if st.session_state['theme'] == 'dark':
        return {'bg': '#0e1117', 'text': '#fafafa', 'normal': '#1f77b4', 'outlier': '#d62728'}
    return {'bg': '#ffffff', 'text': '#262730', 'normal': '#1f77b4', 'outlier': '#d62728'}

def fig_to_bytes_matplotlib(fig, format='png', dpi=300):
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf

def fig_to_bytes_plotly(fig, format='png', scale=2):
    try:
        return BytesIO(fig.to_image(format=format, scale=scale))
    except Exception as e:
        st.warning(f"Plotly export requires kaleido: {e}")
        return None

TOOLTIPS = {
    "Log Transform": "Logarithmic transformation (log1p) that normalizes distributions with strong positive skew. Reduces the influence of extreme values - critical for Mahalanobis Distance.",
    "K-Means": "Partitions data into K groups by minimizing within-cluster variance. Requires pre-specification of K.",
    "DBSCAN": "Density-Based Spatial Clustering. Automatically identifies clusters and marks 'noisy' points. Excellent for structural outlier detection.",
    "Mahalanobis Distance": "Multivariate distance metric accounting for correlations. Identifies outliers from rare combinations of values.",
    "Global Sensitivity": "Percentile of the Chi-Square distribution for the global outlier threshold. 95% = top 5% flagged as outliers.",
    "Smoothing Sigma": "Parameter σ of Gaussian filtering. Higher = smoother boundary; Lower = more responsive to local changes.",
    "Adaptive Boundary": "Dynamic outlier boundary varying by X-axis values. Useful for heteroscedastic data (e.g., price residuals).",
    "Target Quantile": "Percentile of distances used for the adaptive boundary line. 95% = line passes above 95% of points in each window.",
    "Impute": "Strategy for missing values: Mean is sensitive to outliers, Median is more robust. Drop removes rows.",
    "Auto-Clean Text": "Cleans text (lowercase, strip) to prevent duplicate categories in one-hot encoding."
}

if uploaded_file is not None:
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            enc = None if encoding_option == "Auto / UTF-8" else encoding_option.split()[0]
            df_raw = pd.read_csv(uploaded_file, encoding=enc, header=header_row_idx)
        elif filename.endswith(('.xlsx', '.xls')):
            sheet = excel_sheet if excel_sheet.strip() else 0
            skip = range(1, skip_rows_excel + 1) if skip_rows_excel > 0 else None
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, header=header_row_idx, skiprows=skip)
        else:
            st.error("❌ Unsupported file type."); st.stop()
        if df_raw.empty or len(df_raw.columns) < 2:
            st.error("❌ File is empty or has insufficient columns."); st.stop()
        st.success(f"✅ Loaded {filename.upper()}: {len(df_raw):,} rows, {len(df_raw.columns)} cols")
    except Exception as e:
        st.error(f"❌ Error loading file: {type(e).__name__}: {str(e)}"); st.stop()

    st.markdown("---")
    with st.expander("🔎 Data Filtering and Preview", expanded=True):
        filter_cols = st.multiselect("Select columns to filter by:", df_raw.columns.tolist())
        df_filtered = df_raw.copy()
        for col in filter_cols:
            if pd.api.types.is_numeric_dtype(df_filtered[col]):
                rng = st.slider(f"Range for: {col}", float(df_filtered[col].min()), float(df_filtered[col].max()), (float(df_filtered[col].min()), float(df_filtered[col].max())))
                df_filtered = df_filtered[df_filtered[col].between(rng[0], rng[1])]
            else:
                sel = st.multiselect(f"Values for: {col}", df_filtered[col].unique().tolist(), default=df_filtered[col].unique().tolist()[:10])
                if sel: df_filtered = df_filtered[df_filtered[col].isin(sel)]
        st.info(f"📊 Rows after filtering: **{len(df_filtered):,}**")
        st.dataframe(df_filtered.head(50), height=150)

    if len(df_filtered) == 0: st.error("No data left after filtering."); st.stop()

    st.sidebar.subheader("📏 Row Range")
    start_row, end_row = st.sidebar.slider("Process rows:", 0, len(df_filtered), (0, len(df_filtered)), 1)
    df = df_filtered.iloc[start_row:end_row].copy()
    st.sidebar.info(f"Active Rows: **{len(df):,}**")

    st.sidebar.subheader("🎯 Variables")
    all_cols = df.columns.tolist()
    potential = ['year', 'make', 'condition', 'odometer', 'mmr', 'sellingprice', 'mileage', 'price', 'selling_price', 'predicted_price', 'quality_hybrid', 'quality_heuristic', 'quality_residual', 'car_age', 'manufacturer_reliability', 'residuals']
    default_cols = [c for c in all_cols if str(c).lower().strip() in [p.lower() for p in potential]]
    selected_cols = st.sidebar.multiselect("Select Model Variables (Min 2):", all_cols, default=default_cols)

    st.sidebar.markdown("---")
    st.sidebar.subheader("️ Settings")
    global_percent = st.sidebar.slider("Global Sensitivity (%)", 80.0, 99.9, 95.0, 0.1, help=TOOLTIPS["Global Sensitivity"])
    with st.sidebar.expander("📈 Smoothing Settings"):
        sigma_val = st.sidebar.slider("Smoothness Sigma (σ)", 1, 200, 50, 5, help=TOOLTIPS["Smoothing Sigma"])
        quantile_target = st.sidebar.slider("Target Quantile (%)", 80, 99, 95, 1, help=TOOLTIPS["Target Quantile"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Advanced Features")
    missing_strategy = st.sidebar.selectbox("Handle Missing Values", ["Drop Rows", "Impute Mean", "Impute Median"], help=TOOLTIPS["Impute"])
    use_log = st.sidebar.checkbox("Log-Transform (Skewed Data)", value=False, help=TOOLTIPS["Log Transform"])
    use_clustering = st.sidebar.checkbox("Enable Clustering", value=False)
    cluster_algo = "K-Means"
    if use_clustering:
        cluster_algo = st.sidebar.radio("Algorithm", ["K-Means", "DBSCAN"], help=TOOLTIPS["K-Means"] + "\n\n" + TOOLTIPS["DBSCAN"])
        if cluster_algo == "K-Means":
            k_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
        else:
            eps_val = st.sidebar.slider("Epsilon (Distance)", 0.1, 5.0, 0.5, 0.1)
            min_samples_val = st.sidebar.slider("Min Samples", 2, 20, 5)
    auto_clean = st.sidebar.checkbox("Auto-Clean Text", value=True, help=TOOLTIPS["Auto-Clean Text"])

    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if len(selected_cols) < 2:
            st.error("❌ Select at least 2 numeric variables.")
        else:
            with st.spinner('🧮 Calculating Statistics...'):
                df_work = df[selected_cols].copy()
                if len(df_work) < 10: st.error("❌ Need at least 10 rows."); st.stop()
                if auto_clean:
                    for c in df_work.select_dtypes(include=['object']).columns:
                        df_work[c] = df_work[c].astype(str).str.lower().str.strip()
                if missing_strategy == "Drop Rows":
                    df_work.dropna(inplace=True)
                else:
                    strategy = "mean" if "Mean" in missing_strategy else "median"
                    num_cols = df_work.select_dtypes(include=np.number).columns
                    if len(num_cols) > 0:
                        df_work[num_cols] = SimpleImputer(strategy=strategy).fit_transform(df_work[num_cols])
                    df_work.dropna(inplace=True)
                if len(df_work) < 10: st.error("❌ Not enough data after cleaning."); st.stop()
                const_cols = [c for c in df_work.select_dtypes(include=np.number).columns if df_work[c].std() == 0]
                if const_cols:
                    st.warning(f"⚠️ Removed constant columns: {', '.join(const_cols)}")
                    df_work = df_work.drop(columns=const_cols)
                if len(df_work.select_dtypes(include=np.number).columns) < 2:
                    st.error("❌ Need at least 2 numeric variables with variance."); st.stop()
                if use_log:
                    for c in df_work.select_dtypes(include=np.number).columns:
                        if (df_work[c] > 0).all(): df_work[c] = np.log1p(df_work[c])
                        else: st.warning(f"⚠️ Skipped log for '{c}' (contains <= 0).")
                try:
                    df_encoded = pd.get_dummies(df_work, drop_first=True, dtype=int)
                    if df_encoded.shape[0] <= df_encoded.shape[1]:
                        st.error(f"❌ Need more rows ({df_encoded.shape[0]}) than columns ({df_encoded.shape[1]})."); st.stop()
                    data = df_encoded.values
                    mu = np.mean(data, axis=0)
                    cov = np.cov(data.T)
                    if np.isnan(cov).any() or np.isinf(cov).any(): st.error(" Invalid covariance matrix."); st.stop()
                    inv_cov = pinv(cov)
                    distances = df_encoded.apply(lambda row: distance.mahalanobis(row.values, mu, inv_cov), axis=1)
                    distances = distances.fillna(0).replace([np.inf, -np.inf], 0)
                except Exception as e:
                    st.error(f"❌ Calculation Error: {type(e).__name__}: {str(e)}"); st.stop()
                dims = df_encoded.shape[1]
                global_thresh = np.sqrt(chi2.ppf(global_percent/100.0, dims))
                df_res = df.loc[df_work.index].copy()
                df_res['Mahalanobis_Dist'] = distances
                df_res['Status_Global'] = np.where(df_res['Mahalanobis_Dist'] > global_thresh, 'Outlier', 'Normal')
                if use_clustering:
                    scaled_data = StandardScaler().fit_transform(df_encoded)
                    if cluster_algo == "K-Means":
                        if k_clusters >= len(df_encoded): st.error("❌ K must be < number of rows."); st.stop()
                        labels = KMeans(n_clusters=k_clusters, random_state=42, n_init=10).fit_predict(scaled_data)
                    else:
                        labels = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit_predict(scaled_data)
                        if (labels == -1).all(): st.warning("⚠️ All points labeled as noise.")
                    df_res['Cluster'] = labels.astype(str).replace('-1', 'Noise') if cluster_algo == "DBSCAN" else labels.astype(str)
                total_obs = len(df_res)
                outlier_count = (df_res['Status_Global'] == 'Outlier').sum()
                st.session_state['results'] = df_res
                st.session_state['numeric_cols'] = [c for c in selected_cols if pd.api.types.is_numeric_dtype(df_res[c])]
                st.session_state['cat_cols'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
                st.session_state['global_thresh'] = global_thresh
                st.session_state['sigma_val'] = sigma_val
                st.session_state['quantile_target'] = quantile_target
                st.session_state['outlier_stats'] = {
                    'total': total_obs, 'outliers': int(outlier_count), 'normal': int(total_obs - outlier_count),
                    'pct': (outlier_count / total_obs) * 100, 'threshold': global_thresh, 'dimensions': dims
                }
                st.session_state['df_work'] = df_work
                st.session_state['df_encoded'] = df_encoded

    tab1, tab2 = st.tabs([" Outlier Detection", "📊 Advanced Statistical Analysis"])

    with tab1:
        if 'results' in st.session_state:
            res = st.session_state['results']
            numeric_cols = st.session_state['numeric_cols']
            cat_cols = st.session_state['cat_cols']
            global_thresh = st.session_state['global_thresh']
            sigma_val = st.session_state['sigma_val']
            q_target = st.session_state['quantile_target'] / 100.0
            stats = st.session_state['outlier_stats']
            colors = get_theme_colors()

            st.markdown("---")
            st.subheader(" Analysis Summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("📈 Total Observations", f"{stats['total']:,}")
            m2.metric("🔴 Outliers Detected", f"{stats['outliers']:,}", delta=f"{stats['pct']:.2f}%")
            m3.metric("🟢 Normal Points", f"{stats['normal']:,}", delta=f"{100 - stats['pct']:.2f}%")
            m4.metric("🎯 Threshold (χ²)", f"{stats['threshold']:.3f}")
            m5.metric("📐 Dimensions", stats['dimensions'])

            with st.expander(" Detailed Breakdown"):
                c1, c2 = st.columns(2)
                c1.markdown("**Status Distribution:**"); c1.dataframe(res['Status_Global'].value_counts())
                if 'Cluster' in res.columns:
                    c2.markdown("**Cluster Distribution:**"); c2.dataframe(res['Cluster'].value_counts())
                st.markdown("**Distance Statistics:**"); st.dataframe(res['Mahalanobis_Dist'].describe())

            st.markdown("---")
            st.subheader("🎨 Visualization Console")
            v1, v2, v3 = st.columns(3)
            color_opts = ["Status_Global"]
            if 'Cluster' in res.columns: color_opts.insert(0, "Cluster")
            color_opts += cat_cols
            color_by = v1.selectbox("Color Points By:", color_opts)
            plot_type = v2.radio("Visualization Mode:", [
                "Group Analysis", "Adaptive Boundary", "Custom Scatter",
                "Distance vs Variables", "Pair Plot", "Contribution Analysis", "3D View"
            ], horizontal=True)
            total_pts = len(res)
            limit = v3.slider("Max Points to Plot:", 100, total_pts, min(5000, total_pts), 100) if total_pts > 100 else total_pts

            st.markdown("**📥 Export Settings:**")
            e1, e2 = st.columns([1, 3])
            export_fmt = e1.selectbox("Image Format:", ["PNG", "JPEG", "PDF"])
            export_dpi = e2.slider("Export DPI:", 100, 600, 300, 50)

            df_viz = res.sample(limit, random_state=42) if len(res) > limit else res
            st.caption(f"Displaying **{len(df_viz):,}** points.")

            palette = {'Normal': colors['normal'], 'Outlier': colors['outlier']} if color_by == "Status_Global" else None
            if color_by == "Cluster" and 'Noise' in res['Cluster'].unique():
                palette = {c: sns.color_palette("tab10", len(res['Cluster'].unique()))[i].as_hex() for i, c in enumerate(sorted(res['Cluster'].unique()))}
                palette['Noise'] = '#000000'

            def export_btn(fig, name, is_plotly=False):
                fmt = export_fmt.lower()
                mime = {'png': 'image/png', 'jpeg': 'image/jpeg', 'pdf': 'application/pdf'}[fmt]
                buf = fig_to_bytes_plotly(fig, fmt) if is_plotly else fig_to_bytes_matplotlib(fig, fmt, export_dpi)
                if buf: st.download_button(f"📥 Download {name}.{fmt}", buf, f"{name}.{fmt}", mime)

            if plot_type == "Group Analysis":
                grp_col = st.selectbox("Group By:", ["Cluster"] + cat_cols if 'Cluster' in res.columns else cat_cols)
                if grp_col and numeric_cols:
                    fig = px.scatter(df_viz, x=numeric_cols[0], y='Mahalanobis_Dist', color=grp_col, title=f"Group Analysis: {grp_col}")
                    fig.add_hline(y=global_thresh, line_dash="dash", color="red", annotation_text="Threshold")
                    st.plotly_chart(fig, use_container_width=True); export_btn(fig, "group_analysis", True)

            elif plot_type == "Adaptive Boundary":
                if numeric_cols:
                    sort_col = st.selectbox("Sort By (X-Axis):", numeric_cols)
                    df_s = df_viz.sort_values(sort_col)
                    win = max(3, int(len(df_s) * 0.05))
                    raw = df_s['Mahalanobis_Dist'].rolling(win, center=True, min_periods=1).quantile(q_target).bfill().ffill()
                    smooth = gaussian_filter1d(raw.values, sigma=min(sigma_val, len(df_s)//4))
                    fig, ax = plt.subplots(figsize=(12, 6), facecolor=colors['bg']); ax.set_facecolor(colors['bg'])
                    sns.scatterplot(data=df_s, x=sort_col, y='Mahalanobis_Dist', hue=color_by, palette=palette, alpha=0.6, ax=ax)
                    ax.plot(df_s[sort_col].values, smooth, color='red', linewidth=3, label='Adaptive Boundary')
                    ax.set_title(f"Adaptive Boundary (σ={sigma_val})", color=colors['text']); ax.tick_params(colors=colors['text']); ax.legend(); ax.grid(True, alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); export_btn(fig, "adaptive_boundary")

            elif plot_type == "Custom Scatter":
                if len(numeric_cols) >= 2:
                    x_v, y_v = st.columns(2)
                    x_col = x_v.selectbox("X Axis:", numeric_cols, index=0)
                    y_col = y_v.selectbox("Y Axis:", numeric_cols, index=1)
                    fig, ax = plt.subplots(figsize=(10, 6), facecolor=colors['bg']); ax.set_facecolor(colors['bg'])
                    sns.scatterplot(data=df_viz, x=x_col, y=y_col, hue=color_by, palette=palette, alpha=0.6, ax=ax)
                    ax.set_title(f"{x_col} vs {y_col}", color=colors['text']); ax.tick_params(colors=colors['text'])
                    plt.tight_layout(); st.pyplot(fig); export_btn(fig, f"scatter_{x_col}_{y_col}")

            elif plot_type == "Distance vs Variables":
                if numeric_cols:
                    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4*len(numeric_cols)), facecolor=colors['bg'])
                    if len(numeric_cols) == 1: axes = [axes]
                    for i, col in enumerate(numeric_cols):
                        axes[i].set_facecolor(colors['bg'])
                        sns.scatterplot(data=df_viz, x=col, y='Mahalanobis_Dist', hue=color_by, palette=palette, alpha=0.6, ax=axes[i])
                        axes[i].axhline(global_thresh, color='red', linestyle='--', label='Threshold')
                        axes[i].set_title(f"Distance vs {col}", color=colors['text']); axes[i].tick_params(colors=colors['text'])
                    plt.tight_layout(); st.pyplot(fig); export_btn(fig, "distance_vars")

            elif plot_type == "Pair Plot":
                if numeric_cols:
                    st.info("Generating Pair Plot (may take a moment)...")
                    plot_cols = numeric_cols[:4] + ['Mahalanobis_Dist']
                    g = sns.pairplot(df_viz.sample(min(1000, len(df_viz))), vars=plot_cols, hue=color_by, palette=palette, plot_kws={'alpha': 0.6})
                    st.pyplot(g.fig); export_btn(g.fig, "pairplot")

            elif plot_type == "Contribution Analysis":
                outs = res[res['Status_Global'] == 'Outlier'].nlargest(1, 'Mahalanobis_Dist')
                if not outs.empty:
                    st.write(f"### 🔬 Top Outlier Analysis (Index: {outs.index[0]})")
                    z_scores = {c: (outs.iloc[0][c] - res[c].mean()) / res[c].std() if res[c].std() > 0 else 0 for c in numeric_cols}
                    df_z = pd.DataFrame(list(z_scores.items()), columns=['Var', 'Z']).sort_values('Z', key=abs, ascending=False)
                    fig = px.bar(df_z, x='Z', y='Var', orientation='h', color='Z', color_continuous_scale='RdBu_r', title="Z-Score Drivers")
                    st.plotly_chart(fig, use_container_width=True); export_btn(fig, "contribution", True)

            elif plot_type == "3D View":
                if len(numeric_cols) >= 3:
                    x3, y3, z3 = st.columns(3)
                    cx = x3.selectbox("X", numeric_cols, 0); cy = y3.selectbox("Y", numeric_cols, 1); cz = z3.selectbox("Z", numeric_cols, 2)
                    fig = px.scatter_3d(df_viz, x=cx, y=cy, z=cz, color=color_by, color_discrete_map=palette, opacity=0.7, title="3D Mahalanobis Space")
                    st.plotly_chart(fig, use_container_width=True); export_btn(fig, "3d_view", True)

            st.markdown("---")
            st.subheader("📥 Download Results")
            d1, d2 = st.columns(2)
            d1.download_button("📄 Download CSV", res.to_csv(index=False).encode('utf-8'), "results.csv", "text/csv")
            try:
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    res.to_excel(w, sheet_name='Data', index=False)
                    pd.DataFrame([stats]).to_excel(w, sheet_name='Summary', index=False)
                d2.download_button("📊 Download Excel", buf.getvalue(), "results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except: d2.warning("Excel export requires `openpyxl`.")
        else:
            st.info("👆 Run analysis to view results.")

    with tab2:
        st.subheader("📊 Advanced Statistical Analysis")
        st.markdown("Comprehensive statistical testing framework including normality tests, correlation analysis, model performance metrics, and more.")

        if 'df_work' in st.session_state:
            df_work = st.session_state['df_work']
            numeric_cols_work = df_work.select_dtypes(include=np.number).columns.tolist()

            if not numeric_cols_work:
                st.warning("⚠️ No numeric columns available for statistical analysis.")
            else:
                stat_tab1, stat_tab2, stat_tab3, stat_tab4, stat_tab5 = st.tabs([
                    "📈 Descriptive Statistics", "🔬 Normality Tests", "🔗 Correlation Analysis",
                    "📉 Model Performance", " Outlier Detection"
                ])

                with stat_tab1:
                    st.subheader("Descriptive Statistics")
                    desc_stats = df_work[numeric_cols_work].describe().T
                    desc_stats['skewness'] = df_work[numeric_cols_work].skew()
                    desc_stats['kurtosis'] = df_work[numeric_cols_work].kurtosis()
                    st.dataframe(desc_stats)

                    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                    axes = axes.flatten()
                    for idx, col in enumerate(numeric_cols_work[:4]):
                        sns.histplot(df_work[col].dropna(), kde=True, ax=axes[idx])
                        axes[idx].set_title(f'Distribution of {col}')
                    plt.tight_layout(); st.pyplot(fig)

                with stat_tab2:
                    st.subheader("Normality Tests")
                    st.markdown("Testing whether variables follow a normal distribution using multiple statistical tests.")

                    test_vars = st.multiselect("Select variables to test:", numeric_cols_work, default=numeric_cols_work[:3])

                    if test_vars:
                        normality_results = []
                        for var in test_vars:
                            data = df_work[var].dropna()
                            if len(data) < 3:
                                st.warning(f"️ {var}: Insufficient data (< 3 observations)")
                                continue

                            sw_stat, sw_pvalue = shapiro(data)
                            ks_stat, ks_pvalue = kstest(data, 'norm')
                            ad_result = anderson(data, dist='norm')
                            jb_stat, jb_pvalue = jarque_bera(data)

                            normality_results.append({
                                'variable': var,
                                'shapiro_stat': sw_stat,
                                'shapiro_pvalue': sw_pvalue,
                                'ks_stat': ks_stat,
                                'ks_pvalue': ks_pvalue,
                                'ad_stat': ad_result.statistic,
                                'ad_critical_5%': ad_result.critical_values[2],
                                'is_normal_ad': ad_result.statistic < ad_result.critical_values[2],
                                'jb_stat': jb_stat,
                                'jb_pvalue': jb_pvalue,
                                'skewness': data.skew(),
                                'kurtosis': data.kurtosis()
                            })

                            st.markdown(f"**{var.upper()}:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"- Shapiro-Wilk: statistic={sw_stat:.4f}, p-value={sw_pvalue:.4e}")
                                st.write(f"- Kolmogorov-Smirnov: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4e}")
                            with col2:
                                st.write(f"- Anderson-Darling: statistic={ad_result.statistic:.4f}, normal at 5%: {ad_result.statistic < ad_result.critical_values[2]}")
                                st.write(f"- Jarque-Bera: statistic={jb_stat:.4f}, p-value={jb_pvalue:.4e}")
                            st.write(f"- Skewness: {data.skew():.4f}, Kurtosis: {data.kurtosis():.4f}")
                            st.markdown("---")

                        st.dataframe(pd.DataFrame(normality_results))

                        fig, axes = plt.subplots(len(test_vars), 2, figsize=(14, 5*len(test_vars)))
                        if len(test_vars) == 1: axes = [axes]
                        for idx, var in enumerate(test_vars):
                            data = df_work[var].dropna()
                            qqplot(data, line='s', ax=axes[idx][0])
                            axes[idx][0].set_title(f'Q-Q Plot: {var}')
                            sns.histplot(data, kde=True, ax=axes[idx][1])
                            axes[idx][1].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
                            axes[idx][1].legend(); axes[idx][1].set_title(f'Histogram: {var}')
                        plt.tight_layout(); st.pyplot(fig)

                with stat_tab3:
                    st.subheader("Correlation Analysis")
                    st.markdown("Analyzing relationships between variables using Pearson, Spearman, and Kendall correlations.")

                    if len(numeric_cols_work) >= 2:
                        pearson_corr = df_work[numeric_cols_work].corr(method='pearson')
                        spearman_corr = df_work[numeric_cols_work].corr(method='spearman')
                        kendall_corr = df_work[numeric_cols_work].corr(method='kendall')

                        corr_method = st.radio("Select correlation method:", ["Pearson", "Spearman", "Kendall"], horizontal=True)
                        if corr_method == "Pearson": corr_matrix = pearson_corr
                        elif corr_method == "Spearman": corr_matrix = spearman_corr
                        else: corr_matrix = kendall_corr

                        st.dataframe(corr_matrix.round(3))

                        fig, ax = plt.subplots(figsize=(12, 10))
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.3f', linewidths=.5, ax=ax)
                        plt.title(f'{corr_method} Correlation Heatmap'); plt.tight_layout(); st.pyplot(fig)

                        st.subheader("Statistical Significance of Key Correlations")
                        key_pairs = st.multiselect("Select variable pairs to test:",
                                                  [(a, b) for i, a in enumerate(numeric_cols_work) for b in numeric_cols_work[i+1:]],
                                                  default=[(numeric_cols_work[0], numeric_cols_work[1])] if len(numeric_cols_work) >= 2 else [])

                        if key_pairs:
                            significance_results = []
                            for var1, var2 in key_pairs:
                                data1 = df_work[var1].dropna()
                                data2 = df_work[var2].dropna()
                                mask = df_work[[var1, var2]].notna().all(axis=1)
                                aligned_data1 = df_work.loc[mask, var1]
                                aligned_data2 = df_work.loc[mask, var2]

                                r_pearson, p_pearson = spearmanr(aligned_data1, aligned_data2) if corr_method == "Spearman" else (kendalltau(aligned_data1, aligned_data2) if corr_method == "Kendall" else __import__('scipy.stats', fromlist=['pearsonr']).pearsonr(aligned_data1, aligned_data2))
                                r_spearman, p_spearman = spearmanr(aligned_data1, aligned_data2)
                                r_kendall, p_kendall = kendalltau(aligned_data1, aligned_data2)

                                significance_results.append({
                                    'variable_pair': f'{var1} vs {var2}',
                                    'pearson_r': r_pearson if corr_method == "Pearson" else pearson_corr.loc[var1, var2],
                                    'pearson_pvalue': p_pearson if corr_method == "Pearson" else 0,
                                    'spearman_rho': r_spearman,
                                    'spearman_pvalue': p_spearman,
                                    'kendall_tau': r_kendall,
                                    'kendall_pvalue': p_kendall,
                                    'significant_5%': p_pearson < 0.05 if corr_method == "Pearson" else p_spearman < 0.05
                                })

                            st.dataframe(pd.DataFrame(significance_results))

                with stat_tab4:
                    st.subheader("Model Performance Metrics")
                    st.markdown("Evaluating predictive model performance (requires 'selling_price' and 'predicted_price' columns).")

                    if 'selling_price' in df_work.columns and 'predicted_price' in df_work.columns:
                        y_true = df_work['selling_price']
                        y_pred = df_work['predicted_price']
                        mask = y_true.notna() & y_pred.notna()
                        y_true_clean = y_true[mask]
                        y_pred_clean = y_pred[mask]

                        if len(y_true_clean) > 0:
                            r2 = r2_score(y_true_clean, y_pred_clean)
                            adj_r2 = 1 - (1-r2) * (len(y_true_clean)-1)/(len(y_true_clean)-2)
                            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                            mae = mean_absolute_error(y_true_clean, y_pred_clean)
                            mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
                            evs = explained_variance_score(y_true_clean, y_pred_clean)

                            residuals = y_true_clean - y_pred_clean
                            residual_mean = residuals.mean()
                            residual_std = residuals.std()

                            col1, col2, col3 = st.columns(3)
                            col1.metric("R²", f"{r2:.4f}")
                            col2.metric("Adjusted R²", f"{adj_r2:.4f}")
                            col3.metric("RMSE", f"${rmse:,.2f}")

                            col4, col5, col6 = st.columns(3)
                            col4.metric("MAE", f"${mae:,.2f}")
                            col5.metric("MAPE", f"{mape:.2f}%")
                            col6.metric("Explained Variance", f"{evs:.4f}")

                            st.markdown(f"**Residual Statistics:** Mean=${residual_mean:,.2f}, Std=${residual_std:,.2f}")

                            dw_stat = durbin_watson(residuals)
                            st.markdown(f"**Durbin-Watson:** {dw_stat:.4f} ({'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Autocorrelation detected'})")

                            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                            axes[0, 0].scatter(y_pred_clean, y_true_clean, alpha=0.5, edgecolors='k', linewidth=0.5)
                            min_val = min(y_pred_clean.min(), y_true_clean.min())
                            max_val = max(y_pred_clean.max(), y_true_clean.max())
                            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                            axes[0, 0].set_xlabel('Predicted Price'); axes[0, 0].set_ylabel('Actual Price')
                            axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.4f})')

                            axes[0, 1].scatter(y_pred_clean, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
                            axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
                            axes[0, 1].set_xlabel('Predicted Price'); axes[0, 1].set_ylabel('Residuals')
                            axes[0, 1].set_title('Residuals vs Predicted')

                            sns.histplot(residuals, kde=True, ax=axes[1, 0])
                            axes[1, 0].axvline(x=0, color='r', linestyle='--')
                            axes[1, 0].set_xlabel('Residuals'); axes[1, 0].set_title('Distribution of Residuals')

                            qqplot(residuals, line='s', ax=axes[1, 1])
                            axes[1, 1].set_title('Q-Q Plot of Residuals')
                            plt.tight_layout(); st.pyplot(fig)
                        else:
                            st.warning("⚠️ No valid data for model performance calculation.")
                    else:
                        st.warning("⚠️ 'selling_price' and 'predicted_price' columns required for model performance analysis.")

                with stat_tab5:
                    st.subheader("Outlier Detection")
                    st.markdown("Detecting outliers using IQR, Z-score, and MAD methods.")

                    outlier_cols = st.multiselect("Select columns for outlier detection:", numeric_cols_work, default=numeric_cols_work[:3])

                    if outlier_cols:
                        outlier_results = {}
                        for col in outlier_cols:
                            data = df_work[col].dropna()
                            Q1 = data.quantile(0.25)
                            Q3 = data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()

                            z_scores = np.abs(zscore(data))
                            z_outliers = (z_scores > 3).sum()

                            median = data.median()
                            mad = np.median(np.abs(data - median))
                            modified_z_scores = 0.6745 * (data - median) / mad
                            mad_outliers = (np.abs(modified_z_scores) > 3.5).sum()

                            outlier_results[col] = {
                                'total_observations': len(data),
                                'iqr_outliers': iqr_outliers,
                                'iqr_outlier_pct': iqr_outliers / len(data) * 100,
                                'zscore_outliers': z_outliers,
                                'zscore_outlier_pct': z_outliers / len(data) * 100,
                                'mad_outliers': mad_outliers,
                                'mad_outlier_pct': mad_outliers / len(data) * 100,
                                'lower_bound_iqr': lower_bound,
                                'upper_bound_iqr': upper_bound
                            }

                        st.dataframe(pd.DataFrame(outlier_results).T)

                        fig, axes = plt.subplots(1, len(outlier_cols), figsize=(6*len(outlier_cols), 6))
                        if len(outlier_cols) == 1: axes = [axes]
                        for idx, col in enumerate(outlier_cols):
                            sns.boxplot(y=df_work[col], ax=axes[idx])
                            axes[idx].set_title(f'Box Plot: {col}')
                        plt.tight_layout(); st.pyplot(fig)
        else:
            st.info("👆 Run analysis to view statistical results.")

else:
    st.info(" Upload a file to begin.")
    st.markdown("""
    ### 📖 About Ultimate Outlier Detector
    An advanced multivariate tool based on **Mahalanobis Distance** with comprehensive statistical analysis capabilities.
    **Features:** Dark Mode, CSV/Excel Support, 7 Visualization Modes, Clustering (K-Means/DBSCAN), Adaptive Boundaries, Multi-Format Export, and Advanced Statistical Testing.
    """)

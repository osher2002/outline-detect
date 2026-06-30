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
from io import BytesI
import os

# Page configuration
st.set_page_config(page_title="Outlier Detector Pro", layout="wide", page_icon="🔍")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_fine_grid(ax):
    """Add fine grid to matplotlib axes"""
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)

def info_icon(text, label="ℹ️"):
    """
    Creates an info icon with tooltip using Streamlit's help parameter.
    Returns empty string to embed inline via st.markdown with write.
    """
    return label

def tooltip(label, text):
    """Inline tooltip using Streamlit's native help"""
    st.caption(f"**{label}**: {text}")

def fig_to_bytes_matplotlib(fig, format='png', dpi=300):
    """Convert matplotlib figure to bytes for download"""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf

def fig_to_bytes_plotly(fig, format='png', scale=2):
    """Convert plotly figure to bytes for download"""
    try:
        img_bytes = fig.to_image(format=format, scale=scale)
        return BytesIO(img_bytes)
    except Exception as e:
        st.warning(f"Plotly export requires kaleido package: {e}")
        return None

# ============================================================
# TOOLTIP DICTIONARY (Section 3)
# ============================================================

TOOLTIPS = {
    "Log Transform": "טרנספורמציה לוגריתמית (log1p) שמנרמלת התפלגויות עם הטיה חיובית חזקה (right-skewed). מקטינה את השפעת ערכי קיצון והופכת את הנתונים לקרובים יותר להתפלגות נורמלית - קריטי לחישוב Mahalanobis Distance שמניח נורמליות.",
    
    "K-Means": "אלגוריתם אשכולות (clustering) שמחלק את הנתונים ל-K קבוצות על ידי מזעור השונות התוך-קבוצתית. דורש הגדרה מראש של מספר האשכולות. יעיל לזיהוי תת-אוכלוסיות עם פרופילי חריגות שונים.",
    
    "DBSCAN": "Density-Based Spatial Clustering. אלגוריתם מבוסס צפיפות שמזהה אוטומטית את מספר האשכולות ומסמן נקודות 'רועשות' (noise) שאינן שייכות לאף אשכול. מצוין לזיהוי חריגים מבניים.",
    
    "Mahalanobis Distance": "מדד מרחק רב-ממדי שלוקח בחשבון את הקורלציות בין המשתנים (דרך מטריצת השונות-המשותפת ההופכית). בניגוד למרחק אוקלידי, הוא מזהה חריגים שנובעים משילובים נדירים של ערכים, גם אם כל ערך בנפרד נראה סביר.",
    
    "Global Sensitivity": "אחוזון (percentile) של התפלגות חי-בריבוע (Chi-Square) שקובע את סף החריגות הגלובלי. 95% = 5% מהתצפיות יוגדרו כחריגים בהנחת נורמליות רב-ממדית.",
    
    "Smoothing Sigma": "פרמטר σ של סינון גאוסיאני (Gaussian filter). ערך גבוה = קו גבול חלק יותר שמושפע מטווח רחב של נקודות. ערך נמוך = קו מגיב יותר לשינויים מקומיים.",
    
    "Adaptive Boundary": "גבול חריגות דינמי שמשתנה לפי ערכי ציר ה-X (למשל, מחיר). שימושי כאשר שונות הנתונים משתנה (heteroscedasticity) - למשל, רכבים יקרים טבעיים שיהיו להם שאריות גדולות יותר.",
    
    "Target Quantile": "האחוזון של מרחקי Mahalanobis שישמש כבסיס לקו הגבול האדפטיבי. 95% = הקו יעבור מעל 95% מהנקודות בכל 'חלון' לאורך ציר ה-X.",
    
    "Impute": "אסטרטגיה למילוי ערכים חסרים: ממוצע (mean) רגיש לחריגים, חציון (median) עמיד יותר. Drop פשוט מסיר שורות עם ערכים חסרים.",
    
    "Auto-Clean Text": "מנקה אוטומטית טקסט (lowercase, strip) כדי למנוע כפילויות כמו 'Toyota' לעומת 'toyota ' שיוצרות קטגוריות נפרדות ב-one-hot encoding."
}

# ============================================================
# PAGE TITLE
# ============================================================

st.title("🔍 Ultimate Outlier Detector")
st.markdown("### Mahalanobis Analysis with Filtering, Clustering, and Multi-Format Export")
st.markdown("---")

# ============================================================
# SIDEBAR - DATA INPUT (Section 5: XLSX Support)
# ============================================================

with st.sidebar:
    st.header("📂 1. Data Input")
    
    # Support CSV, XLSX, XLS
    uploaded_file = st.file_uploader(
        "Upload Data File", 
        type=["csv", "xlsx", "xls"],
        help="תומך ב-CSV, Excel (XLSX/XLS)"
    )
    
    with st.expander("⚙️ Advanced Load Settings"):
        header_row_idx = st.number_input("Header Row Index", 0, 10, 0)
        encoding_option = st.selectbox("Encoding (CSV)", ["Auto / UTF-8", "ISO-8859-1", "cp1252"])
        
        # Excel-specific settings
        excel_sheet = st.text_input("Sheet Name (Excel)", value="", help="השאר ריק לטעינת הגיליון הראשון")
        skip_rows_excel = st.number_input("Skip Rows (Excel)", 0, 20, 0)

    st.markdown("---")
    st.header("⚙️ 2. Analysis Configuration")

# ============================================================
# MAIN LOGIC
# ============================================================

if uploaded_file is not None:
    # --- FILE LOADING (Section 5) ---
    try:
        filename = uploaded_file.name.lower()
        
        if filename.endswith('.csv'):
            enc = None if encoding_option == "Auto / UTF-8" else encoding_option.split()[0]
            df_raw = pd.read_csv(uploaded_file, encoding=enc, header=header_row_idx)
            file_type = "CSV"
        elif filename.endswith(('.xlsx', '.xls')):
            sheet = excel_sheet if excel_sheet.strip() else 0
            df_raw = pd.read_excel(
                uploaded_file, 
                sheet_name=sheet, 
                header=header_row_idx,
                skiprows=range(1, skip_rows_excel + 1) if skip_rows_excel > 0 else None
            )
            file_type = "Excel"
        else:
            st.error("Unsupported file type")
            st.stop()
            
        st.success(f"✅ Loaded {file_type}: {uploaded_file.name} ({len(df_raw):,} rows, {len(df_raw.columns)} columns)")
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # ============================================================
    # DATA FILTERING & PREVIEW
    # ============================================================
    
    st.markdown("---")
    with st.expander("🔎 Data Filtering and Preview", expanded=True):
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
                selected_vals = st.multiselect(f"Values for: {col}", unique_vals, default=unique_vals[:min(10, len(unique_vals))])
                if selected_vals:
                    df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]
        
        st.info(f"📊 Rows after filtering: **{len(df_filtered):,}** (Original: {len(df_raw):,})")
        st.dataframe(df_filtered.head(50), height=150)

    total_rows = len(df_filtered)
    if total_rows == 0:
        st.error("No data left after filtering.")
        st.stop()

    # --- Row Range Control ---
    st.sidebar.subheader("📏 Row Range (Subset)")
    if total_rows > 1:
        start_row, end_row = st.sidebar.slider("Process rows:", 0, total_rows, (0, total_rows), 1)
    else:
        start_row, end_row = 0, total_rows
    
    df = df_filtered.iloc[start_row:end_row].copy()
    st.sidebar.info(f"Active Rows: **{len(df):,}**")

    # ============================================================
    # VARIABLE SELECTION
    # ============================================================
    
    st.sidebar.subheader("🎯 Variables")
    all_cols = df.columns.tolist()
    potential = ['year', 'make', 'condition', 'odometer', 'mmr', 'sellingprice', 'mileage', 
                 'price', 'selling_price', 'predicted_price', 'quality_hybrid', 'quality_heuristic',
                 'quality_residual', 'car_age', 'manufacturer_reliability', 'residuals']
    default_cols = [c for c in all_cols if str(c).lower().strip() in [p.lower() for p in potential]]
    selected_cols = st.sidebar.multiselect("Select Model Variables (Min 2):", all_cols, default=default_cols)

    # ============================================================
    # THRESHOLDS & SETTINGS (with Tooltips - Section 3)
    # ============================================================
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎚️ Settings")
    
    # Global Sensitivity with tooltip
    global_percent = st.sidebar.slider(
        f"Global Sensitivity (%) {info_icon('🛈')}", 
        80.0, 99.9, 95.0, 0.1,
        help=TOOLTIPS["Global Sensitivity"]
    )
    
    with st.sidebar.expander("📈 Smoothing Settings"):
        sigma_val = st.slider(
            f"Smoothness Sigma (σ) {info_icon('🛈')}", 
            1, 200, 50, 5,
            help=TOOLTIPS["Smoothing Sigma"]
        )
        quantile_target = st.slider(
            f"Target Quantile (%) {info_icon('🛈')}", 
            80, 99, 95, 1,
            help=TOOLTIPS["Target Quantile"]
        )

    # ============================================================
    # ADVANCED FEATURES (with Tooltips - Section 3)
    # ============================================================
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Advanced Features")
    
    # Missing Data
    missing_strategy = st.sidebar.selectbox(
        f"Handle Missing Values {info_icon('🛈')}", 
        ["Drop Rows", "Impute Mean", "Impute Median"],
        help=TOOLTIPS["Impute"]
    )
    
    # Log Transform with tooltip
    use_log = st.sidebar.checkbox(
        f"Log-Transform (Skewed Data) {info_icon('🛈')}", 
        value=False,
        help=TOOLTIPS["Log Transform"]
    )
    
    # Clustering with tooltips
    use_clustering = st.sidebar.checkbox("Enable Clustering", value=False)
    cluster_algo = "K-Means"
    if use_clustering:
        cluster_algo = st.sidebar.radio(
            f"Algorithm {info_icon('🛈')}", 
            ["K-Means", "DBSCAN"],
            help="K-Means: " + TOOLTIPS["K-Means"] + "\n\nDBSCAN: " + TOOLTIPS["DBSCAN"]
        )
        if cluster_algo == "K-Means":
            k_clusters = st.sidebar.slider(
                f"Number of Clusters (K) {info_icon('🛈')}", 
                2, 10, 3,
                help=TOOLTIPS["K-Means"]
            )
        else:
            eps_val = st.sidebar.slider("Epsilon (Distance)", 0.1, 5.0, 0.5, 0.1)
            min_samples_val = st.sidebar.slider("Min Samples", 2, 20, 5)

    auto_clean = st.sidebar.checkbox(
        f"Auto-Clean Text {info_icon('🛈')}", 
        value=True,
        help=TOOLTIPS["Auto-Clean Text"]
    )
    
    # ============================================================
    # EXECUTION
    # ============================================================
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if len(selected_cols) < 2:
            st.error("Select at least 2 numeric variables.")
        else:
            with st.spinner('🧮 Calculating Statistics...'):
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
                    distances = df_encoded.apply(
                        lambda row: distance.mahalanobis(row.values, mu, inv_cov), 
                        axis=1
                    )
                except Exception as e:
                    st.error(f"Calculation Error: {e}")
                    st.stop()
                
                dims = df_encoded.shape[1]
                global_thresh = np.sqrt(chi2.ppf(global_percent/100.0, dims))
                
                # Results DataFrame
                df_res = df.loc[df_work.index].copy()
                df_res['Mahalanobis_Dist'] = distances
                df_res['Status_Global'] = np.where(
                    df_res['Mahalanobis_Dist'] > global_thresh, 
                    'Outlier', 
                    'Normal'
                )
                
                # Clustering Logic
                if use_clustering:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df_encoded)
                    
                    if cluster_algo == "K-Means":
                        model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                        labels = model.fit_predict(scaled_data)
                    else:  # DBSCAN
                        model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
                        labels = model.fit_predict(scaled_data)
                    
                    df_res['Cluster'] = labels.astype(str)
                    if cluster_algo == "DBSCAN":
                        df_res['Cluster'] = df_res['Cluster'].replace('-1', 'Noise')

                # ============================================================
                # SECTION 1: OUTLIER COUNT DISPLAY
                # ============================================================
                
                # Calculate statistics
                total_obs = len(df_res)
                outlier_count = (df_res['Status_Global'] == 'Outlier').sum()
                normal_count = total_obs - outlier_count
                outlier_pct = (outlier_count / total_obs) * 100
                
                # Save all to session state
                st.session_state['results'] = df_res
                st.session_state['numeric_cols'] = [c for c in selected_cols if pd.api.types.is_numeric_dtype(df_res[c])]
                st.session_state['cat_cols'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
                st.session_state['global_thresh'] = global_thresh
                st.session_state['sigma_val'] = sigma_val 
                st.session_state['quantile_target'] = quantile_target
                st.session_state['has_clusters'] = use_clustering
                
                # Outlier statistics
                st.session_state['outlier_stats'] = {
                    'total': total_obs,
                    'outliers': int(outlier_count),
                    'normal': int(normal_count),
                    'pct': outlier_pct,
                    'threshold': global_thresh,
                    'dimensions': dims
                }

    # ============================================================
    # VISUALIZATION SECTION
    # ============================================================
    
    if 'results' in st.session_state:
        res = st.session_state['results']
        numeric_cols = st.session_state['numeric_cols']
        cat_cols = st.session_state['cat_cols']
        global_thresh = st.session_state['global_thresh']
        sigma_val = st.session_state['sigma_val']
        q_target = st.session_state['quantile_target'] / 100.0
        outlier_stats = st.session_state['outlier_stats']
        
        # ============================================================
        # SECTION 1: METRICS DASHBOARD - OUTLIER COUNT
        # ============================================================
        
        st.markdown("---")
        st.subheader("📊 Analysis Summary")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        
        with m1:
            st.metric(
                label="📈 Total Observations",
                value=f"{outlier_stats['total']:,}"
            )
        
        with m2:
            st.metric(
                label="🔴 Outliers Detected",
                value=f"{outlier_stats['outliers']:,}",
                delta=f"{outlier_stats['pct']:.2f}%"
            )
        
        with m3:
            st.metric(
                label="🟢 Normal Points",
                value=f"{outlier_stats['normal']:,}",
                delta=f"{100 - outlier_stats['pct']:.2f}%"
            )
        
        with m4:
            st.metric(
                label="🎯 Threshold (χ²)",
                value=f"{outlier_stats['threshold']:.3f}"
            )
        
        with m5:
            st.metric(
                label="📐 Dimensions",
                value=outlier_stats['dimensions']
            )
        
        # Detailed outlier breakdown
        with st.expander("📋 Detailed Outlier Breakdown"):
            col_b1, col_b2 = st.columns(2)
            
            with col_b1:
                st.markdown("**Outlier Status Distribution:**")
                status_counts = res['Status_Global'].value_counts()
                st.dataframe(status_counts)
            
            with col_b2:
                if 'Cluster' in res.columns:
                    st.markdown("**Cluster Distribution:**")
                    cluster_counts = res['Cluster'].value_counts()
                    st.dataframe(cluster_counts)
            
            st.markdown("**Mahalanobis Distance Statistics:**")
            dist_stats = res['Mahalanobis_Dist'].describe()
            st.dataframe(dist_stats)
        
        st.markdown("---")
        st.subheader("🎨 Visualization Console")

        c1, c2, c3 = st.columns(3)
        
        # Color Selection
        color_opts = ["Status_Global"]
        if 'Cluster' in res.columns: 
            color_opts.insert(0, "Cluster")
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
            limit_plot = c3.slider(
                "Max Points to Plot:", 
                100, total_points, 
                min(5000, total_points), 100
            )
        else:
            limit_plot = total_points
        
        # ============================================================
        # SECTION 4: EXPORT FORMAT SELECTION
        # ============================================================
        
        st.markdown("**📥 Export Settings:**")
        exp_col1, exp_col2 = st.columns([1, 3])
        
        with exp_col1:
            export_format = st.selectbox(
                "Image Format:",
                ["PNG", "JPEG", "PDF"],
                help="PNG: איכות גבוהה עם שקיפות | JPEG: קובץ קטן | PDF: וקטורי לאיכות מקסימלית"
            )
        
        with exp_col2:
            export_dpi = st.slider("Export DPI:", 100, 600, 300, 50)
        
        # Data Subset
        if len(res) > limit_plot:
            df_viz_base = res.sample(limit_plot, random_state=42)
            st.caption(f"Showing sample of **{limit_plot:,}** points out of {total_points:,}")
        else:
            df_viz_base = res
            st.caption(f"Showing all **{total_points:,}** points.")

        # Define Palette
        custom_palette = None
        if color_by == "Status_Global":
            custom_palette = {'Normal': '#1f77b4', 'Outlier': '#d62728'}
        elif color_by == "Cluster" and 'Noise' in res['Cluster'].unique():
            unique_clusters = sorted(res['Cluster'].unique())
            pal = sns.color_palette("tab10", len(unique_clusters)).as_hex()
            custom_palette = {grp: col for grp, col in zip(unique_clusters, pal)}
            if 'Noise' in custom_palette:
                custom_palette['Noise'] = '#000000'

        # Helper function for export button
        def render_export_button(fig, fig_name, fig_type='matplotlib'):
            """Render download button for a figure"""
            format_lower = export_format.lower()
            mime_types = {
                'png': 'image/png',
                'jpeg': 'image/jpeg',
                'pdf': 'application/pdf'
            }
            
            if fig_type == 'matplotlib':
                buf = fig_to_bytes_matplotlib(fig, format=format_lower, dpi=export_dpi)
                if buf:
                    st.download_button(
                        label=f"📥 Download {fig_name}.{format_lower}",
                        data=buf,
                        file_name=f"{fig_name}.{format_lower}",
                        mime=mime_types[format_lower]
                    )
            elif fig_type == 'plotly':
                buf = fig_to_bytes_plotly(fig, format=format_lower, scale=2)
                if buf:
                    st.download_button(
                        label=f"📥 Download {fig_name}.{format_lower}",
                        data=buf,
                        file_name=f"{fig_name}.{format_lower}",
                        mime=mime_types[format_lower]
                    )

        # ============================================================
        # VISUALIZATION MODES
        # ============================================================
        
        # 1. GROUP ANALYSIS (K-GROUPS)
        if plot_type == "Group Analysis (K-Groups)":
            default_grp = "Cluster" if 'Cluster' in res.columns else (cat_cols[0] if cat_cols else None)
            
            if not default_grp and not cat_cols:
                st.warning("No grouping columns found.")
            else:
                gc1, gc2, gc3 = st.columns(3)
                
                grp_list = []
                if 'Cluster' in res.columns: 
                    grp_list.append('Cluster')
                grp_list += cat_cols
                
                group_col = gc1.selectbox("Group By:", grp_list, index=0)
                
                uniqs = res[group_col].unique()
                sel_grps = gc2.multiselect(
                    "Select Groups:", 
                    uniqs, 
                    default=uniqs[:3] if len(uniqs) > 0 else None
                )
                x_axis = gc3.selectbox("X Axis Variable:", numeric_cols, index=0)

                if sel_grps:
                    df_g = res[res[group_col].isin(sel_grps)]
                    if len(df_g) > limit_plot: 
                        df_g = df_g.sample(limit_plot, random_state=42)
                        
                    fig = px.scatter(
                        df_g, x=x_axis, y='Mahalanobis_Dist', 
                        color=group_col, symbol=group_col,
                        title=f"Group Comparison: {group_col}",
                        height=600
                    )
                    
                    fig.add_hline(
                        y=global_thresh, 
                        line_dash="dash", 
                        line_color="black", 
                        annotation_text=f"Threshold ({global_thresh:.2f})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export button (Section 4)
                    render_export_button(fig, "group_analysis", 'plotly')
                else:
                    st.info("Please select at least one group.")

        # 2. ADAPTIVE SMOOTHING
        elif plot_type == "Adaptive Boundary (Smooth Line)":
            if len(numeric_cols) > 0:
                # Tooltip for adaptive boundary
                with st.expander(f"ℹ️ What is Adaptive Boundary?"):
                    st.markdown(TOOLTIPS["Adaptive Boundary"])
                
                xc, yc = st.columns([1, 3])
                sort_col = xc.selectbox("Sort By (X-Axis):", numeric_cols, index=0)
                
                with yc:
                    df_sorted = res.sort_values(by=sort_col).copy()
                    
                    min_win = max(2, int(len(df_sorted) * 0.05))
                    rolling = max(5, min_win)
                    
                    raw_line = df_sorted['Mahalanobis_Dist'].rolling(
                        window=rolling, center=True
                    ).quantile(q_target)
                    raw_line = raw_line.bfill().ffill()
                    
                    try:
                        smoothed = gaussian_filter1d(raw_line.values, sigma=sigma_val)
                    except:
                        smoothed = raw_line.values
                    
                    df_sorted['Smooth_Threshold'] = smoothed
                    df_sorted['Status_Adaptive'] = np.where(
                        df_sorted['Mahalanobis_Dist'] > df_sorted['Smooth_Threshold'], 
                        'Outlier', 
                        'Normal'
                    )

                    if len(df_sorted) > limit_plot:
                        df_viz_ad = df_sorted.sample(limit_plot, random_state=42).sort_values(by=sort_col)
                    else:
                        df_viz_ad = df_sorted

                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    hue = 'Status_Adaptive' if color_by == "Status_Global" else color_by
                    pal = {'Normal': '#1f77b4', 'Outlier': '#d62728'} if hue == 'Status_Adaptive' else custom_palette

                    sns.scatterplot(
                        data=df_viz_ad, x=sort_col, y='Mahalanobis_Dist', 
                        hue=hue, palette=pal, alpha=0.6, s=30, ax=ax
                    )
                    
                    ax.plot(
                        df_sorted[sort_col].values, 
                        df_sorted['Smooth_Threshold'].values, 
                        color='black', linewidth=3, label='Adaptive Boundary'
                    )
                    
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                    ax.set_title(f"Adaptive Boundary Analysis (σ={sigma_val}, Quantile={q_target*100:.0f}%)")
                    add_fine_grid(ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Export button (Section 4)
                    render_export_button(fig, "adaptive_boundary", 'matplotlib')
            else:
                st.warning("No numeric variables.")

        # 3. CUSTOM SCATTER
        elif plot_type == "Custom Pair Scatter (X vs Y)":
            if len(numeric_cols) >= 2:
                c1, c2 = st.columns(2)
                x_var = c1.selectbox("X Axis:", numeric_cols, index=0)
                y_var = c2.selectbox("Y Axis:", numeric_cols, index=1)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    data=df_viz_base, x=x_var, y=y_var, 
                    hue=color_by, palette=custom_palette, alpha=0.6, s=30, ax=ax
                )
                
                ax.set_title(f"Scatter: {x_var} vs {y_var}")
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                add_fine_grid(ax)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Export button (Section 4)
                render_export_button(fig, f"scatter_{x_var}_vs_{y_var}", 'matplotlib')
            else:
                st.warning("Need at least 2 numeric variables.")

        # 4. N GRAPHS
        elif plot_type == "Distance vs. Variables (N-Graphs)":
            if numeric_cols:
                fig, axes = plt.subplots(
                    len(numeric_cols), 1, 
                    figsize=(10, 4 * len(numeric_cols)), 
                    constrained_layout=True
                )
                if len(numeric_cols) == 1: 
                    axes = [axes]
                
                for i, col in enumerate(numeric_cols):
                    ax = axes[i]
                    sns.scatterplot(
                        data=df_viz_base, x=col, y='Mahalanobis_Dist', 
                        hue=color_by, palette=custom_palette, alpha=0.6, ax=ax
                    )
                    ax.axhline(
                        global_thresh, color='black', linestyle='--', 
                        label=f'Global Threshold ({global_thresh:.2f})'
                    )
                    ax.set_ylabel("Mahalanobis Distance")
                    ax.set_title(f"Distance vs {col}")
                    
                    if i == 0: 
                        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                    else: 
                        if ax.get_legend(): 
                            ax.get_legend().remove()
                    add_fine_grid(ax)
                
                st.pyplot(fig)
                
                # Export button (Section 4)
                render_export_button(fig, "distance_vs_variables", 'matplotlib')

        # 5. PAIR PLOT
        elif plot_type == "Pair Plot (NxN Matrix)":
            if numeric_cols:
                st.info("Generating Pair Plot (this may take a moment)...")
                plot_cols = numeric_cols + ['Mahalanobis_Dist']
                
                if len(df_viz_base) > 1000:
                    df_pair = df_viz_base.sample(1000, random_state=42)
                else:
                    df_pair = df_viz_base

                g = sns.pairplot(
                    df_pair, vars=plot_cols, hue=color_by, 
                    palette=custom_palette, plot_kws={'alpha': 0.6, 's': 20}
                )
                for ax in g.axes.flatten():
                    if ax is not None: 
                        add_fine_grid(ax)
                
                plt.tight_layout()
                st.pyplot(g.fig)
                
                # Export button (Section 4)
                render_export_button(g.fig, "pairplot_matrix", 'matplotlib')

        # 6. CONTRIBUTION ANALYSIS
        elif plot_type == "Outlier Contribution Analysis":
            outs = res[res['Status_Global'] == 'Outlier'].sort_values(
                'Mahalanobis_Dist', ascending=False
            )
            if not outs.empty:
                st.write("### 🏆 Top Extreme Outliers:")
                st.dataframe(outs.head(10))
                
                top_row = outs.iloc[0]
                st.markdown(f"#### 🔬 Driver Analysis for Index: {top_row.name}")
                
                z_scores = {}
                for c in numeric_cols:
                    m = res[c].mean()
                    s = res[c].std()
                    z_scores[c] = (top_row[c] - m) / s if s > 0 else 0
                
                df_z = pd.DataFrame(list(z_scores.items()), columns=['Var', 'Z'])
                df_z['Abs'] = df_z['Z'].abs()
                df_z = df_z.sort_values('Abs', ascending=False)
                
                fig = px.bar(
                    df_z, x='Z', y='Var', orientation='h', 
                    color='Z', color_continuous_scale='RdBu_r',
                    title="Z-Score Contribution to Outlier Status"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Export button (Section 4)
                render_export_button(fig, "outlier_contribution", 'plotly')
            else:
                st.success("✅ No outliers found in the current analysis.")

        # 7. 3D SCATTER
        elif plot_type == "3D Scatter View":
            if len(numeric_cols) >= 3:
                c3a, c3b, c3c = st.columns(3)
                x3 = c3a.selectbox("X", numeric_cols, index=0)
                y3 = c3b.selectbox("Y", numeric_cols, index=1)
                z3 = c3c.selectbox("Z", numeric_cols, index=2)
                
                fig = px.scatter_3d(
                    df_viz_base, x=x3, y=y3, z=z3, color=color_by, 
                    color_discrete_map=custom_palette, opacity=0.7, 
                    title=f"3D View: {x3} × {y3} × {z3}",
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Export button (Section 4)
                render_export_button(fig, "3d_scatter", 'plotly')
            else:
                st.warning("Need 3+ numeric variables for 3D view.")

        # ============================================================
        # DATA DOWNLOAD SECTION
        # ============================================================
        
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            csv = res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Results (CSV)", 
                data=csv, 
                file_name="outlier_analysis_results.csv", 
                mime="text/csv",
                use_container_width=True
            )
        
        with dl_col2:
            # Excel export
            try:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    res.to_excel(writer, sheet_name='Results', index=False)
                    
                    # Add summary sheet
                    summary_df = pd.DataFrame([
                        {'Metric': 'Total Observations', 'Value': outlier_stats['total']},
                        {'Metric': 'Outliers Detected', 'Value': outlier_stats['outliers']},
                        {'Metric': 'Outlier Percentage', 'Value': f"{outlier_stats['pct']:.2f}%"},
                        {'Metric': 'Threshold', 'Value': outlier_stats['threshold']},
                        {'Metric': 'Dimensions', 'Value': outlier_stats['dimensions']}
                    ])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="📊 Download Results (Excel)", 
                    data=excel_buffer.getvalue(), 
                    file_name="outlier_analysis_results.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"Excel export unavailable: {e}")
        
        # TODO Section 2: Error fixes placeholder
        # TODO Section 6: Additional statistical tools placeholder

else:
    st.info("👆 Upload a CSV or Excel file to start the analysis.")
    
    # Welcome / Info Section
    st.markdown("---")
    st.markdown("""
    ### 📖 About This Tool
    
    **Ultimate Outlier Detector** הוא כלי מתקדם לזיהוי חריגים רב-ממדי המבוסס על 
    **Mahalanobis Distance**, המותאם במיוחד לניתוח נתוני שוק (כמו רכבים יד שנייה).
    
    #### ✨ Key Features:
    - 🎯 **זיהוי חריגים רב-ממדי** - לוקח בחשבון קורלציות בין משתנים
    - 📊 **תמיכה ב-CSV ו-Excel** - טעינה גמישה של נתונים
    - 🎨 **ויזואליזציות מתקדמות** - 7 מצבי תצוגה שונים
    - 🤖 **Clustering** - K-Means ו-DBSCAN לזיהוי תת-אוכלוסיות
    - 📈 **Adaptive Boundary** - גבול דינמי להטרוסקדסטיות
    - 📥 **ייצוא רב-פורמטי** - PNG, JPEG, PDF, CSV, Excel
    
    #### 📚 Key Concepts (See tooltips in sidebar):
    - Mahalanobis Distance
    - Log Transform
    - K-Means / DBSCAN Clustering
    - Adaptive Boundary
    - Global Sensitivity Threshold
    """)

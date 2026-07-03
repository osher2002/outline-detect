import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy import stats
from scipy.stats import (chi2, shapiro, kstest, anderson, jarque_bera,
                         levene, bartlett, f_oneway, ttest_ind, ttest_rel,
                         mannwhitneyu, kruskal, chi2_contingency,
                         spearmanr, kendalltau, pearsonr, boxcox,
                         zscore, probplot, norm)
from scipy.linalg import pinv
from scipy.ndimage import gaussian_filter1d
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error, explained_variance_score)
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import OLS
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
import traceback
import sys

# ============================================================
# 1. PAGE CONFIGURATION & THEME MANAGEMENT
# ============================================================

if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

def apply_theme_css():
    """Apply CSS styling based on current theme"""
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
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background-color: #1a1d21; border-radius: 4px 4px 0 0; }
        .stTabs [aria-selected="true"] { background-color: #2d333b !important; }
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
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 4px 4px 0 0; }
        .stTabs [aria-selected="true"] { background-color: #ffffff !important; }
        </style>
        """, unsafe_allow_html=True)

apply_theme_css()
st.set_page_config(page_title="Outlier Detector Pro", layout="wide", page_icon="🔍")

# ============================================================
# 2. THEME TOGGLE & SIDEBAR SETUP
# ============================================================

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
    uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx", "xls"], 
                                     help="Supports CSV and Excel formats")
    
    with st.expander("⚙️ Advanced Load Settings"):
        header_row_idx = st.number_input("Header Row Index", 0, 10, 0)
        encoding_option = st.selectbox("Encoding (CSV only)", ["Auto / UTF-8", "ISO-8859-1", "cp1252"])
        excel_sheet = st.text_input("Sheet Name (Excel)", value="", help="Leave empty for first sheet")
        skip_rows_excel = st.number_input("Skip Rows (Excel)", 0, 20, 0)
    
    st.divider()
    st.header("⚙️ 2. Analysis Configuration")

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_theme_colors():
    """Get color palette based on current theme"""
    if st.session_state['theme'] == 'dark':
        return {
            'bg': '#0e1117', 'text': '#fafafa', 'normal': '#1f77b4', 'outlier': '#d62728',
            'grid': '#4a5568', 'axis': '#fafafa', 'accent': '#e6a817'
        }
    return {
        'bg': '#ffffff', 'text': '#262730', 'normal': '#1f77b4', 'outlier': '#d62728',
        'grid': '#e0e0e0', 'axis': '#262730', 'accent': '#e6a817'
    }

def fig_to_bytes_matplotlib(fig, format='png', dpi=300):
    """Convert matplotlib figure to bytes for download"""
    try:
        buf = BytesIO()
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Matplotlib export failed: {e}")
        return None

def fig_to_bytes_plotly(fig, format='png', scale=2):
    """Convert plotly figure to bytes for download"""
    try:
        img_bytes = fig.to_image(format=format, scale=scale)
        return BytesIO(img_bytes)
    except Exception as e:
        st.warning(f"Plotly export requires kaleido: {e}")
        return None

def safe_export_button(fig, name, is_plotly=False, export_format='PNG', export_dpi=300):
    """Safe export button with error handling"""
    try:
        fmt = export_format.lower()
        mime = {'png': 'image/png', 'jpeg': 'image/jpeg', 'pdf': 'application/pdf'}.get(fmt, 'image/png')
        if is_plotly:
            buf = fig_to_bytes_plotly(fig, fmt)
        else:
            buf = fig_to_bytes_matplotlib(fig, fmt, export_dpi)
        if buf:
            st.download_button(f"📥 Download {name}.{fmt}", buf, f"{name}.{fmt}", mime)
    except Exception as e:
        st.error(f"Export failed: {e}")

def add_fine_grid(ax):
    """Add fine grid to matplotlib axes"""
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)

# ============================================================
# 4. TOOLTIPS DICTIONARY
# ============================================================

TOOLTIPS = {
    "Yeo-Johnson": "Power transformation supporting positive, negative, and zero values. Finds optimal λ to maximize normality.",
    "K-Means": "Partitions data into K groups by minimizing within-cluster variance. Requires pre-specification of K.",
    "DBSCAN": "Density-Based Spatial Clustering. Automatically identifies clusters and marks 'noisy' points.",
    "Mahalanobis Distance": "Multivariate distance metric accounting for correlations. Identifies outliers from rare combinations.",
    "Global Sensitivity": "Percentile of Chi-Square distribution for global outlier threshold. 95% = top 5% flagged.",
    "Smoothing Sigma": "Parameter σ of Gaussian filtering. Higher = smoother boundary; Lower = more responsive.",
    "Adaptive Boundary": "Dynamic outlier boundary varying by X-axis values. Useful for heteroscedastic data.",
    "Target Quantile": "Percentile of distances for adaptive boundary. 95% = line passes above 95% of points.",
    "Impute": "Strategy for missing values: Mean is sensitive to outliers, Median is more robust.",
    "Auto-Clean Text": "Cleans text (lowercase, strip) to prevent duplicate categories in encoding.",
    "Isolation Forest": "Tree-based anomaly detection. Isolates anomalies by random partitioning.",
    "Local Outlier Factor": "Density-based local outlier detection. Compares local density to neighbors.",
    "Robust Covariance": "Minimum Covariance Determinant - resistant to outliers in covariance estimation."
}

# ============================================================
# 5. COMPREHENSIVE STATISTICAL ANALYSIS CLASS
# ============================================================

class ComprehensiveStatisticalAnalyzer:
    """
    Comprehensive statistical testing framework for Quality Index model evaluation.
    Integrates all 12 sections from the original testing framework.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}
        self.test_reports = []
        self.colors = get_theme_colors()
    
    def run_all_tests(self):
        """Execute complete statistical testing battery"""
        self.descriptive_statistics()
        self.distribution_analysis()
        self.correlation_analysis()
        self.model_performance_metrics()
        self.residual_analysis()
        self.quality_score_analysis()
        self.outlier_detection()
        self.multivariate_outlier_detection_mahalanobis()
        self.feature_importance_analysis()
        self.statistical_significance_tests()
        self.confidence_intervals()
        self.segment_analysis()
        return self.results
    
    def descriptive_statistics(self):
        """Section 1: Descriptive Statistics"""
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            return
        
        desc_stats = self.df[numeric_cols].describe().T
        desc_stats['skewness'] = self.df[numeric_cols].skew()
        desc_stats['kurtosis'] = self.df[numeric_cols].kurtosis()
        desc_stats['missing_values'] = self.df[numeric_cols].isnull().sum()
        desc_stats['missing_pct'] = (self.df[numeric_cols].isnull().sum() / len(self.df) * 100).round(2)
        
        self.results['descriptive_stats'] = desc_stats
        self.test_reports.append({
            'section': 'Descriptive Statistics',
            'summary': f"Dataset contains {len(self.df)} observations with {len(numeric_cols)} numeric variables."
        })
    
    def distribution_analysis(self):
        """Section 2: Distribution Analysis and Normality Tests"""
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 3:
            return
        
        normality_results = []
        for var in numeric_cols[:10]:  # Limit to first 10 for performance
            data = self.df[var].dropna()
            if len(data) < 3:
                continue
            
            try:
                sw_stat, sw_pvalue = shapiro(data)
            except:
                sw_stat, sw_pvalue = np.nan, np.nan
            
            try:
                ks_stat, ks_pvalue = kstest(data, 'norm')
            except:
                ks_stat, ks_pvalue = np.nan, np.nan
            
            try:
                ad_result = anderson(data, dist='norm')
                ad_stat = ad_result.statistic
                ad_critical = ad_result.critical_values[2] if len(ad_result.critical_values) > 2 else np.nan
            except:
                ad_stat, ad_critical = np.nan, np.nan
            
            try:
                jb_stat, jb_pvalue = jarque_bera(data)
            except:
                jb_stat, jb_pvalue = np.nan, np.nan
            
            normality_results.append({
                'variable': var,
                'shapiro_stat': sw_stat,
                'shapiro_pvalue': sw_pvalue,
                'ks_stat': ks_stat,
                'ks_pvalue': ks_pvalue,
                'ad_stat': ad_stat,
                'ad_critical_5%': ad_critical,
                'jb_stat': jb_stat,
                'jb_pvalue': jb_pvalue,
                'skewness': data.skew(),
                'kurtosis': data.kurtosis()
            })
        
        if normality_results:
            self.results['normality_tests'] = pd.DataFrame(normality_results)
            self.test_reports.append({
                'section': 'Distribution Analysis',
                'summary': f"Normality tests completed for {len(normality_results)} variables."
            })
    
    def correlation_analysis(self):
        """Section 3: Correlation Analysis"""
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            return
        
        try:
            pearson_corr = self.df[numeric_cols].corr(method='pearson')
            spearman_corr = self.df[numeric_cols].corr(method='spearman')
            kendall_corr = self.df[numeric_cols].corr(method='kendall')
            
            self.results['correlation_analysis'] = {
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'kendall': kendall_corr
            }
            self.test_reports.append({
                'section': 'Correlation Analysis',
                'summary': f"Correlation matrices computed for {len(numeric_cols)} variables."
            })
        except Exception as e:
            st.warning(f"Correlation analysis failed: {e}")
    
    def model_performance_metrics(self):
        """Section 4: Model Performance Metrics"""
        if 'selling_price' not in self.df.columns or 'predicted_price' not in self.df.columns:
            return
        
        y_true = self.df['selling_price']
        y_pred = self.df['predicted_price']
        mask = y_true.notna() & y_pred.notna()
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) < 2:
            return
        
        try:
            r2 = r2_score(y_true_clean, y_pred_clean)
            adj_r2 = 1 - (1-r2) * (len(y_true_clean)-1)/(len(y_true_clean)-2)
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
            evs = explained_variance_score(y_true_clean, y_pred_clean)
            
            residuals = y_true_clean - y_pred_clean
            
            n = len(y_true_clean)
            k = 2
            rss = np.sum((y_true_clean - y_pred_clean)**2)
            aic = n * np.log(rss/n) + 2*k
            bic = n * np.log(rss/n) + k * np.log(n)
            
            self.results['model_performance'] = {
                'r2': r2, 'adjusted_r2': adj_r2, 'rmse': rmse, 'mae': mae,
                'mape': mape, 'explained_variance': evs,
                'residual_mean': residuals.mean(), 'residual_std': residuals.std(),
                'aic': aic, 'bic': bic
            }
            self.test_reports.append({
                'section': 'Model Performance',
                'summary': f"R²={r2:.4f}, RMSE=${rmse:,.2f}, MAE=${mae:,.2f}"
            })
        except Exception as e:
            st.warning(f"Model performance metrics failed: {e}")
    
    def residual_analysis(self):
        """Section 5: Residual Analysis"""
        if 'selling_price' not in self.df.columns or 'predicted_price' not in self.df.columns:
            return
        
        residuals = self.df['selling_price'] - self.df['predicted_price']
        mask = residuals.notna()
        residuals_clean = residuals[mask]
        y_pred_clean = self.df['predicted_price'][mask]
        
        if len(residuals_clean) < 10:
            return
        
        try:
            bp_test = het_breuschpagan(residuals_clean, y_pred_clean)
            heteroscedasticity_bp = bp_test[1] < 0.05
        except:
            heteroscedasticity_bp = False
        
        try:
            white_test = het_white(residuals_clean, y_pred_clean)
            heteroscedasticity_white = white_test[1] < 0.05
        except:
            heteroscedasticity_white = False
        
        dw_stat = durbin_watson(residuals_clean)
        autocorrelation = not (1.5 < dw_stat < 2.5)
        
        try:
            sw_stat, sw_pvalue = shapiro(residuals_clean)
        except:
            sw_pvalue = np.nan
        
        self.results['residual_analysis'] = {
            'heteroscedasticity_bp': heteroscedasticity_bp,
            'heteroscedasticity_white': heteroscedasticity_white,
            'durbin_watson': dw_stat,
            'autocorrelation': autocorrelation,
            'normality_pvalue': sw_pvalue,
            'residual_mean': residuals_clean.mean(),
            'residual_std': residuals_clean.std()
        }
        self.test_reports.append({
            'section': 'Residual Analysis',
            'summary': f"DW={dw_stat:.4f}, Heteroscedasticity: {'Yes' if heteroscedasticity_bp else 'No'}"
        })
    
    def quality_score_analysis(self):
        """Section 6: Quality Score Analysis"""
        quality_cols = [c for c in ['quality_heuristic', 'quality_residual', 'quality_hybrid'] if c in self.df.columns]
        if len(quality_cols) < 2:
            return
        
        try:
            quality_desc = self.df[quality_cols].describe().T
            quality_desc['range'] = quality_desc['max'] - quality_desc['min']
            quality_desc['cv'] = quality_desc['std'] / quality_desc['mean'] * 100
            
            self.results['quality_score_analysis'] = {'descriptive': quality_desc}
            self.test_reports.append({
                'section': 'Quality Score Analysis',
                'summary': f"Analyzed {len(quality_cols)} quality metrics."
            })
        except Exception as e:
            st.warning(f"Quality score analysis failed: {e}")
    
    def outlier_detection(self):
        """Section 7: Outlier Detection"""
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            return
        
        outlier_results = {}
        for col in numeric_cols[:10]:
            data = self.df[col].dropna()
            if len(data) < 10:
                continue
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            
            try:
                z_scores = np.abs(zscore(data))
                z_outliers = (z_scores > 3).sum()
            except:
                z_outliers = 0
            
            median = data.median()
            mad = np.median(np.abs(data - median))
            if mad > 0:
                modified_z_scores = 0.6745 * (data - median) / mad
                mad_outliers = (np.abs(modified_z_scores) > 3.5).sum()
            else:
                mad_outliers = 0
            
            outlier_results[col] = {
                'total_observations': len(data),
                'iqr_outliers': iqr_outliers,
                'iqr_outlier_pct': iqr_outliers / len(data) * 100,
                'zscore_outliers': z_outliers,
                'zscore_outlier_pct': z_outliers / len(data) * 100,
                'mad_outliers': mad_outliers,
                'mad_outlier_pct': mad_outliers / len(data) * 100
            }
        
        if outlier_results:
            self.results['outlier_detection'] = pd.DataFrame(outlier_results).T
            self.test_reports.append({
                'section': 'Outlier Detection',
                'summary': f"Outlier analysis completed for {len(outlier_results)} variables."
            })
    
    def multivariate_outlier_detection_mahalanobis(self):
        """Section 7.5: Advanced Multivariate Outlier Detection using Mahalanobis Distance"""
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 3:
            return
        
        clean_df = self.df[numeric_cols].dropna()
        if len(clean_df) < 10:
            return
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_df)
        
        try:
            cov_est = MinCovDet(random_state=42, support_fraction=0.9)
            cov_est.fit(scaled_data)
            mean = cov_est.location_
            cov = cov_est.covariance_
            cov_inv = np.linalg.inv(cov)
        except:
            mean = np.mean(scaled_data, axis=0)
            cov = np.cov(scaled_data, rowvar=False)
            cov_inv = pinv(cov)
        
        diff = scaled_data - mean
        distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        dist_series = pd.Series(distances, index=clean_df.index, name='mahalanobis_distance')
        
        dof = len(numeric_cols)
        threshold = np.sqrt(chi2.ppf(0.975, df=dof))
        outliers = dist_series > threshold
        
        self.results['mahalanobis_outliers'] = {
            'distances': dist_series,
            'threshold': threshold,
            'outlier_flags': outliers,
            'n_outliers': int(outliers.sum()),
            'outlier_pct': float(outliers.mean() * 100)
        }
        self.test_reports.append({
            'section': 'Mahalanobis Outlier Detection',
            'summary': f"Detected {outliers.sum()} multivariate outliers ({outliers.mean()*100:.2f}%)"
        })
    
    def feature_importance_analysis(self):
        """Section 8: Feature Importance Analysis"""
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            return
        
        target_cols = [c for c in ['quality_hybrid', 'selling_price'] if c in self.df.columns]
        if not target_cols:
            return
        
        importance_matrix = pd.DataFrame(index=numeric_cols[:10], columns=target_cols)
        for feature in numeric_cols[:10]:
            for target in target_cols:
                try:
                    corr = self.df[[feature, target]].corr().iloc[0, 1]
                    importance_matrix.loc[feature, target] = corr
                except:
                    importance_matrix.loc[feature, target] = np.nan
        
        self.results['feature_importance'] = {'correlation_matrix': importance_matrix.astype(float)}
        self.test_reports.append({
            'section': 'Feature Importance',
            'summary': f"Feature importance analysis completed for {len(target_cols)} targets."
        })
    
    def statistical_significance_tests(self):
        """Section 9: Statistical Significance Tests"""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        
        if not cat_cols or len(numeric_cols) < 2:
            return
        
        test_results = []
        for cat_col in cat_cols[:3]:
            unique_vals = self.df[cat_col].dropna().unique()
            if len(unique_vals) < 2 or len(unique_vals) > 10:
                continue
            
            for num_col in numeric_cols[:5]:
                groups = [self.df[self.df[cat_col] == val][num_col].dropna() for val in unique_vals[:5]]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    try:
                        if len(groups) == 2:
                            stat, p_value = mannwhitneyu(groups[0], groups[1])
                            test_name = 'Mann-Whitney U'
                        else:
                            stat, p_value = kruskal(*groups)
                            test_name = 'Kruskal-Wallis'
                        
                        test_results.append({
                            'categorical': cat_col,
                            'numeric': num_col,
                            'test': test_name,
                            'statistic': stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        })
                    except:
                        pass
        
        if test_results:
            self.results['significance_tests'] = pd.DataFrame(test_results)
            self.test_reports.append({
                'section': 'Statistical Significance',
                'summary': f"Completed {len(test_results)} significance tests."
            })
    
    def confidence_intervals(self):
        """Section 10: Confidence Intervals"""
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            return
        
        ci_results = []
        for metric in numeric_cols[:10]:
            data = self.df[metric].dropna()
            if len(data) < 10:
                continue
            
            n = len(data)
            mean = data.mean()
            std = data.std()
            se = std / np.sqrt(n)
            
            try:
                ci_95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
                ci_99 = stats.t.interval(0.99, df=n-1, loc=mean, scale=se)
            except:
                ci_95 = (np.nan, np.nan)
                ci_99 = (np.nan, np.nan)
            
            ci_results.append({
                'metric': metric,
                'mean': mean,
                'std': std,
                'n': n,
                'ci_95_lower': ci_95[0],
                'ci_95_upper': ci_95[1],
                'ci_99_lower': ci_99[0],
                'ci_99_upper': ci_99[1]
            })
        
        if ci_results:
            self.results['confidence_intervals'] = pd.DataFrame(ci_results)
            self.test_reports.append({
                'section': 'Confidence Intervals',
                'summary': f"Confidence intervals calculated for {len(ci_results)} metrics."
            })
    
    def segment_analysis(self):
        """Section 11: Segment-Based Analysis"""
        if 'quality_hybrid' not in self.df.columns:
            return
        
        try:
            self.df['quality_segment'] = pd.qcut(self.df['quality_hybrid'], q=4, 
                                                  labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                                                  duplicates='drop')
            segment_stats = self.df.groupby('quality_segment').agg({
                'selling_price': ['mean', 'std', 'count'] if 'selling_price' in self.df.columns else [],
                'quality_hybrid': ['mean', 'std', 'count']
            }).dropna(axis=1, how='all')
            
            self.results['segment_analysis'] = {'statistics': segment_stats}
            self.test_reports.append({
                'section': 'Segment Analysis',
                'summary': f"Segment analysis completed with {self.df['quality_segment'].nunique()} segments."
            })
        except Exception as e:
            st.warning(f"Segment analysis failed: {e}")

# ============================================================
# 6. DATA LOADING
# ============================================================

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
        st.error(f" Error loading file: {type(e).__name__}: {str(e)}"); st.stop()

    st.markdown("---")
    with st.expander("🔎 Data Filtering and Preview", expanded=True):
        filter_cols = st.multiselect("Select columns to filter by:", df_raw.columns.tolist())
        df_filtered = df_raw.copy()
        for col in filter_cols:
            if pd.api.types.is_numeric_dtype(df_filtered[col]):
                rng = st.slider(f"Range for: {col}", float(df_filtered[col].min()), 
                               float(df_filtered[col].max()), 
                               (float(df_filtered[col].min()), float(df_filtered[col].max())))
                df_filtered = df_filtered[df_filtered[col].between(rng[0], rng[1])]
            else:
                sel = st.multiselect(f"Values for: {col}", df_filtered[col].unique().tolist(), 
                                    default=df_filtered[col].unique().tolist()[:10])
                if sel: df_filtered = df_filtered[df_filtered[col].isin(sel)]
        st.info(f"📊 Rows after filtering: **{len(df_filtered):,}**")
        st.dataframe(df_filtered.head(50), height=150)

    if len(df_filtered) == 0: st.error("No data left after filtering."); st.stop()

    st.sidebar.subheader("📏 Row Range")
    start_row, end_row = st.sidebar.slider("Process rows:", 0, len(df_filtered), 
                                            (0, len(df_filtered)), 1)
    df = df_filtered.iloc[start_row:end_row].copy()
    st.sidebar.info(f"Active Rows: **{len(df):,}**")

    st.sidebar.subheader("🎯 Variables")
    all_cols = df.columns.tolist()
    potential = ['year', 'make', 'condition', 'odometer', 'mmr', 'sellingprice', 'mileage', 
                 'price', 'selling_price', 'predicted_price', 'quality_hybrid', 'quality_heuristic',
                 'quality_residual', 'car_age', 'manufacturer_reliability', 'residuals']
    default_cols = [c for c in all_cols if str(c).lower().strip() in [p.lower() for p in potential]]
    selected_cols = st.sidebar.multiselect("Select Model Variables (Min 2):", all_cols, default=default_cols)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎚️ Settings")
    global_percent = st.sidebar.slider("Global Sensitivity (%)", 80.0, 99.9, 95.0, 0.1, 
                                       help=TOOLTIPS["Global Sensitivity"])
    with st.sidebar.expander("📈 Smoothing Settings"):
        sigma_val = st.sidebar.slider("Smoothness Sigma (σ)", 1, 200, 50, 5, 
                                      help=TOOLTIPS["Smoothing Sigma"])
        quantile_target = st.sidebar.slider("Target Quantile (%)", 80, 99, 95, 1, 
                                            help=TOOLTIPS["Target Quantile"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Advanced Features")
    missing_strategy = st.sidebar.selectbox("Handle Missing Values", 
                                            ["Drop Rows", "Impute Mean", "Impute Median"], 
                                            help=TOOLTIPS["Impute"])
    
    transform_method = st.sidebar.selectbox("Data Transformation", 
                                            ["None", "Yeo-Johnson", "Log1p (Legacy)", "Robust Scaling"], 
                                            help=TOOLTIPS["Yeo-Johnson"])
    
    use_clustering = st.sidebar.checkbox("Enable Clustering", value=False)
    cluster_algo = "K-Means"
    if use_clustering:
        cluster_algo = st.sidebar.radio("Algorithm", ["K-Means", "DBSCAN"], 
                                        help=TOOLTIPS["K-Means"] + "\n\n" + TOOLTIPS["DBSCAN"])
        if cluster_algo == "K-Means":
            k_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
        else:
            eps_val = st.sidebar.slider("Epsilon (Distance)", 0.1, 5.0, 0.5, 0.1)
            min_samples_val = st.sidebar.slider("Min Samples", 2, 20, 5)
    
    auto_clean = st.sidebar.checkbox("Auto-Clean Text", value=True, help=TOOLTIPS["Auto-Clean Text"])

    st.sidebar.markdown("---")
    if st.sidebar.button(" Run Analysis", type="primary", use_container_width=True):
        if len(selected_cols) < 2:
            st.error("❌ Select at least 2 numeric variables.")
        else:
            with st.spinner('🧮 Calculating Statistics...'):
                try:
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
                    
                    if transform_method == "Yeo-Johnson":
                        num_cols_yj = df_work.select_dtypes(include=np.number).columns.tolist()
                        pt = PowerTransformer(method='yeo-johnson', standardize=True)
                        try:
                            transformed_data = pt.fit_transform(df_work[num_cols_yj])
                            df_work[num_cols_yj] = transformed_data
                            st.success(f"✅ Yeo-Johnson transformation applied to {len(num_cols_yj)} columns")
                        except Exception as e:
                            st.warning(f"⚠️ Yeo-Johnson failed: {e}. Using raw data.")
                    elif transform_method == "Log1p (Legacy)":
                        for c in df_work.select_dtypes(include=np.number).columns:
                            if (df_work[c] > 0).all():
                                df_work[c] = np.log1p(df_work[c])
                            else:
                                st.warning(f"️ Skipped log for '{c}' (contains <= 0).")
                    elif transform_method == "Robust Scaling":
                        num_cols_rs = df_work.select_dtypes(include=np.number).columns.tolist()
                        rs = RobustScaler()
                        df_work[num_cols_rs] = rs.fit_transform(df_work[num_cols_rs])
                        st.success(f"✅ Robust scaling applied to {len(num_cols_rs)} columns")
                    
                    try:
                        df_encoded = pd.get_dummies(df_work, drop_first=True, dtype=int)
                        if df_encoded.shape[0] <= df_encoded.shape[1]:
                            st.error(f"❌ Need more rows ({df_encoded.shape[0]}) than columns ({df_encoded.shape[1]})."); st.stop()
                        data = df_encoded.values
                        mu = np.mean(data, axis=0)
                        cov = np.cov(data.T)
                        if np.isnan(cov).any() or np.isinf(cov).any(): st.error("❌ Invalid covariance matrix."); st.stop()
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
                        'total': total_obs, 'outliers': int(outlier_count), 
                        'normal': int(total_obs - outlier_count),
                        'pct': (outlier_count / total_obs) * 100, 
                        'threshold': global_thresh, 'dimensions': dims
                    }
                    st.session_state['df_work'] = df_work
                    st.session_state['df_encoded'] = df_encoded
                    
                    # Run comprehensive statistical analysis
                    with st.spinner('📊 Running comprehensive statistical analysis...'):
                        analyzer = ComprehensiveStatisticalAnalyzer(df_res)
                        stat_results = analyzer.run_all_tests()
                        st.session_state['stat_results'] = stat_results
                        st.session_state['test_reports'] = analyzer.test_reports
                    
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ Critical error: {type(e).__name__}: {str(e)}")
                    st.error(traceback.format_exc())

    # ============================================================
    # 7. MAIN TABS
    # ============================================================
    
    tab1, tab2 = st.tabs(["🔍 Outlier Detection", "📊 Advanced Statistical Analysis"])

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
            st.subheader("📊 Analysis Summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("📈 Total Observations", f"{stats['total']:,}")
            m2.metric("🔴 Outliers Detected", f"{stats['outliers']:,}", delta=f"{stats['pct']:.2f}%")
            m3.metric("🟢 Normal Points", f"{stats['normal']:,}", delta=f"{100 - stats['pct']:.2f}%")
            m4.metric("🎯 Threshold (χ²)", f"{stats['threshold']:.3f}")
            m5.metric("📐 Dimensions", stats['dimensions'])

            with st.expander("📋 Detailed Breakdown"):
                c1, c2 = st.columns(2)
                c1.markdown("**Status Distribution:**"); c1.dataframe(res['Status_Global'].value_counts())
                if 'Cluster' in res.columns:
                    c2.markdown("**Cluster Distribution:**"); c2.dataframe(res['Cluster'].value_counts())
                st.markdown("**Distance Statistics:**"); st.dataframe(res['Mahalanobis_Dist'].describe())

            st.markdown("---")
            st.subheader(" Visualization Console")
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
                palette = {c: sns.color_palette("tab10", len(res['Cluster'].unique()))[i].as_hex() 
                          for i, c in enumerate(sorted(res['Cluster'].unique()))}
                palette['Noise'] = '#000000'

            if plot_type == "Group Analysis":
                grp_col = st.selectbox("Group By:", ["Cluster"] + cat_cols if 'Cluster' in res.columns else cat_cols)
                if grp_col and numeric_cols:
                    fig = px.scatter(df_viz, x=numeric_cols[0], y='Mahalanobis_Dist', color=grp_col, 
                                    title=f"Group Analysis: {grp_col}")
                    fig.add_hline(y=global_thresh, line_dash="dash", line_color="red", annotation_text="Threshold")
                    st.plotly_chart(fig, use_container_width=True)
                    safe_export_button(fig, "group_analysis", True, export_fmt, export_dpi)

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
                    ax.set_title(f"Adaptive Boundary (σ={sigma_val})", color=colors['text'])
                    ax.tick_params(colors=colors['text']); ax.legend(); ax.grid(True, alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); safe_export_button(fig, "adaptive_boundary", False, export_fmt, export_dpi)

            elif plot_type == "Custom Scatter":
                if len(numeric_cols) >= 2:
                    x_v, y_v = st.columns(2)
                    x_col = x_v.selectbox("X Axis:", numeric_cols, index=0)
                    y_col = y_v.selectbox("Y Axis:", numeric_cols, index=1)
                    fig, ax = plt.subplots(figsize=(10, 6), facecolor=colors['bg']); ax.set_facecolor(colors['bg'])
                    sns.scatterplot(data=df_viz, x=x_col, y=y_col, hue=color_by, palette=palette, alpha=0.6, ax=ax)
                    ax.set_title(f"{x_col} vs {y_col}", color=colors['text']); ax.tick_params(colors=colors['text'])
                    plt.tight_layout(); st.pyplot(fig); safe_export_button(fig, f"scatter_{x_col}_{y_col}", False, export_fmt, export_dpi)

            elif plot_type == "Distance vs Variables":
                if numeric_cols:
                    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4*len(numeric_cols)), facecolor=colors['bg'])
                    if len(numeric_cols) == 1: axes = [axes]
                    for i, col in enumerate(numeric_cols):
                        axes[i].set_facecolor(colors['bg'])
                        sns.scatterplot(data=df_viz, x=col, y='Mahalanobis_Dist', hue=color_by, palette=palette, alpha=0.6, ax=axes[i])
                        axes[i].axhline(global_thresh, color='red', linestyle='--', label='Threshold')
                        axes[i].set_title(f"Distance vs {col}", color=colors['text']); axes[i].tick_params(colors=colors['text'])
                    plt.tight_layout(); st.pyplot(fig); safe_export_button(fig, "distance_vars", False, export_fmt, export_dpi)

            elif plot_type == "Pair Plot":
                if numeric_cols:
                    st.info("Generating Pair Plot (may take a moment)...")
                    plot_cols = numeric_cols[:4] + ['Mahalanobis_Dist']
                    g = sns.pairplot(df_viz.sample(min(1000, len(df_viz))), vars=plot_cols, hue=color_by, palette=palette, plot_kws={'alpha': 0.6})
                    st.pyplot(g.fig); safe_export_button(g.fig, "pairplot", False, export_fmt, export_dpi)

            elif plot_type == "Contribution Analysis":
                outs = res[res['Status_Global'] == 'Outlier'].nlargest(1, 'Mahalanobis_Dist')
                if not outs.empty:
                    st.write(f"### 🔬 Top Outlier Analysis (Index: {outs.index[0]})")
                    z_scores = {c: (outs.iloc[0][c] - res[c].mean()) / res[c].std() if res[c].std() > 0 else 0 for c in numeric_cols}
                    df_z = pd.DataFrame(list(z_scores.items()), columns=['Var', 'Z']).sort_values('Z', key=abs, ascending=False)
                    fig = px.bar(df_z, x='Z', y='Var', orientation='h', color='Z', color_continuous_scale='RdBu_r', title="Z-Score Drivers")
                    st.plotly_chart(fig, use_container_width=True); safe_export_button(fig, "contribution", True, export_fmt, export_dpi)

            elif plot_type == "3D View":
                if len(numeric_cols) >= 3:
                    x3, y3, z3 = st.columns(3)
                    cx = x3.selectbox("X", numeric_cols, 0); cy = y3.selectbox("Y", numeric_cols, 1); cz = z3.selectbox("Z", numeric_cols, 2)
                    fig = px.scatter_3d(df_viz, x=cx, y=cy, z=cz, color=color_by, color_discrete_map=palette, opacity=0.7, title="3D Mahalanobis Space")
                    st.plotly_chart(fig, use_container_width=True); safe_export_button(fig, "3d_view", True, export_fmt, export_dpi)

            st.markdown("---")
            st.subheader("📥 Download Results")
            d1, d2 = st.columns(2)
            d1.download_button(" Download CSV", res.to_csv(index=False).encode('utf-8'), "results.csv", "text/csv")
            try:
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    res.to_excel(w, sheet_name='Data', index=False)
                    pd.DataFrame([stats]).to_excel(w, sheet_name='Summary', index=False)
                d2.download_button("📊 Download Excel", buf.getvalue(), "results.xlsx", 
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except: d2.warning("Excel export requires `openpyxl`.")
        else:
            st.info(" Run analysis to view results.")

    with tab2:
        st.subheader("📊 Advanced Statistical Analysis")
        st.markdown("Comprehensive statistical testing framework with 12 analysis sections.")

        if 'stat_results' in st.session_state:
            stat_results = st.session_state['stat_results']
            test_reports = st.session_state['test_reports']
            
            stat_tab1, stat_tab2, stat_tab3, stat_tab4, stat_tab5, stat_tab6 = st.tabs([
                " Descriptive", "🔬 Normality", "🔗 Correlation",
                " Model Performance", "🎯 Outlier Detection", "📋 Summary Report"
            ])

            with stat_tab1:
                st.subheader("Descriptive Statistics")
                if 'descriptive_stats' in stat_results:
                    st.dataframe(stat_results['descriptive_stats'])
                else:
                    st.info("No descriptive statistics available.")

            with stat_tab2:
                st.subheader("Normality Tests")
                if 'normality_tests' in stat_results:
                    st.dataframe(stat_results['normality_tests'])
                else:
                    st.info("No normality tests available.")

            with stat_tab3:
                st.subheader("Correlation Analysis")
                if 'correlation_analysis' in stat_results:
                    corr_data = stat_results['correlation_analysis']
                    corr_method = st.radio("Select correlation method:", ["pearson", "spearman", "kendall"], horizontal=True)
                    if corr_method in corr_data:
                        st.dataframe(corr_data[corr_method].round(3))
                        fig, ax = plt.subplots(figsize=(12, 10))
                        mask = np.triu(np.ones_like(corr_data[corr_method], dtype=bool))
                        sns.heatmap(corr_data[corr_method], mask=mask, annot=True, cmap='coolwarm', 
                                   center=0, fmt='.3f', linewidths=.5, ax=ax)
                        plt.title(f'{corr_method.capitalize()} Correlation Heatmap')
                        plt.tight_layout(); st.pyplot(fig)
                else:
                    st.info("No correlation analysis available.")

            with stat_tab4:
                st.subheader("Model Performance Metrics")
                if 'model_performance' in stat_results:
                    perf = stat_results['model_performance']
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R²", f"{perf['r2']:.4f}")
                    col2.metric("Adjusted R²", f"{perf['adjusted_r2']:.4f}")
                    col3.metric("RMSE", f"${perf['rmse']:,.2f}")
                    col4, col5, col6 = st.columns(3)
                    col4.metric("MAE", f"${perf['mae']:,.2f}")
                    col5.metric("MAPE", f"{perf['mape']:.2f}%")
                    col6.metric("Explained Variance", f"{perf['explained_variance']:.4f}")
                else:
                    st.info("No model performance metrics available.")

            with stat_tab5:
                st.subheader("Outlier Detection")
                if 'outlier_detection' in stat_results:
                    st.dataframe(stat_results['outlier_detection'])
                if 'mahalanobis_outliers' in stat_results:
                    mahal = stat_results['mahalanobis_outliers']
                    st.markdown(f"**Mahalanobis Outliers:** {mahal['n_outliers']} ({mahal['outlier_pct']:.2f}%)")
                    st.markdown(f"**Threshold:** {mahal['threshold']:.4f}")
                else:
                    st.info("No outlier detection results available.")

            with stat_tab6:
                st.subheader("Comprehensive Summary Report")
                if test_reports:
                    for report in test_reports:
                        st.markdown(f"**{report['section']}:** {report['summary']}")
                        st.markdown("---")
                else:
                    st.info("No summary report available.")
        else:
            st.info("👆 Run analysis to view statistical results.")

else:
    st.info(" Upload a file to begin.")
    st.markdown("""
    ### 📖 About Ultimate Outlier Detector Pro
    An advanced multivariate tool based on **Mahalanobis Distance** with comprehensive statistical analysis capabilities.
    
    **Features:** 
    -  Dark Mode
    - 📊 CSV/Excel Support
    - 🎨 7 Visualization Modes
    - 🤖 Clustering (K-Means/DBSCAN)
    - 📈 Adaptive Boundaries
    - 📥 Multi-Format Export
    - 🔬 12-Section Statistical Analysis Framework
    - 🔄 Yeo-Johnson Transformation
    """)

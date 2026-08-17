import streamlit as st
import pandas as pd
import base64
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# 1. Page Configuration
st.set_page_config(
    page_title="Traffic Analysis Dashboard - DBSCAN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to convert local image to base64
def get_base64_image(image_path):
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception:
            return None
    return None

# 2. Session State Initialization
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'light'

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Overview'

if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None

if 'df_preprocessed' not in st.session_state:
    st.session_state.df_preprocessed = None    

if 'dbscan_res' not in st.session_state:
    st.session_state.dbscan_res = None    

def toggle_theme():
    if st.session_state.theme_mode == 'light':
        st.session_state.theme_mode = 'dark'
    else:
        st.session_state.theme_mode = 'light'

def set_page(page_name):
    st.session_state.current_page = page_name

def render_themed_table(df, index_name="#"):
    """Render DataFrame as a custom HTML table styled 100% via CSS variables
    (light/dark mode support)."""
    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    rows_html = ""
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{val}</td>" for val in row)
        rows_html += f"<tr><td class='row-idx'>{idx}</td>{cells}</tr>"

    table_html = f"""
    <div class="themed-table-wrapper">
        <table class="themed-table">
            <thead>
                <tr><th class='row-idx'>{index_name}</th>{headers}</tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return table_html

# 3. Custom CSS Injection
def apply_custom_css():
    if st.session_state.theme_mode == 'light':
        theme_vars = """
        :root {
            --bg-main: #F8FAFC;
            --bg-card: #FFFFFF;
            --bg-sidebar: #FFFFFF;
            --text-primary: #1E293B;
            --text-secondary: #64748B;
            --border-color: #E2E8F0;
            --primary-accent: #2563EB;
            --hover-bg: #F1F5F9;
            --dropzone-bg: rgba(37, 99, 235, 0.05);
            --dropzone-border: #2563EB;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        """
    else:
        theme_vars = """
        :root {
            --bg-main: #0F172A;
            --bg-card: #1E293B;
            --bg-sidebar: #1E293B;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --border-color: #334155;
            --primary-accent: #60A5FA;
            --hover-bg: #334155;
            --dropzone-bg: rgba(96, 165, 250, 0.1);
            --dropzone-border: #60A5FA;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
        }
        """

    css_styles = f"""
    <style>
        {theme_vars}

        /* Global Background & Typography */
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: var(--bg-main) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}

        /* Header Bar Styling */
        [data-testid="stHeader"] {{
            background-color: var(--bg-main) !important;
            border-bottom: 1px solid var(--border-color);
        }}

        /* Arrow collapse/expand sidebar */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        [data-testid="stExpandSidebarButton"],
        [data-testid="baseButton-headerNoPadding"],
        [data-testid="stHeader"] button,
        [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"] {{
            color: #334155 !important;
        }}
        [data-testid="stSidebarCollapseButton"] svg,
        [data-testid="collapsedControl"] svg,
        [data-testid="stExpandSidebarButton"] svg,
        [data-testid="baseButton-headerNoPadding"] svg,
        [data-testid="stHeader"] svg,
        [data-testid="stHeader"] button svg,
        [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"] svg {{
            fill: #334155 !important;
            color: #334155 !important;
            opacity: 1 !important;
        }}
        [data-testid="stSidebarCollapseButton"]:hover svg,
        [data-testid="collapsedControl"]:hover svg,
        [data-testid="stExpandSidebarButton"]:hover svg,
        [data-testid="stHeader"] button:hover svg {{
            fill: var(--primary-accent) !important;
            color: var(--primary-accent) !important;
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-color) !important;
        }}
        
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            overflow-y: hidden !important; 
            overflow-x: hidden !important;
        }}

        /* Header Branding Container */
        .sidebar-header-container {{
            text-align: center;
            padding: 0 0 12px 0 !important;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 12px;
        }}
        .sidebar-header-container img {{
            width: 90px;
            height: auto;
            margin-bottom: 8px;
        }}

        /* Navigation Section Title */
        .nav-section-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px !important;
            padding-left: 5px;
            display: block;
        }}

        /* Navigation Buttons */
        div.element-container:has(.stButton) + div.element-container:has(.stButton) {{
            margin-top: -12px !important;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            width: 100% !important;
            background-color: transparent !important;
            color: var(--text-primary) !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            display: flex !important;
            justify-content: flex-start !important; 
            align-items: center !important;
            text-align: left !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease-in-out !important;
        }}

        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] .stButton > button p,
        [data-testid="stSidebar"] .stButton > button div {{
            text-align: left !important;
            justify-content: flex-start !important;
            align-items: center !important;
            width: 100% !important;
            display: block !important;
            margin: 0 !important;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background-color: var(--hover-bg) !important;
            color: var(--primary-accent) !important;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background-color: var(--hover-bg) !important;
            color: var(--primary-accent) !important;
            font-weight: 600 !important;
            border-left: 4px solid var(--primary-accent) !important;
            border-radius: 4px 6px 6px 4px !important;
        }}

        /* Theme Toggle Button Styling */
        div.element-container:has(.sidebar-footer-container) + div.element-container .stButton > button {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            justify-content: center !important; 
            text-align: center !important;
            box-shadow: var(--shadow-sm) !important;
            border-left: 1px solid var(--border-color) !important; 
            margin-top: 15px !important; 
        }}

        div.element-container:has(.sidebar-footer-container) + div.element-container .stButton > button *,
        div.element-container:has(.sidebar-footer-container) + div.element-container .stButton > button p {{
            text-align: center !important;
            justify-content: center !important;
            margin: 0 auto !important;
        }}

        div.element-container:has(.sidebar-footer-container) + div.element-container .stButton > button:hover {{
            border-color: var(--primary-accent) !important;
            background-color: var(--hover-bg) !important;
        }}

        /* Sidebar Footer Container */
        .sidebar-footer-container {{
            margin-top: 280px !important; 
            border-top: 1px solid var(--border-color);
            padding-top: 10px;
        }}
        
        .sidebar-copyright {{
            font-size: 0.72rem;
            color: var(--text-secondary);
            line-height: 1.4;
            text-align: left !important;
            padding-left: 6px;
        }}

        /* File Uploader Widget Styling */
        div[data-testid="stFileUploader"] {{
            background-color: transparent !important;
        }}

        div[data-testid="stFileUploader"] > div:first-child *,
        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] label p,
        div[data-testid="stFileUploader"] label span {{
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }}

        div[data-testid="stFileUploader"] section,
        [data-testid="stFileUploadDropzone"] {{
            background-color: var(--dropzone-bg) !important;
            border: 2px dashed var(--dropzone-border) !important;
            border-radius: 10px !important;
            padding: 24px !important;
            transition: all 0.25s ease-in-out !important;
        }}

        div[data-testid="stFileUploader"] section *:not(button):not(svg),
        [data-testid="stFileUploadDropzone"] *:not(button):not(svg) {{
            color: var(--text-primary) !important;
            background-color: transparent !important;
        }}

        div[data-testid="stFileUploader"] section small,
        [data-testid="stFileUploadDropzone"] small {{
            color: var(--text-secondary) !important;
        }}

        div[data-testid="stFileUploader"] section:hover,
        div[data-testid="stFileUploader"] section:focus-within {{
            background-color: var(--dropzone-bg) !important;
            border-color: var(--dropzone-border) !important;
        }}

        div[data-testid="stFileUploader"] section button {{
            background-color: var(--bg-card) !important;
            color: var(--primary-accent) !important;
            border: 1px solid var(--primary-accent) !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
        }}

        div[data-testid="stFileUploader"] section button p,
        div[data-testid="stFileUploader"] section button span {{
            color: var(--primary-accent) !important;
            background-color: transparent !important;
        }}

        div[data-testid="stFileUploader"] section button:hover,
        div[data-testid="stFileUploader"] section button:focus {{
            background-color: var(--primary-accent) !important;
            color: #FFFFFF !important;
            border-color: var(--primary-accent) !important;
        }}

        div[data-testid="stFileUploader"] section button:hover p,
        div[data-testid="stFileUploader"] section button:focus p,
        div[data-testid="stFileUploader"] section button:hover span,
        div[data-testid="stFileUploader"] section button:focus span,
        div[data-testid="stFileUploader"] section button:hover svg,
        div[data-testid="stFileUploader"] section button:focus svg {{
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }}

        [data-testid="stFileUploadDropzone"] > svg,
        div[data-testid="stFileUploader"] section > svg {{
            display: none !important;
        }}

        [data-testid="stFileUploaderFile"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }}

        [data-testid="stFileUploaderFile"] *:not(button):not(svg) {{
            color: var(--text-primary) !important;
            background-color: transparent !important;
        }}

        [data-testid="stFileUploaderFile"] small {{
            color: var(--text-secondary) !important;
        }}

        [data-testid="stFileUploaderFile"] button {{
            background-color: transparent !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-secondary) !important;
        }}

        [data-testid="stFileUploaderFile"] button svg {{
            fill: var(--text-secondary) !important;
            color: var(--text-secondary) !important;
        }}

        [data-testid="stFileUploaderFile"] button:hover {{
            border-color: var(--primary-accent) !important;
        }}

        [data-testid="stFileUploaderFile"] button:hover svg {{
            fill: var(--primary-accent) !important;
            color: var(--primary-accent) !important;
        }}

        [data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileIcon"] {{
            background-color: transparent !important;
        }}

        [data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileIcon"] svg {{
            color: var(--primary-accent) !important;
            fill: var(--primary-accent) !important;
        }}
        
        div[data-testid="stExpander"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: none !important;
        }}

        div[data-testid="stExpander"] details {{
            border: none !important;
            background-color: transparent !important;
        }}

        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary:hover,
        div[data-testid="stExpander"] summary:focus,
        div[data-testid="stExpander"] summary:active,
        div[data-testid="stExpander"] details[open] summary {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border-bottom: none !important;
        }}

        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] summary svg {{
            color: var(--text-primary) !important;
            fill: var(--text-primary) !important;
            font-weight: 600 !important;
        }}

        [data-testid="stExpanderDetails"] {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border-top: 1px dashed var(--border-color) !important; /* Garis pemisah */
        }}

        [data-testid="stExpanderDetails"] p, 
        [data-testid="stExpanderDetails"] li, 
        [data-testid="stExpanderDetails"] span {{
            color: var(--text-primary) !important;
        }}

        [data-testid="stExpanderDetails"] code {{
            background-color: var(--hover-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: monospace !important;
        }}


        /* Plotly Chart Container */
        div:has(> [data-testid="stPlotlyChart"]) {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            padding: 8px !important;
        }}

        /* Custom Table Styling */
        .themed-table-wrapper {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow-x: auto;
            max-height: 320px;
            overflow-y: auto;
        }}

        table.themed-table {{
            width: 100%;
            border-collapse: separate !important;
            border-spacing: 0;
            font-size: 0.82rem;
            margin-bottom: 0 !important; /* Menghilangkan spasi kosong di dasar tabel */
        }}  

        table.themed-table thead th {{
            background-color: var(--hover-bg);
            color: var(--text-primary);
            font-weight: 600;
            text-align: right;
            padding: 8px 12px;
            border-bottom: none !important; 
            box-shadow: 0 -1px 0 var(--hover-bg), 0 1px 0 var(--border-color) !important;
            position: sticky;
            z-index: 10;
            top: 0;
        }}

        table.themed-table thead th.row-idx,
        table.themed-table tbody td.row-idx {{
            text-align: left;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        table.themed-table tbody td {{
            color: var(--text-primary);
            text-align: right;
            padding: 7px 12px;
            border-bottom: 1px solid var(--border-color);
            white-space: nowrap;
        }}

        table.themed-table tbody tr:last-child td {{
            border-bottom: none;
        }}

        table.themed-table tbody tr:hover td {{
            background-color: var(--hover-bg);
        }}

        /* Primary Button Styling */
        div[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"],
        section[data-testid="stMain"] .stButton > button[kind="primary"] {{
            background-color: var(--primary-accent) !important;
            border: 1px solid var(--primary-accent) !important;
            color: #FFFFFF !important;
        }}

        div[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"]:hover,
        section[data-testid="stMain"] .stButton > button[kind="primary"]:hover {{
            background-color: var(--primary-accent) !important;
            opacity: 0.9;
            border-color: var(--primary-accent) !important;
        }}

        div[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"] *,
        section[data-testid="stMain"] .stButton > button[kind="primary"] * {{
            color: #FFFFFF !important;
        }}
        
        div[data-testid="stDateInput"] label p,
        div[data-testid="stDateInput"] label span {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}
        
        div[data-testid="stDateInput"] > div {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
            overflow: hidden !important;
        }}
        
        div[data-testid="stDateInput"] div[data-baseweb="input"],
        div[data-testid="stDateInput"] div[data-baseweb="base-input"],
        div[data-testid="stDateInput"] input,
        div[data-testid="stDateInput"] div[data-baseweb="base-input"] > div {{
            background-color: transparent !important;
            color: var(--text-primary) !important;
            border: none !important;
            box-shadow: none !important;
        }}
        
        div[data-testid="stDateInput"] svg {{
            fill: var(--text-secondary) !important;
        }}

        div[data-baseweb="popover"] > div,
        div[data-baseweb="calendar"] {{
            background-color: #FFFFFF !important; 
            border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important;
        }}

        div[data-baseweb="calendar"] * {{
            color: #1E293B !important;
        }}

        div[data-baseweb="calendar"] [role="presentation"],
        div[data-baseweb="calendar"] header {{
            background-color: transparent !important;
        }}

        div[data-baseweb="calendar"] button {{
            background-color: transparent !important;
        }}

        div[data-baseweb="calendar"] button svg {{
            fill: #1E293B !important;
            color: #1E293B !important;
        }}

        div[data-baseweb="calendar"] div[role="gridcell"] {{
            color: #1E293B !important;
            background-color: transparent !important;
        }}

        div[data-baseweb="calendar"] [role="gridcell"]:not([aria-selected="true"]):hover,
        div[data-baseweb="calendar"] button:hover {{
            color: #2563EB !important;
            font-weight: bold !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }}

        div[data-baseweb="calendar"] [role="gridcell"][aria-selected="true"] {{
            background-color: #2563EB !important;
            color: #FFFFFF !important;
        }}

        div[data-baseweb="calendar"] [aria-disabled="true"],
        div[data-baseweb="calendar"] [aria-disabled="true"] > div,
        div[data-baseweb="calendar"] [role="gridcell"]:empty {{
            background-color: transparent !important;
            color: #64748B !important; 
            opacity: 0.3 !important;
            border: none !important;
            pointer-events: none !important;
        }}

        /* Metric Card Container */
        .metric-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            box-shadow: var(--shadow-sm);
        }}

        /* --- MODULAR PREPROCESSING CARDS --- */
        .prep-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--primary-accent);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
            box-shadow: var(--shadow-sm);
        }}

        .prep-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 6px;
            display: flex;
            align-items: center;
        }}

        .prep-desc {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 12px;
            line-height: 1.5;
        }}

        .var-badge {{
            background-color: var(--hover-bg);
            color: var(--primary-accent);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
            display: inline-block;
            border: 1px solid var(--border-color);
        }}

        .log-box {{
            background-color: var(--bg-main);
            border: 1px dashed var(--border-color);
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 12px;
        }}

        .log-item {{
            margin-bottom: 4px;
        }}
        
        [data-testid="stAlert"] {{
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            border-radius: 9px !important;
            border: 1px solid var(--border-color) !important;
            box-shadow: var(--shadow-sm) !important;
        }}
        
        [data-testid="stAlert"] * {{
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }}
        
        [data-testid="stAlert"]:has(svg[data-testid="stIconWarning"]) {{
            background-color: rgba(245, 158, 11, 0.25) !important; 
        }}
        [data-testid="stAlert"]:has(svg[data-testid="stIconWarning"]) svg {{
            fill: #D97706 !important;
            color: #D97706 !important;
        }}

        [data-testid="stAlert"]:has(svg[data-testid="stIconSuccess"]) {{
            background-color: rgba(16, 185, 129, 0.25) !important;
        }}

        [data-testid="stAlert"]:has(svg[data-testid="stIconSuccess"]) svg {{
            fill: #059669 !important;
            color: #059669 !important;
        }}

        [data-testid="stAlert"]:has(svg[data-testid="stIconError"]) {{
            background-color: rgba(239, 68, 68, 0.25) !important;
        }}

        [data-testid="stAlert"]:has(svg[data-testid="stIconError"]) svg {{
            fill: #DC2626 !important;
            color: #DC2626 !important;
        }}
        
        [data-testid="stAlert"]:has(svg[data-testid="stIconInfo"]) {{
            background-color: rgba(59, 130, 246, 0.25) !important;
        }}
        [data-testid="stAlert"]:has(svg[data-testid="stIconInfo"]) svg {{
            fill: #2563EB !important;
            color: #2563EB !important;
        }}

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
            background-color: var(--primary-accent) !important;
        }}
        [data-testid="stTabs"] [data-baseweb="tab-border"] {{
            background-color: var(--border-color) !important;
        }}

        [data-testid="stTabs"] button p {{
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }}

        [data-testid="stTabs"] button[aria-selected="true"] p {{
            color: var(--primary-accent) !important;
        }}

        [data-testid="stTabs"] button:hover p {{
            color: var(--primary-accent) !important;
        }}
        
        div[data-baseweb="select"] > div {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            cursor: pointer !important;
        }}

        div[data-baseweb="select"] > div:hover {{
            border-color: var(--primary-accent) !important;
        }}

        div[data-baseweb="select"] > div:focus,
        div[data-baseweb="select"] > div:focus-within {{
            box-shadow: none !important;
        }}

        div[data-baseweb="select"] * {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}

        div[data-baseweb="select"] svg {{
            fill: var(--text-secondary) !important;
        }}

        ul[data-baseweb="menu"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 4px !important;
        }}

        ul[data-baseweb="menu"] li {{
            color: var(--text-primary) !important;
            background-color: transparent !important;
            border-radius: 4px !important;
            font-weight: 500 !important;
        }}

        ul[data-baseweb="menu"] li:hover,
        ul[data-baseweb="menu"] li[aria-selected="true"] {{
            background-color: var(--hover-bg) !important;
            color: var(--primary-accent) !important;
            font-weight: bold !important;
        }}

        div[data-testid="stSlider"] label p,
        div[data-testid="stSlider"] label span {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}
        
        div[data-testid="stSlider"] div[data-baseweb="slider"] [data-testid="stTickBarMin"],
        div[data-testid="stSlider"] div[data-baseweb="slider"] [data-testid="stTickBarMax"],
        div[data-testid="stSlider"] div[data-baseweb="slider"] [role="slider"] > div {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}

        /* Slider Track Customization */
        div[data-baseweb="slider"] [role="slider"] {{
            background-color: #3B82F6 !important;
            border: none !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }}

        div[data-baseweb="slider"] [role="slider"]:hover,
        div[data-baseweb="slider"] [role="slider"]:focus,
        div[data-baseweb="slider"] [role="slider"]:active {{
            box-shadow: 0 0 0 0.2rem rgba(59, 130, 246, 0.4) !important;
            outline: none !important;
        }}
        
        div[data-baseweb="slider"] > div > div > div > div:first-child {{
            background-color: #3B82F6 !important;
        }}

        div[data-testid="stNumberInput"] label p,
        div[data-testid="stNumberInput"] label span {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}

        div[data-testid="stNumberInput"] > div {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
            overflow: hidden !important;
        }}

        div[data-testid="stNumberInput"] div[data-baseweb="input"],
        div[data-testid="stNumberInput"] div[data-baseweb="base-input"],
        div[data-testid="stNumberInput"] input,
        div[data-testid="stNumberInput"] div[data-baseweb="base-input"] > div {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: none !important;
            box-shadow: none !important;
        }}

        div[data-testid="stNumberInput"] button {{
            background-color: var(--hover-bg) !important;
            border-left: 1px solid var(--border-color) !important;
            border-right: none !important;
            border-top: none !important;
            border-bottom: none !important;
        }}

        div[data-testid="stNumberInput"] button:hover {{
            background-color: var(--dropzone-bg) !important;
        }}

        div[data-testid="stNumberInput"] button svg {{
            fill: var(--text-primary) !important;
            color: var(--text-primary) !important;
        }}
    </style>
    """
    st.markdown(css_styles, unsafe_allow_html=True)

# 4. Menu Handlers
def interpret_traffic_condition(vc_val, speed_val):
    if vc_val >= 0.75:
        vc = "high"
    elif vc_val <= 0.50:
        vc = "low"
    else:
        vc = "medium"
        
    if speed_val >= 60.0:
        speed = "high"
    elif speed_val <= 25.0:
        speed = "low"
    else:
        speed = "medium"
    
    if vc == "high":
        if speed == "high":
            return "Busy Traffic"
        else:
            return "Heavy / Congested"
    elif vc == "low":
        if speed == "low":
            return "Slow Traffic"
        elif speed == "medium":
            return "Stable Traffic"
        else:
            return "Free Flow"
    else:
        if speed == "low":
            return "Starting to Congest"
        else:
            return "Moderate Traffic"

def show_overview():
    st.markdown("### Data Overview")
    st.markdown("Welcome to the **Heavy Vehicle Operational Pattern Dashboard**. This menu provides key performance metrics, dataset summaries, and general analytical insights.")

    st.markdown("""
        <style>
        [data-stale="true"] {
            opacity: 0 !important;
            display: none !important;
            transition: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state.get('df_preprocessed') is None or st.session_state.get('dbscan_res') is None:
        st.info("Welcome! Please upload your dataset and run the clustering model in the **Model Evaluation** menu to unlock the full overview dashboard.", icon="👋")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><h4>Total Observations</h4><p style="font-size: 1.8rem; font-weight:700; margin:0;">-</p><span style="font-size:0.8rem; color:var(--text-secondary);">Waiting for data...</span></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h4>Formed Clusters</h4><p style="font-size: 1.8rem; font-weight:700; margin:0;">-</p><span style="font-size:0.8rem; color:var(--text-secondary);">Waiting for data...</span></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h4>Primary Algorithm</h4><p style="font-size: 1.8rem; font-weight:700; margin:0;">DBSCAN</p><span style="font-size:0.8rem; color:var(--text-secondary);">Density-Based Clustering</span></div>', unsafe_allow_html=True)
        return

    df = st.session_state.df_preprocessed.copy()
    res = st.session_state.dbscan_res
    
    df['Cluster'] = res['labels']
    df['Cluster_Str'] = df['Cluster'].apply(lambda x: f"Cluster {int(x)}" if x != -1 else "Noise (-1)")
    
    total_obs = len(df)
    n_clusters = res['n_clusters']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h4>Total Observations</h4><p style="font-size: 1.8rem; font-weight:700; margin:0;">{total_obs:,}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h4>Formed Clusters</h4><p style="font-size: 1.8rem; font-weight:700; margin:0;">{n_clusters}</div>', unsafe_allow_html=True)
    with col3:
        sil_score = res['sil_score']
        sil_text = f"{sil_score:.3f}" if sil_score != -1 else "N/A"
        st.markdown(f'<div class="metric-card"><h4>Global Silhouette</h4><p style="font-size: 1.8rem; font-weight:700; margin:0;">{sil_text}</p></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
    
    df['Hour'] = df['periode_jam'].dt.hour
    day_map = {
        'Monday': 'Monday', 'Tuesday': 'Tuesday', 'Wednesday': 'Wednesday', 
        'Thursday': 'Thursday', 'Friday': 'Friday', 'Saturday': 'Saturday', 'Sunday': 'Sunday'
    }
    df['Day'] = df['periode_jam'].dt.day_name().map(day_map)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    is_dark = st.session_state.theme_mode == 'dark'
    bg_color = "#1E293B" if is_dark else "#FFFFFF"
    txt_color = "#F8FAFC" if is_dark else "#000000"
    grid_color = "#334155" if is_dark else "#E2E8F0"
    
    category_order = sorted(df['Cluster_Str'].unique())
    if 'Noise (-1)' in category_order:
        category_order.remove('Noise (-1)')
        category_order.append('Noise (-1)')
        
    color_map = {'Noise (-1)': '#94A3B8'}
    palette = px.colors.qualitative.Set1
    for i, cat in enumerate([c for c in category_order if c != 'Noise (-1)']):
        color_map[cat] = palette[i % len(palette)]

    df_valid = df[df['Cluster'] != -1].copy()

    c1, c2 = st.columns([1.2, 1.8])
    
    with c1:
        st.markdown("**Cluster Characteristic Profiling**")
        st.markdown("<span style='font-size:0.85rem; padding-top:0; color:var(--text-secondary);'>Average scaled feature values per valid cluster.</span>", unsafe_allow_html=True)

        df_valid = df[df['Cluster'] != -1].copy()
        
        cluster_means = df_valid.groupby('Cluster_Str')[['volume_kendaraan_berat', 'kecepatan_rata2', 'vc_ratio']].mean().reset_index()
        
        fig_radar = go.Figure()
        for idx, row in cluster_means.iterrows():
            cluster_name = row['Cluster_Str']
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['volume_kendaraan_berat'], row['kecepatan_rata2'], row['vc_ratio']],
                theta=['Volume', 'Speed', 'V/C Ratio'],
                fill='toself',
                name=cluster_name,
                line_color=color_map.get(cluster_name, '#000000'),
                opacity=0.7
            ))
            
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=grid_color, color=txt_color),
                angularaxis=dict(gridcolor=grid_color, color=txt_color),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=True,
            legend=dict(orientation="h", y=-0.2, x=0),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=txt_color),
            margin=dict(t=20, b=20, l=40, r=40),
            height=350
        )
        st.plotly_chart(fig_radar, use_container_width=True, theme=None, config={"displayModeBar": False})
        
    with c2:
        st.markdown("**Daily Traffic Cluster Distribution**")
        st.markdown("<span style='font-size:0.85rem; color:var(--text-secondary);'>Which clusters appear mostly on which days.</span>", unsafe_allow_html=True)
        
        daily_cluster = df.groupby(['Day', 'Cluster_Str']).size().reset_index(name='Count')
        fig_daily = px.bar(
            daily_cluster, x='Day', y='Count', color='Cluster_Str',
            barmode='stack',
            color_discrete_map=color_map,
            category_orders={'Cluster_Str': category_order, 'Day': day_order}
        )
        fig_daily.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=txt_color),
            margin=dict(t=20, b=30, l=50, r=20),
            height=350,
            xaxis=dict(title="", gridcolor=grid_color, color=txt_color),
            yaxis=dict(title="Observation Count", gridcolor=grid_color, color=txt_color),
            legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_daily, use_container_width=True, theme=None, config={"displayModeBar": False})
        
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("**Hourly Traffic Operational Patterns**")
    st.markdown("<span style='font-size:0.85rem; color:var(--text-secondary);'>Operational cluster distribution across the 24-hour cycle, segmented by day.</span>", unsafe_allow_html=True)
    
    tabs_day = st.tabs(day_order)
    for i, day in enumerate(day_order):
        with tabs_day[i]:
            df_day = df[df['Day'] == day]
            if df_day.empty:
                st.info(f"No data available for {day}.")
                continue
                
            hourly_cluster = df_day.groupby(['Hour', 'Cluster_Str']).size().reset_index(name='Count')
            fig_hourly = px.bar(
                hourly_cluster, x='Hour', y='Count', color='Cluster_Str',
                barmode='stack',
                color_discrete_map=color_map,
                category_orders={'Cluster_Str': category_order}
            )
            fig_hourly.update_layout(
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
                font=dict(color=txt_color),
                margin=dict(t=20, b=40, l=40, r=20),
                height=300,
                xaxis=dict(
                    title="Hour of Day (00:00 - 23:59)", 
                    tickmode='linear', tick0=0, dtick=1,
                    gridcolor=grid_color, color=txt_color
                ),
                yaxis=dict(title="Observation Count", gridcolor=grid_color, color=txt_color),
                legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_hourly, use_container_width=True, theme=None, config={"displayModeBar": False}, key=f"hourly_chart_{day}")

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("**Cluster Characteristic Summary & Interpretation**")
    st.markdown("<span style='font-size:0.85rem; color:var(--text-secondary);'>Detailed summary, average metric values, and traffic condition interpretation for each formed cluster.</span>", unsafe_allow_html=True)
    
    numeric_cols = ['volume_kendaraan_berat', 'kecepatan_rata2', 'vc_ratio']
    
    df_raw = st.session_state.df_raw.loc[df.index].copy()
    if 'truk_besar' in df_raw.columns:
        df_raw.rename(columns={'truk_besar': 'volume_kendaraan_berat'}, inplace=True)
        
    df_raw['Cluster'] = df['Cluster']

    for col in numeric_cols:
        df_raw[col] = df_raw[col].astype(str).str.replace(',', '.', regex=False)
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    summary_df_real = df_raw.groupby('Cluster')[numeric_cols].mean().reset_index()
    summary_df_real['Count'] = df_raw.groupby('Cluster').size().values
    
    cluster_ids = sorted([c for c in summary_df_real['Cluster'].tolist() if c != -1])
    tab_names = [f"Cluster {int(c)}" for c in cluster_ids]
    interp_tabs = st.tabs(tab_names)
    
    for i, c_id in enumerate(cluster_ids):
        row = summary_df_real[summary_df_real['Cluster'] == c_id].iloc[0]
        
        count = row['Count']
        vol = row['volume_kendaraan_berat']
        spd = row['kecepatan_rata2']
        vcr = row['vc_ratio']

        interpretation = interpret_traffic_condition(vcr, spd)
        
        c_name = f"Cluster {int(c_id)}"
        cluster_color = color_map.get(c_name, 'var(--primary-accent)')
        
        interp_color = "var(--text-primary)"
        if "Congested" in interpretation or "Heavy" in interpretation:
            interp_color = "#EF4444"
        elif "Free Flow" in interpretation or "Stable" in interpretation:
            interp_color = "#10B981"
        elif "Busy" in interpretation or "Slow" in interpretation or "Moderate" in interpretation or "Starting" in interpretation:
            interp_color = "#F59E0B"
        
        with interp_tabs[i]:
            st.markdown(f"""
            <div style="background-color:var(--bg-card); border:1px solid var(--border-color); border-top:4px solid {cluster_color}; border-radius:8px; padding:20px; box-shadow:var(--shadow-sm); margin-top:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                    <h4 style="margin:0; font-size:1.4rem; color:{cluster_color};">{c_name}</h4>
                    <span style="background-color:var(--hover-bg); padding:4px 12px; border-radius:12px; font-size:1rem; color:var(--text-secondary); font-weight:600;">{int(count):,} data points</span>
                </div>
                <div style="margin-bottom:25px; padding-bottom:15px; border-bottom:1px solid var(--border-color);">
                    <div style="font-size:1.3rem; font-weight:700; color:{interp_color}; display:flex; align-items:center; gap:8px;">
                        {interpretation}
                    </div>
                </div>
                <div style="display:flex; gap:20px; flex-wrap:wrap;">
                    <div style="flex:1; min-width:150px; background-color:var(--hover-bg); padding:15px; border-radius:8px; text-align:center;">
                        <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:5px;">Avg Volume (Vehicles)</div>
                        <div style="font-size:1.8rem; font-weight:700; color:var(--text-primary);">{vol:,.1f}</div>
                    </div>
                    <div style="flex:1; min-width:150px; background-color:var(--hover-bg); padding:15px; border-radius:8px; text-align:center;">
                        <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:5px;">Avg Speed (km/h)</div>
                        <div style="font-size:1.8rem; font-weight:700; color:var(--text-primary);">{spd:,.2f}</div>
                    </div>
                    <div style="flex:1; min-width:150px; background-color:var(--hover-bg); padding:15px; border-radius:8px; text-align:center;">
                        <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:5px;">Avg V/C Ratio</div>
                        <div style="font-size:1.8rem; font-weight:700; color:var(--text-primary);">{vcr:,.3f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def sync_eps_from_slider():
    st.session_state.eps_input = st.session_state.eps_slider

def sync_eps_from_input():
    st.session_state.eps_slider = st.session_state.eps_input

def sync_minpts_from_slider():
    st.session_state.minpts_input = st.session_state.minpts_slider

def sync_minpts_from_input():
    st.session_state.minpts_slider = st.session_state.minpts_input

def show_model_evaluation():
    if 'pending_auto_params' in st.session_state:
        st.session_state.eps_slider = st.session_state.pending_auto_params['eps']
        st.session_state.eps_input = st.session_state.pending_auto_params['eps']
        st.session_state.minpts_slider = st.session_state.pending_auto_params['minpts']
        st.session_state.minpts_input = st.session_state.pending_auto_params['minpts']
        del st.session_state.pending_auto_params

    if 'eps_slider' not in st.session_state:
        st.session_state.eps_slider = 0.040
    if 'eps_input' not in st.session_state:
        st.session_state.eps_input = 0.040
    if 'minpts_slider' not in st.session_state:
        st.session_state.minpts_slider = 10
    if 'minpts_input' not in st.session_state:
        st.session_state.minpts_input = 10

    st.markdown("### DBSCAN Model Evaluation")
    st.markdown("Execution of density-based clustering algorithm and parameter optimization.")
    
    if st.session_state.df_preprocessed is None:
        st.warning("Please upload and save a dataset first in the Upload Dataset menu.")
        return
        
    df = st.session_state.df_preprocessed.copy()
    kolom_numerik = ['volume_kendaraan_berat', 'kecepatan_rata2', 'vc_ratio']
    X = df[kolom_numerik].values
    
    col_ctrl, col_metrics = st.columns([1.3, 2.7])
    
    # --- PARAMETER CONTROL PANEL ---
    with col_ctrl:
        
        tab_auto, tab_manual_slider, tab_manual_input = st.tabs(["Auto Configuration", "Manual Refinement", "Manual Refinement (Input)"])
        
        with tab_auto:
            st.markdown("<span style='font-size:0.85rem; font-weight:600; color:var(--text-secondary); display:block; margin-bottom:4px;'>OPTIMIZATION TARGET</span>", unsafe_allow_html=True)
            
            auto_mode = st.selectbox(
                "Optimization Goal", 
                ["Absolute Highest Score", "Absolute Lowest Noise Ratio", "Optimized (Noise < 40%, SC > 0.500)", "K-Distance Method"], 
                label_visibility="collapsed"
            )
            
            if auto_mode == "Absolute Highest Score":
                st.info("Searches for the highest possible Silhouette score.")
            elif auto_mode == "Absolute Lowest Noise Ratio":
                st.info("Searches for the lowest possible noise ratio.")
            elif auto_mode == "K-Distance Method":
                st.info("Uses K-Distance method to find optimal parameters.")
            else:
                st.info("Searches for a balanced cluster structure prioritizing acceptable noise levels (≤ 40%) and score (> 0.500).")
                
            run_auto_ph = st.empty()
            run_auto = run_auto_ph.button("Run Auto Configuration", type="primary", use_container_width=True)

        with tab_manual_slider:
            eps_val_slider = st.slider("Epsilon (Eps)", 0.010, 0.300, key="eps_slider", step=0.001, format="%.3f", on_change=sync_eps_from_slider)
            min_pts_val_slider = st.slider("MinPts", 1, 300, key="minpts_slider", step=1, on_change=sync_minpts_from_slider)
            run_manual_slider = st.button("Run Manual Refinement", type="primary", use_container_width=True, key="btn_manual_slider")
            
        with tab_manual_input:
            eps_val_input = st.number_input("Epsilon (Eps)", min_value=0.010, max_value=0.300, key="eps_input", step=0.001, format="%.3f", on_change=sync_eps_from_input)
            min_pts_val_input = st.number_input("MinPts", min_value=1, max_value=300, key="minpts_input", step=1, on_change=sync_minpts_from_input)
            run_manual_input = st.button("Run Manual Refinement", type="primary", use_container_width=True, key="btn_manual_input")

        run_manual = run_manual_slider or run_manual_input
        eps_val = st.session_state.eps_slider
        min_pts_val = st.session_state.minpts_slider
                
    # --- MANUAL EXECUTION LOGIC ---
    if run_manual:
        db = DBSCAN(eps=eps_val, min_samples=min_pts_val)
        labels = db.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = list(labels).count(-1) / len(labels)
        
        sil = -1
        if n_clusters > 1:
            try:
                core_mask = labels != -1
                labels_core = labels[core_mask]
                X_core = X[core_mask]
                if len(set(labels_core)) > 1:
                    s_size = 1000 if len(X_core) > 1000 else None
                    sil = silhouette_score(X_core, labels_core, sample_size=s_size, random_state=42)
            except:
                pass
            
        st.session_state.dbscan_res = {
            'eps': eps_val, 'minpts': min_pts_val, 'labels': labels,
            'sil_score': sil, 'noise': noise_ratio, 
            'n_clusters': n_clusters
        }

    # --- RENDER EVALUATION METRICS (Top Grid) ---
    if st.session_state.dbscan_res is not None:
        res = st.session_state.dbscan_res
        df['Cluster'] = res['labels']
        
        sil_color = "#10B981" if res['sil_score'] >= 0.450 else "#F59E0B"
        noise_color = "#10B981" if res['noise'] <= 0.40 else "#EF4444"
        
        sil_text = f"{res['sil_score']:.3f}" if res['sil_score'] != -1 else "N/A"
        
        with col_metrics:
            # Row 1 of 2
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="metric-card"><h5>Silhouette</h5><p style="font-size: 1.4rem; font-weight:700; color:{sil_color}; margin:0; padding-top:4px;">{sil_text}</p></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><h5>Noise Ratio</h5><p style="font-size: 1.4rem; font-weight:700; color:{noise_color}; margin:0; padding-top:4px;">{res["noise"]*100:.1f}%</p></div>', unsafe_allow_html=True)
            
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            
            # Row 2 of 2
            m3, m4 = st.columns(2)
            with m3:
                st.markdown(f'<div class="metric-card" style="margin-bottom:10px;"><h5>Clusters</h5><p style="font-size: 1.4rem; font-weight:700; color:var(--primary-accent); margin:0; padding-top:4px;">{res["n_clusters"]}</p></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card" style="margin-bottom:10px;"><h5>Parameters</h5><p style="font-size: 1.4rem; font-weight:700; color:var(--text-secondary); margin:0; padding-top:4px;">{res["eps"]:.3f} / {res["minpts"]}</p></div>', unsafe_allow_html=True)

    loading_ph = st.empty()
    vis_ph = st.empty()

    if 'auto_notif' in st.session_state:
        if st.session_state.auto_notif_type == 'success':
            st.success(st.session_state.auto_notif)
        else:
            st.error(st.session_state.auto_notif)
        del st.session_state.auto_notif
        del st.session_state.auto_notif_type

    loading_ph = st.empty()
    vis_ph = st.empty()

    # --- AUTO CONFIG EXECUTION LOGIC ---
    if run_auto:
        st.markdown("""
        <style>
        div[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"], 
        section[data-testid="stMain"] .stButton > button[kind="primary"] {
            background-color: #EF4444 !important;
            border-color: #EF4444 !important;
        }
        div[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"]:hover, 
        section[data-testid="stMain"] .stButton > button[kind="primary"]:hover {
            background-color: #DC2626 !important;
            border-color: #DC2626 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # KEMBALIKAN TOMBOL CANCEL SESUAI REQUEST DOSEN PEMBIMBING
        run_auto_ph.button("Cancel", type="primary", use_container_width=True)
        
        best_result = None
        
        with loading_ph.container():
            is_dark = st.session_state.theme_mode == 'dark'
            spinner_stroke = "#60A5FA" if is_dark else "var(--primary-accent)"
            spinner_text = "#F8FAFC" if is_dark else "var(--text-primary)"
            
            st.markdown(f"""
            <style>
            [data-testid="stSpinner"] p {{
                color: {spinner_text} !important;
                font-weight: 600 !important;
                font-size: 0.95rem !important;
                text-shadow: {'0 0 4px rgba(255,255,255,0.2)' if is_dark else 'none'} !important;
            }}
            [data-testid="stSpinner"] svg circle {{
                stroke: {spinner_stroke} !important;
                stroke-width: 4.5px !important;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            with st.spinner("Executing Grid Search or K-Distance Method..."):
                
                eps_space = np.arange(0.01, 0.72, 0.02).tolist() 
                minpts_space = range(10, 155, 5) 
                grid_results = []
                
                if auto_mode == "K-Distance Method":
                    progress_bar = st.progress(0)
                    total_steps = len(minpts_space)
                    
                    for idx, m in enumerate(minpts_space):
                        progress_bar.progress((idx + 1) / total_steps)
                        
                        nn = NearestNeighbors(n_neighbors=m).fit(X)
                        distances, _ = nn.kneighbors(X)
                        k_dist = np.sort(distances[:, -1])[::-1]
                        
                        n_points = len(k_dist)
                        all_coords = np.vstack((range(n_points), k_dist)).T
                        first_point = all_coords[0]
                        line_vec = all_coords[-1] - all_coords[0]
                        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
                        vec_from_first = all_coords - first_point
                        scalar_prod = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
                        vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
                        vec_to_line = vec_from_first - vec_from_first_parallel
                        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
                        
                        elbow_idx = int(np.argmax(dist_to_line))
                        optimal_eps = round(float(k_dist[elbow_idx]), 3)
                        
                        if optimal_eps < 0.01:
                            optimal_eps = 0.01
                            
                        db = DBSCAN(eps=optimal_eps, min_samples=m).fit(X)
                        labels = db.labels_
                        
                        n_clusters = labels.max() + 1
                        if n_clusters < 2 or n_clusters > 10:
                            continue
                         
                        noise_ratio = np.sum(labels == -1) / len(X)
                        
                        try:
                            core_mask = labels != -1
                            labels_core = labels[core_mask]
                            X_core = X[core_mask]
                            
                            s_size = 1500 if len(X_core) > 1500 else None
                            sil = silhouette_score(X_core, labels_core, sample_size=s_size, random_state=42)
                            
                            grid_results.append({
                                'eps': optimal_eps,
                                'minpts': m,
                                'sil_score': sil,
                                'noise': noise_ratio
                            })
                        except:
                            continue
                else:
                    progress_bar = st.progress(0)
                    total_combinations = len(eps_space) * len(minpts_space)
                    current_step = 0
                    
                    for e in eps_space:
                        for m in minpts_space:
                            current_step += 1
                            if current_step % 10 == 0: 
                                progress_bar.progress(current_step / total_combinations)
                                
                            db = DBSCAN(eps=e, min_samples=m).fit(X)
                            labels = db.labels_
                            
                            n_clusters = labels.max() + 1
                            if n_clusters < 2 or n_clusters > 10:
                                continue
                                
                            noise_ratio = np.sum(labels == -1) / len(X)
                            
                            if auto_mode == "Optimized (Noise < 40%, SC > 0.500)" and noise_ratio > 0.40:
                                continue
                                
                            try:
                                core_mask = labels != -1
                                labels_core = labels[core_mask]
                                X_core = X[core_mask]
                                
                                s_size = 1500 if len(X_core) > 1500 else None
                                sil = silhouette_score(X_core, labels_core, sample_size=s_size, random_state=42)
                                
                                grid_results.append({
                                    'eps': e,
                                    'minpts': m,
                                    'sil_score': sil,
                                    'noise': noise_ratio
                                })
                            except:
                                continue
                
                if len(grid_results) > 0:
                    if auto_mode in ["Absolute Highest Score", "K-Distance Method"]:
                        best_result = max(grid_results, key=lambda x: x['sil_score'])
                        
                    elif auto_mode == "Absolute Lowest Noise Ratio":
                        best_result = min(grid_results, key=lambda x: (x['noise'], -x['sil_score']))
                        
                    elif auto_mode == "Optimized (Noise < 40%, SC > 0.500)":
                        valid_results = [r for r in grid_results if r['sil_score'] > 0.500]
                        if len(valid_results) > 0:
                            best_result = max(valid_results, key=lambda x: x['sil_score'] - x['noise'])
        
        loading_ph.empty() 
        
        if best_result is not None:
            db_final = DBSCAN(eps=best_result['eps'], min_samples=best_result['minpts']).fit(X)
            final_labels = db_final.labels_
            
            st.session_state.pending_auto_params = {
                'eps': best_result['eps'],
                'minpts': best_result['minpts']
            }
            
            st.session_state.dbscan_res = {
                'eps': best_result['eps'], 
                'minpts': best_result['minpts'], 
                'labels': final_labels,
                'sil_score': best_result['sil_score'], 
                'noise': best_result['noise'], 
                'n_clusters': int(final_labels.max() + 1)
            }
            st.session_state.auto_notif = f"Parameters found for {auto_mode}!"
            st.session_state.auto_notif_type = "success"
            st.rerun() 
        else:
            if auto_mode == "K-Distance Method":
                st.session_state.auto_notif = "No valid cluster structure found. Try manual refinement or other auto configuration modes."
            else:
                st.session_state.auto_notif = "No valid cluster structure found under current constraints. Try manual refinement."
                
            st.session_state.auto_notif_type = "error"
            st.rerun()

    # --- RENDER CLUSTER VISUALIZATIONS (Bottom Grid) ---
    if st.session_state.dbscan_res is not None:
        with vis_ph.container():
            st.markdown("<hr style='border:1px dashed var(--border-color); margin: 20px 0;'>", unsafe_allow_html=True)
            
            v1, v2 = st.columns([2, 1])
            
            df['Cluster_Str'] = df['Cluster'].astype(str)
            df.loc[df['Cluster'] == -1, 'Cluster_Str'] = 'Noise (-1)'
            
            category_order = sorted(df['Cluster_Str'].unique())
            if 'Noise (-1)' in category_order:
                category_order.remove('Noise (-1)')
                category_order.append('Noise (-1)')
                
            is_dark = st.session_state.theme_mode == 'dark'
            bg_color = "#1E293B" if is_dark else "#F8FAFC"
            txt_color = "#F8FAFC" if is_dark else "#000000"
            grid_color = "#475569" if is_dark else "#CBD5E1"
            
            color_map = {'Noise (-1)': '#94A3B8'}
            palette = px.colors.qualitative.Set1
            for i, cat in enumerate([c for c in category_order if c != 'Noise (-1)']):
                color_map[cat] = palette[i % len(palette)]
            
            with v1:
                st.markdown("**3D Cluster Representation**")
                
                fig_3d = px.scatter_3d(
                    df, x='volume_kendaraan_berat', y='kecepatan_rata2', z='vc_ratio',
                    color='Cluster_Str', opacity=0.8,
                    color_discrete_map=color_map,
                    category_orders={'Cluster_Str': category_order}
                )
                
                fig_3d.update_traces(marker=dict(size=4, line=dict(width=0)))
                fig_3d.update_layout(
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=txt_color),
                    margin=dict(l=0, r=0, b=0, t=0),
                    legend=dict(title=dict(text="Cluster Label"), orientation="v", yanchor="top", y=1, xanchor="left", x=0.01),
                    scene=dict(
                        xaxis=dict(title='Volume', backgroundcolor=bg_color, gridcolor=grid_color, color=txt_color),
                        yaxis=dict(title='Speed', backgroundcolor=bg_color, gridcolor=grid_color, color=txt_color),
                        zaxis=dict(title='V/C Ratio', backgroundcolor=bg_color, gridcolor=grid_color, color=txt_color),
                    )
                )
                st.plotly_chart(fig_3d, use_container_width=True, theme=None, config={"displayModeBar": False})
                
            with v2:
                st.markdown("**Cluster Distribution**")
                
                cluster_counts = df.groupby('Cluster_Str').size().reset_index(name='Count')
                cluster_counts.rename(columns={'Cluster_Str': 'Cluster'}, inplace=True)
                
                fig_bar = px.bar(
                    cluster_counts, x='Cluster', y='Count', color='Cluster',
                    text='Count', color_discrete_map=color_map,
                    category_orders={'Cluster': category_order}
                )
                fig_bar.update_layout(
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=txt_color),
                    showlegend=False,
                    margin=dict(l=60, r=40, b=35, t=40),
                    xaxis_title="Cluster",
                    yaxis_title="Observations",
                    xaxis=dict(tickfont=dict(color=txt_color)),
                    yaxis=dict(tickfont=dict(color=txt_color))
                )
                fig_bar.update_traces(textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True, theme=None, config={"displayModeBar": False})
                
            st.markdown("**Cluster Characteristic Summary**")
            summary_df = df.groupby('Cluster')[kolom_numerik].mean().reset_index()
            summary_df['Count'] = df.groupby('Cluster').size().values
            
            for col in kolom_numerik:
                summary_df[col] = summary_df[col].map(lambda x: f"{x:.3f}")
                
            st.markdown(render_themed_table(summary_df, index_name="#"), unsafe_allow_html=True)

def show_preprocessing():
    st.markdown("### Data Preprocessing")
    st.markdown("Overview of feature extraction, anomaly exclusion, missing value imputation, and feature scaling.")
    
    if 'df_preprocessed' not in st.session_state or 'prep_metrics' not in st.session_state:
        st.warning("Please upload and save a dataset first in the Upload Dataset menu.")
        return

    st.markdown("""
            <style>
            [data-stale="true"] {
                opacity: 0 !important;
                display: none !important;
                transition: none !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
    df_preprocessed = st.session_state.df_preprocessed
    metrics = st.session_state.prep_metrics
    
    excluded_rows = metrics["excluded_rows"]
    akhir_baris = metrics["akhir_baris"]
    total_missing_dict = metrics["total_missing_dict"]
    total_missing_sum = metrics["total_missing_sum"]
    kolom_numerik = ['volume_kendaraan_berat', 'kecepatan_rata2', 'vc_ratio']

    # ---------------------------------------------------------
    # TOP ROW: 2 EQUAL-HEIGHT & SYMMETRICAL COLUMNS
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="prep-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="prep-title">Variable Selection & Anomaly Filtering</div>
                <div class="prep-desc">Isolates core operational variables and excludes holiday anomaly periods to retain standard traffic patterns.</div>
                <div style="margin-bottom: 10px;">
                    <span style="font-size:0.72rem; color:var(--text-secondary); font-weight:600; display:block; margin-bottom:4px;">SELECTED FEATURES:</span>
                    <span class="var-badge">volume_kendaraan_berat</span>
                    <span class="var-badge">kecepatan_rata2</span>
                    <span class="var-badge">vc_ratio</span>
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="font-size:0.72rem; color:var(--text-secondary); font-weight:600; display:block; margin-bottom:4px;">EXCLUDED ANOMALIES:</span>
                    <span class="var-badge" style="border-color: #EF4444; color: #EF4444; background-color: rgba(239, 68, 68, 0.08);">New Year Period</span>
                    <span class="var-badge" style="border-color: #EF4444; color: #EF4444; background-color: rgba(239, 68, 68, 0.08);">Eid Mubarak Period (H-7 to H+7)</span>
                </div>
            </div>
            <div class="log-box">
                Filtered Out: <b style="color:#EF4444;">{excluded_rows}</b> anomaly rows | Remaining Active Dataset: <b>{akhir_baris}</b> rows
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        missing_badges = []
        for col_name, cnt in total_missing_dict.items():
            if cnt > 0:
                badge = f"<span class='var-badge' style='border-color: #F59E0B; color: #D97706; background-color: rgba(245, 158, 11, 0.08);'><b>{col_name}</b>: {cnt} null/zero</span>"
            else:
                badge = f"<span class='var-badge' style='border-color: #10B981; color: #10B981; background-color: rgba(16, 185, 129, 0.08);'><b>{col_name}</b>: 0 missing</span>"
            missing_badges.append(badge)
            
        missing_badges_html = " ".join(missing_badges)

        if total_missing_sum > 0:
            imputation_status_html = '<b style="color:#10B981;">Complete</b>'
        else:
            imputation_status_html = '<b style="color:gray;">-</b>'

        st.markdown(f"""
        <div class="prep-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="prep-title">Missing Value Imputation</div>
                <div class="prep-desc">Identifies zero/null values across all variables and replaces them with medians by the hour imputation.</div>
                <div style="margin-bottom: 10px;">
                    <span style="font-size:0.72rem; color:var(--text-secondary); font-weight:600; display:block; margin-bottom:4px;">DETECTED MISSING / ZERO VALUES:</span>
                    {missing_badges_html}
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="font-size:0.72rem; color:var(--text-secondary); font-weight:600; display:block; margin-bottom:4px;">IMPUTATION TECHNIQUE:</span>
                    <span class="var-badge" style="border-color:var(--primary-accent); color:var(--primary-accent); background-color:var(--dropzone-bg);">Median by the Hour</span>
                </div>
            </div>
            <div class="log-box">
                Total Imputed Cells: <b style="color:#D97706;">{total_missing_sum}</b> values | Imputation Status: {imputation_status_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # BOTTOM ROW: FULL WIDTH CARD (MIN-MAX NORMALIZATION)
    # ---------------------------------------------------------
    st.markdown("""
    <div class="prep-card" style="margin-top: 12px;">
        <div class="prep-title">Min-Max Scaling Normalization</div>
        <div class="prep-desc">Standardizes numerical features to prevent distance-calculation bias during DBSCAN density clustering computation.</div>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center; margin-bottom: 8px;">
            <div>
                <span style="font-size:0.72rem; color:var(--text-secondary); font-weight:600; display:block; margin-bottom:4px;">TRANSFORMED FEATURES:</span>
                <span class="var-badge">volume_kendaraan_berat</span>
                <span class="var-badge">kecepatan_rata2</span>
                <span class="var-badge">vc_ratio</span>
            </div>
            <div>
                <span style="font-size:0.72rem; color:var(--text-secondary); font-weight:600; display:block; margin-bottom:4px;">SCALE RANGE & PRECISION:</span>
                <span class="var-badge" style="background-color: var(--dropzone-bg);">[0.000, 1.000]</span>
                <span class="var-badge" style="background-color: var(--dropzone-bg);">3 Decimal Floating Point</span>
            </div>
        </div>
        <div class="log-box">
            Scaler Engine: <b>sklearn.preprocessing.MinMaxScaler</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # VISUALIZATION (POST PREPROCESSING)
    # ---------------------------------------------------------
    st.markdown("<br><b>Traffic Data Trend Chart (Post-Preprocessing)</b>", unsafe_allow_html=True)
    
    # Ambil tema dari session state (opsional jika Anda punya fitur toggle dark mode)
    is_dark = st.session_state.get('theme_mode', 'light') == 'dark'
    chart_bg = "#1E293B" if is_dark else "#FFFFFF"
    chart_grid = "#334155" if is_dark else "#E2E8F0"
    chart_text = "#F8FAFC" if is_dark else "#000000"

    fig = go.Figure()
    # Menggunakan go.Scattergl untuk mencegah UI stuck/hang pada dataset puluhan ribu baris
    fig.add_trace(go.Scattergl(
        x=df_preprocessed['periode_jam'], y=df_preprocessed['volume_kendaraan_berat'],
        mode='lines', name='volume_kendaraan_berat', line=dict(color='#2563EB', width=1.5)
    ))
    fig.add_trace(go.Scattergl(
        x=df_preprocessed['periode_jam'], y=df_preprocessed['kecepatan_rata2'],
        mode='lines', name='kecepatan_rata2', line=dict(color='#F59E0B', width=1.5)
    ))
    fig.add_trace(go.Scattergl(
        x=df_preprocessed['periode_jam'], y=df_preprocessed['vc_ratio'],
        mode='lines', name='vc_ratio', line=dict(color='#10B981', width=1.5)
    ))
    
    fig.update_layout(
        plot_bgcolor=chart_bg,
        paper_bgcolor=chart_bg,
        font=dict(color=chart_text),
        xaxis=dict(
            gridcolor=chart_grid, 
            zerolinecolor=chart_grid, 
            color=chart_text,
            tickfont=dict(color=chart_text),
        ),
        yaxis=dict(
            gridcolor=chart_grid, 
            zerolinecolor=chart_grid, 
            color=chart_text,
            tickfont=dict(color=chart_text)
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="left", 
            x=0, 
            font=dict(color=chart_text)
        ),
        margin=dict(l=10, r=10, t=10, b=30), 
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    
    col_tampil1, col_tampil2 = st.columns([1.2, 1])
    with col_tampil1:
        st.markdown("**Data Preview (First 8 Rows)**")
        sample_df = df_preprocessed.head(8).copy()
        for col in kolom_numerik:
            sample_df[col] = sample_df[col].map(lambda x: f"{x:.3f}")
        sample_df['periode_jam'] = sample_df['periode_jam'].astype(str)
        st.markdown(render_themed_table(sample_df), unsafe_allow_html=True)
        
    with col_tampil2:
        st.markdown("**Descriptive Statistics**")
        desc_df = df_preprocessed[kolom_numerik].describe().copy()
        desc_df_fmt = desc_df.map(lambda x: f"{x:.3f}")
        st.markdown(render_themed_table(desc_df_fmt, index_name=""), unsafe_allow_html=True)

def run_background_preprocessing(df_raw_input):
    df = df_raw_input.copy()
    kolom_utama = ['periode_jam', 'truk_besar', 'kecepatan_rata2', 'vc_ratio']
    if not all(col in df.columns for col in kolom_utama):
        return None, None, f"Please ensure these columns exist in your dataset: {kolom_utama}"
        
    df_seleksi = df[kolom_utama].copy()
    df_seleksi.rename(columns={'truk_besar': 'volume_kendaraan_berat'}, inplace=True)
    df_seleksi['periode_jam'] = pd.to_datetime(df_seleksi['periode_jam'], dayfirst=True)

    # 1. Anomaly Filtering
    tahun_unik = df_seleksi['periode_jam'].dt.year.unique()
    tanggal_anomali = []

    lebaran_dict = {
        2024: '2024-04-10',
        2025: '2025-03-30',
        2026: '2026-03-19',
        2027: '2027-03-09',
        2028: '2028-02-26',
        2029: '2029-02-14',
        2030: '2030-02-04'
    }

    for tahun in tahun_unik:
        # A. New Years Eve 
        for hari in range(1, 6):
            tanggal_anomali.append(f"{tahun}-01-0{hari}")
            
        # B. Eid (H-7 s.d H+7)
        if tahun in lebaran_dict:
            lebaran_date = pd.to_datetime(lebaran_dict[tahun])
            
            awal_mudik = lebaran_date - pd.Timedelta(days=7)
            akhir_balik = lebaran_date + pd.Timedelta(days=8) 
            
            periode_lebaran = pd.date_range(start=awal_mudik, end=akhir_balik)
            tanggal_anomali.extend(periode_lebaran.strftime('%Y-%m-%d').tolist())

    awal_baris = len(df_seleksi)
    df_seleksi['tanggal_saja'] = df_seleksi['periode_jam'].dt.strftime('%Y-%m-%d')
    df_final = df_seleksi[~df_seleksi['tanggal_saja'].isin(tanggal_anomali)].copy()
    df_final.drop(columns=['tanggal_saja'], inplace=True)
    
    akhir_baris = len(df_final)
    excluded_rows = awal_baris - akhir_baris

    # 2. Missing Value Imputation (Hourly Median)
    df_preprocessed = df_final.copy()
    df_preprocessed['jam_ke'] = df_preprocessed['periode_jam'].dt.hour
    kolom_numerik = ['volume_kendaraan_berat', 'kecepatan_rata2', 'vc_ratio']

    for col in kolom_numerik:
        df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.', regex=False)
        df_preprocessed[col] = pd.to_numeric(df_preprocessed[col], errors='coerce')
            
    total_missing_dict = {}
    for col in kolom_numerik:
        df_preprocessed[col] = df_preprocessed[col].replace(0, float('nan'))
        missing_count = df_preprocessed[col].isnull().sum()
        total_missing_dict[col] = int(missing_count)
        
        if missing_count > 0:
            median_spesifik_jam = df_preprocessed.groupby('jam_ke')[col].transform('median')
            df_preprocessed[col] = df_preprocessed[col].fillna(median_spesifik_jam)
            
    df_preprocessed.drop(columns=['jam_ke'], inplace=True)
    total_missing_sum = sum(total_missing_dict.values())

    # 3. Min-Max Scaling Normalization
    scaler = MinMaxScaler()
    df_preprocessed[kolom_numerik] = scaler.fit_transform(df_preprocessed[kolom_numerik])
    
    prep_metrics = {
        "excluded_rows": excluded_rows,
        "akhir_baris": akhir_baris,
        "total_missing_dict": total_missing_dict,
        "total_missing_sum": total_missing_sum
    }
    
    return df_preprocessed, prep_metrics, None

def show_upload_dataset():
    st.markdown("### Dataset Management")
    st.markdown("Upload raw dataset (.csv/.xlsx) for processing and model training.")

    st.markdown("<div style='padding-top: 15px;'></div>", unsafe_allow_html=True)
    st.info("Please upload the supported dataset format and requirements to run the clustering model")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"], key="dataset_file_uploader")
    cache_key = "parsed_upload_cache"


    with st.expander("Supported Dataset Format & Requirements", expanded=False):
        st.markdown("""
        To ensure smooth data parsing and preprocessing, please make sure your dataset adheres to the following structure:
        
        **1. File Format:**
        - Accepted file types: **.csv** or **.xlsx**
        - Data header offset: The parser **automatically detects and skips** metadata rows to find the data headers.
        
        **2. Mandatory Columns:**
        - `periode_jam` *(Datetime)* : Observation timestamp (e.g., `DD/MM/YYYY HH:MM`).
        - `truk_besar` *(Numeric)* : Heavy vehicle traffic volume count.
        - `kecepatan_rata2` *(Numeric)* : Average traffic speed (km/h).
        - `vc_ratio` *(Numeric)* : Volume-to-Capacity ratio.
        
        *Note: If your file uses zero (0) values for unrecorded sensor readings, the system will automatically impute them using hourly group medians during preprocessing.*
        """)

    # 1. Update cache ONLY if a new file is uploaded
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        need_parse = (
            cache_key not in st.session_state
            or st.session_state[cache_key].get("file_id") != file_id
        )

        if need_parse:
            try:
                # Cari baris header secara dinamis
                skip_idx = find_header_row(uploaded_file)
                
                # Tarik data berdasarkan skiprows yang ditemukan
                uploaded_file.seek(0)
                if uploaded_file.name.endswith('.csv'):
                    df_parsed = pd.read_csv(uploaded_file, skiprows=skip_idx)
                else:
                    df_parsed = pd.read_excel(uploaded_file, skiprows=skip_idx)
                    
                st.session_state[cache_key] = {"file_id": file_id, "df": df_parsed, "error": None}
            except Exception as e:
                st.session_state[cache_key] = {"file_id": file_id, "df": None, "error": str(e)}

    # 2. Render UI directly from CACHE
    if cache_key in st.session_state:
        cache = st.session_state[cache_key]

        if cache["error"] is not None:
            st.error(f"Failed to read the file: {cache['error']}")
            return

        if cache["df"] is not None:
            df = cache["df"].copy()

            # Date range filtering check
            if 'periode_jam' in df.columns:
                df['periode_jam'] = pd.to_datetime(df['periode_jam'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['periode_jam'])

                st.markdown("<hr style='border:1px dashed var(--border-color);'>", unsafe_allow_html=True)
                st.markdown("#### Filter Observation Period")
                st.markdown("<p>Select start and end dates to filter the dataset before entering Preprocessing.</p>", unsafe_allow_html=True)

                min_date = df['periode_jam'].min().date()
                max_date = df['periode_jam'].max().date()

                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key="filter_start_date")
                with col2:
                    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key="filter_end_date")

                mask = (df['periode_jam'].dt.date >= start_date) & (df['periode_jam'].dt.date <= end_date)
                df_filtered = df.loc[mask].copy()
                if st.button("Save & Proceed", type="primary", key="save_proceed_btn"):
                    st.session_state.df_raw = df_filtered 
                    
                    with st.spinner("Processing data in background..."):
                        df_prep, metrics, err = run_background_preprocessing(st.session_state.df_raw)
                        
                    if err:
                        st.error(err)
                    else:
                        st.session_state.df_preprocessed = df_prep
                        st.session_state.prep_metrics = metrics
                        st.session_state.dbscan_res = None 
                        st.success(f"Data period {start_date} to {end_date} saved and preprocessed successfully!")               

                st.markdown("<hr style='border:1px dashed var(--border-color);'>", unsafe_allow_html=True)
                st.markdown("#### Dataset Preview")
                st.markdown(f"<p>Displaying the first 10 rows of the selected period. Total data found: <b>{len(df_filtered):,} rows</b>.</p>", unsafe_allow_html=True)

                sample_df = df_filtered.head(10).copy()

                for col in sample_df.columns:
                    if col != 'periode_jam':
                        try:
                                
                            col_num = pd.to_numeric(sample_df[col])
                            
                            def format_num(x):
                                if pd.isna(x):
                                    return "NaN"
                                return f"{float(x):.3f}".rstrip('0').rstrip('.') if float(x) != 0 else "0"
                                    
                            sample_df[col] = col_num.apply(format_num)
                        except ValueError:
                            sample_df[col] = sample_df[col].astype(str)
                
                st.markdown(render_themed_table(sample_df), unsafe_allow_html=True)

            else:
                st.error("Error: Column 'periode_jam' was not found in the dataset. Please check the raw data format or the dataset preview above.")

def find_header_row(uploaded_file):
    
    target_cols = ['periode_jam', 'truk_besar', 'kecepatan_rata2', 'vc_ratio']
    uploaded_file.seek(0)
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df_temp = pd.read_csv(uploaded_file, nrows=30, header=None, dtype=str)
        else:
            df_temp = pd.read_excel(uploaded_file, nrows=30, header=None, dtype=str)
        
        for idx, row in df_temp.iterrows():
            row_values = [str(val).strip().lower() for val in row.values if pd.notna(val)]
            if any(col in row_values for col in target_cols):
                return idx
    except Exception:
        pass 
        
    return 0

def main():
    apply_custom_css()

    # Define relative logo path
    logo_path = os.path.join("Gambar", "perhubungan.png")
    logo_b64 = get_base64_image(logo_path)

    # --- Sidebar Section ---
    with st.sidebar:
        # Header / Ministry Branding
        if logo_b64:
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Kemenhub Logo">'
        else:
            logo_html = '<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Logo_Kementerian_Perhubungan_Republik_Indonesia.png/600px-Logo_Kementerian_Perhubungan_Republik_Indonesia.png" alt="Kemenhub Logo">'

        st.markdown(f"""
            <div class="sidebar-header-container">
                {logo_html}
                <h3 style="margin:0; font-size:1.05rem; letter-spacing:1px; color:var(--text-primary);">MINISTRY OF TRANSPORTATION</h3>
                <p style="margin:2px 0 0 0; font-size:0.72rem; color:var(--text-secondary);">REPUBLIC OF INDONESIA</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation Section Title
        st.markdown('<div class="nav-section-title">NAVIGATION</div>', unsafe_allow_html=True)
        
        # --- Navigation Buttons ---
        nav_container = st.container()
        with nav_container:
            if st.button("Overview", type="primary" if st.session_state.current_page == "Overview" else "secondary", use_container_width=True):
                set_page("Overview")
                st.rerun()
                
            if st.button("Model Evaluation", type="primary" if st.session_state.current_page == "Model Evaluation" else "secondary", use_container_width=True):
                set_page("Model Evaluation")
                st.rerun()
                
            if st.button("Preprocessing", type="primary" if st.session_state.current_page == "Preprocessing" else "secondary", use_container_width=True):
                set_page("Preprocessing")
                st.rerun()
                
            if st.button("Upload Dataset", type="primary" if st.session_state.current_page == "Upload Dataset" else "secondary", use_container_width=True):
                set_page("Upload Dataset")
                st.rerun()

        # --- Sidebar Footer Section ---
        st.markdown('<div class="sidebar-footer-container"></div>', unsafe_allow_html=True)
        
        theme_label = "Switch to Dark Mode" if st.session_state.theme_mode == 'light' else "Switch to Light Mode"
        if st.button(theme_label, key="sidebar_theme_toggle", use_container_width=True):
            toggle_theme()
            st.rerun()

        st.markdown("""
            <div class="sidebar-copyright">
                © 2026 Zaki Firmansyah<br>All rights reserved.
            </div>
        """, unsafe_allow_html=True)

    # --- Routing Content ---
    if st.session_state.current_page == "Overview":
        show_overview()
    elif st.session_state.current_page == "Model Evaluation":
        show_model_evaluation()
    elif st.session_state.current_page == "Preprocessing":
        show_preprocessing()
    elif st.session_state.current_page == "Upload Dataset":
        show_upload_dataset()

if __name__ == "__main__":
    main()
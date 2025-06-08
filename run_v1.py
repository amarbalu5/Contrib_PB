import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import os
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import agentops
    AGENTOPS_AVAILABLE = True
except ImportError:
    AGENTOPS_AVAILABLE = False
    st.warning("AgentOps not installed. Multi-agent features will be limited.")

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    st.warning("CrewAI not installed. Multi-agent analysis will use basic summarization.")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    import plotly.io as pio
    import kaleido
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False
    st.warning("Install 'kaleido' package for PDF export functionality")

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Level Contribution Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data Models (fallback if pydantic not available)
if PYDANTIC_AVAILABLE:
    class ContributionData(BaseModel):
        """Data model for contribution analysis"""
        sbu: str = Field(..., description="Strategic Business Unit")
        department: str = Field(..., description="Department")
        wm_year_month: str = Field(..., description="Year-Month in YYYY-MM format")
        contribution_old: float = Field(..., description="Old contribution value")
        contribution_new: float = Field(..., description="New contribution value")
        retailer: str = Field(..., description="Retailer name")
        breakdown: str = Field(..., description="Breakdown category")
        data_source: str = Field(..., description="Data source")

    class HierarchyAnalysisResult(BaseModel):
        """Result model for hierarchy analysis"""
        sbu: str = Field(..., description="SBU")
        department: str = Field(..., description="Department")
        retailer: str = Field(..., description="Retailer")
        breakdown: str = Field(..., description="Breakdown")
        data_source: str = Field(..., description="Data source")
        total_periods: int = Field(..., description="Total time periods analyzed")
        consistent_periods: int = Field(..., description="Number of consistent periods")
        inconsistent_periods: int = Field(..., description="Number of inconsistent periods")
        avg_difference: float = Field(..., description="Average difference")
        max_difference: float = Field(..., description="Maximum difference")
        min_difference: float = Field(..., description="Minimum difference")
        std_difference: float = Field(..., description="Standard deviation of differences")
        trend_direction: str = Field(..., description="Whether new is generally higher or lower")
        consistency_rate: float = Field(..., description="Percentage of consistent periods")
        volatility_score: float = Field(..., description="Measure of contribution volatility")
        summary: str = Field(..., description="AI-generated summary")

# Dashboard Configuration
class DashboardConfig:
    """Configuration class for the dashboard"""
    THRESHOLD_DEFAULT = 0.000001
    GROQ_MODELS = [
        "llama3-70b-8192",
        "gemma-7b-it"
    ]
    
    # Only Level 1 - Full Detail
    HIERARCHY_LEVELS = {
        'Level 1 - Full Detail': ['SBU', 'Department', 'Retailer', 'Breakdown', 'Data_source']
    }

@st.cache_data
def load_and_process_data(uploaded_file) -> pd.DataFrame:
    """Load and preprocess the uploaded CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Convert contribution columns to numeric
        df['Contribution_old'] = pd.to_numeric(df['Contribution_old'], errors='coerce')
        df['Contribution_new'] = pd.to_numeric(df['Contribution_new'], errors='coerce')
        
        # Remove rows with NaN values in key columns
        df = df.dropna(subset=['Contribution_old', 'Contribution_new'])
        
        # Sort by date
        df = df.sort_values('wm_year_month')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def calculate_detailed_metrics(group_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """Calculate detailed metrics for a group"""
    # Calculate differences
    differences = abs(group_df['Contribution_new'] - group_df['Contribution_old'])
    is_consistent = differences <= threshold
    
    # Basic metrics
    total_periods = len(group_df)
    consistent_periods = is_consistent.sum()
    inconsistent_periods = total_periods - consistent_periods
    consistency_rate = (consistent_periods / total_periods) * 100 if total_periods > 0 else 0
    
    # Difference statistics
    avg_difference = differences.mean()
    max_difference = differences.max()
    min_difference = differences.min()
    std_difference = differences.std() if len(differences) > 1 else 0
    
    # Trend analysis
    avg_new = group_df['Contribution_new'].mean()
    avg_old = group_df['Contribution_old'].mean()
    if avg_new > avg_old * 1.01:  # 1% threshold for significance
        trend_direction = "Significantly Higher"
    elif avg_new < avg_old * 0.99:
        trend_direction = "Significantly Lower"
    else:
        trend_direction = "Similar"
    
    # Volatility score (coefficient of variation of differences)
    volatility_score = (std_difference / avg_difference) if avg_difference > 0 else 0
    
    return {
        'total_periods': total_periods,
        'consistent_periods': consistent_periods,
        'inconsistent_periods': inconsistent_periods,
        'consistency_rate': consistency_rate,
        'avg_difference': avg_difference,
        'max_difference': max_difference,
        'min_difference': min_difference,
        'std_difference': std_difference,
        'trend_direction': trend_direction,
        'volatility_score': volatility_score,
        'avg_old': avg_old,
        'avg_new': avg_new
    }

def analyze_hierarchy_level(df: pd.DataFrame, hierarchy_cols: List[str], level_name: str, threshold: float) -> List[Dict]:
    """Analyze contributions at a specific hierarchy level"""
    results = []
    
    # Group by the specified columns
    grouped = df.groupby(hierarchy_cols)
    
    for group_key, group_df in grouped:
        metrics = calculate_detailed_metrics(group_df, threshold)
        
        # Create separate columns for each hierarchy level
        if isinstance(group_key, tuple):
            hierarchy_dict = {col: val for col, val in zip(hierarchy_cols, group_key)}
        else:
            hierarchy_dict = {hierarchy_cols[0]: group_key}
        
        result = {
            **hierarchy_dict,
            **metrics,
            'summary': ""
        }
        
        results.append(result)
    
    return results

def generate_basic_summary(result: Dict) -> str:
    """Generate a basic summary without AI - simplified version"""
    group_desc = f"SBU: {result.get('SBU', 'N/A')}, Department: {result.get('Department', 'N/A')}, Retailer: {result.get('Retailer', 'N/A')}"
    
    summary = f"""
    **Analysis Summary for {group_desc}:**
    
    • **Consistency:** {result['consistency_rate']:.1f}% of periods are consistent
    • **Trend:** New contributions are {result['trend_direction'].lower()} than old contributions
    • **Average Difference:** {result['avg_difference']:.8f}
    • **Time Coverage:** {result['total_periods']} periods analyzed
    • **Inconsistent Periods:** {result['inconsistent_periods']} out of {result['total_periods']}
    • **Maximum Difference:** {result['max_difference']:.8f}
    """
    
    return summary

def generate_ai_summary_for_hierarchy(results: List[Dict], groq_client, level_name: str) -> List[Dict]:
    """Generate AI summaries for each result in a hierarchy level - simplified version"""
    if not GROQ_AVAILABLE or not groq_client:
        # Use basic summaries
        for result in results:
            result['summary'] = generate_basic_summary(result)
        return results
    
    try:
        for result in results:
            # Prepare detailed analysis data
            group_desc = f"SBU: {result.get('SBU', 'N/A')}, Department: {result.get('Department', 'N/A')}, Retailer: {result.get('Retailer', 'N/A')}"
            
            analysis_text = f"""
            Group: {group_desc}
            
            METRICS:
            - Total Periods: {result['total_periods']}
            - Consistent Periods: {result['consistent_periods']} ({result['consistency_rate']:.1f}%)
            - Inconsistent Periods: {result['inconsistent_periods']}
            - Average Difference: {result['avg_difference']:.8f}
            - Maximum Difference: {result['max_difference']:.8f}
            - Trend Direction: {result['trend_direction']}
            - Volatility Score: {result['volatility_score']:.2f}
            """
            
            try:
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a data analyst. Provide only factual analysis of the contribution data:
                            1. Key findings and trends (factual observations only)
                            2. Statistical summary
                            Focus on data patterns and numerical insights only. Do not include business implications, recommendations, or risk assessments."""
                        },
                        {
                            "role": "user",
                            "content": f"Provide factual data analysis for this contribution data:\n{analysis_text}"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                result['summary'] = response.choices[0].message.content
            except Exception as e:
                result['summary'] = generate_basic_summary(result)
                st.warning(f"Using basic summary for {group_desc}: {str(e)}")
        
        return results
        
    except Exception as e:
        st.error(f"Error generating AI summaries: {str(e)}")
        # Fallback to basic summaries
        for result in results:
            result['summary'] = generate_basic_summary(result)
        return results

def create_line_plot_for_group(df: pd.DataFrame, group_result: Dict) -> go.Figure:
    """Create line plot for a specific group"""
    # Extract group information and filter data
    filters = {
        'SBU': group_result.get('SBU'),
        'Department': group_result.get('Department'),
        'Retailer': group_result.get('Retailer'),
        'Breakdown': group_result.get('Breakdown'),
        'Data_source': group_result.get('Data_source')
    }
    
    # Filter dataframe
    filtered_df = df.copy()
    for col, val in filters.items():
        if val is not None:
            filtered_df = filtered_df[filtered_df[col] == val]
    
    # Create group description for title
    group_desc = f"SBU: {filters['SBU']}, Dept: {filters['Department']}, Retailer: {filters['Retailer']}"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['wm_year_month'],
        y=filtered_df['Contribution_old'],
        mode='lines+markers',
        name='Old Contribution',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df['wm_year_month'],
        y=filtered_df['Contribution_new'],
        mode='lines+markers',
        name='New Contribution',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title=f'Line Plot - {group_desc}',
        xaxis_title='Time Period',
        yaxis_title='Contribution Value',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def filter_results_by_hierarchy(results: List[Dict], hierarchy_filters: Dict) -> List[Dict]:
    """Filter results based on hierarchy selections"""
    filtered_results = []
    
    for result in results:
        include = True
        for filter_col, filter_values in hierarchy_filters.items():
            if filter_values and result.get(filter_col) not in filter_values:
                include = False
                break
        
        if include:
            filtered_results.append(result)
    
    return filtered_results

def create_multiple_line_plots(df: pd.DataFrame, results: List[Dict]) -> List[go.Figure]:
    """Create line plots for multiple groups"""
    figures = []
    
    for result in results:
        try:
            fig = create_line_plot_for_group(df, result)
            figures.append(fig)
        except Exception as e:
            group_desc = f"SBU: {result.get('SBU')}, Dept: {result.get('Department')}"
            st.error(f"Error creating plot for {group_desc}: {str(e)}")
            continue
    
    return figures

def export_plots_to_pdf(figures: List[go.Figure]) -> List[bytes]:
    """Export multiple plots to PDF - returns list of PDF bytes"""
    if not PDF_EXPORT_AVAILABLE:
        st.error("PDF export not available. Please install 'kaleido' package.")
        return []
    
    pdf_files = []
    try:
        for i, fig in enumerate(figures):
            pdf_bytes = fig.to_image(format="pdf", engine="kaleido")
            pdf_files.append(pdf_bytes)
        
        return pdf_files
    
    except Exception as e:
        st.error(f"Error creating PDFs: {str(e)}")
        return []

def export_hierarchy_report(results: List[Dict], level_name: str) -> str:
    """Export detailed report for a hierarchy level"""
    report = f"""
CONTRIBUTION ANALYSIS REPORT - {level_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
- Total Groups Analyzed: {len(results)}
- Overall Consistency Rate: {np.mean([r['consistency_rate'] for r in results]):.1f}%
- Average Volatility Score: {np.mean([r['volatility_score'] for r in results]):.2f}

DETAILED ANALYSIS BY GROUP:
{'='*50}
"""
    
    for i, result in enumerate(results, 1):
        group_desc = f"SBU: {result.get('SBU')}, Department: {result.get('Department')}, Retailer: {result.get('Retailer')}"
        report += f"""
GROUP {i}: {group_desc}
{'-'*40}
• Total Periods: {result['total_periods']}
• Consistent Periods: {result['consistent_periods']} ({result['consistency_rate']:.1f}%)
• Inconsistent Periods: {result['inconsistent_periods']}
• Average Difference: {result['avg_difference']:.8f}
• Maximum Difference: {result['max_difference']:.8f}
• Volatility Score: {result['volatility_score']:.2f}
• Trend: {result['trend_direction']}

SUMMARY:
{result['summary']}

"""
    
    return report

# Main Streamlit Application
def main():
    st.title("🎯 Contribution Analysis Dashboard")
    st.markdown("### Level 1 - Full Detail Analysis")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Keys
    groq_api_key = None
    if GROQ_AVAILABLE:
        groq_api_key = st.sidebar.text_input("Groq API Key", type="password", 
                                            help="Enter your Groq API key for AI summarization")
    else:
        st.sidebar.info("Install 'groq' package for AI-powered insights")
    
    # Analysis parameters
    threshold = st.sidebar.number_input("Difference Threshold", 
                                      value=DashboardConfig.THRESHOLD_DEFAULT,
                                      format="%.9f",
                                      help="Threshold for determining consistency")
    
    # Initialize APIs
    groq_client = None
    if groq_api_key and GROQ_AVAILABLE:
        try:
            groq_client = Groq(api_key=groq_api_key)
            st.sidebar.success("✅ Groq API connected")
        except Exception as e:
            st.sidebar.error(f"❌ Groq API error: {str(e)}")
    
    # File upload
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV file with contribution data", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading and processing data..."):
            df = load_and_process_data(uploaded_file)
        
        if not df.empty:
            st.success(f"✅ Data loaded successfully! {len(df)} rows processed.")
            
            # Create tabs
            tab1, tab2 = st.tabs(["📊 Analysis Results", "📈 Visualization & Export"])
            
            with tab1:
                # Data preview
                with st.expander("📊 Data Preview", expanded=False):
                    st.dataframe(df.head(), use_container_width=True)
                    st.write(f"**Shape:** {df.shape}")
                    st.write(f"**Columns:** {list(df.columns)}")
                
                # Analysis
                st.header("🔍 Level 1 - Full Detail Analysis")
                
                if st.button("🚀 Run Analysis", type="primary"):
                    hierarchy_cols = DashboardConfig.HIERARCHY_LEVELS['Level 1 - Full Detail']
                    
                    with st.spinner("Analyzing Level 1 - Full Detail..."):
                        # Analyze this hierarchy level
                        results = analyze_hierarchy_level(df, hierarchy_cols, 'Level 1 - Full Detail', threshold)
                        
                        # Generate summaries
                        results = generate_ai_summary_for_hierarchy(results, groq_client, 'Level 1 - Full Detail')
                        
                        # Store results in session state
                        st.session_state['analysis_results'] = results
                        st.session_state['dataframe'] = df
                    
                    st.success("✅ Analysis completed!")
                
                # Display results if available
                if 'analysis_results' in st.session_state:
                    results = st.session_state['analysis_results']
                    
                    # Hierarchy Filters
                    st.subheader("🔍 Hierarchy Filters")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    # Get unique values for each hierarchy level
                    unique_sbu = list(df['SBU'].unique())
                    unique_dept = list(df['Department'].unique())
                    unique_retailer = list(df['Retailer'].unique())
                    unique_breakdown = list(df['Breakdown'].unique())
                    unique_data_source = list(df['Data_source'].unique())
                    
                    with col1:
                        selected_sbu = st.multiselect("SBU", unique_sbu)
                    with col2:
                        selected_dept = st.multiselect("Department", unique_dept)
                    with col3:
                        selected_retailer = st.multiselect("Retailer", unique_retailer)
                    with col4:
                        selected_breakdown = st.multiselect("Breakdown", unique_breakdown)
                    with col5:
                        selected_data_source = st.multiselect("Data Source", unique_data_source)
                    
                    # Apply filters
                    hierarchy_filters = {
                        'SBU': selected_sbu,
                        'Department': selected_dept,
                        'Retailer': selected_retailer,
                        'Breakdown': selected_breakdown,
                        'Data_source': selected_data_source
                    }
                    
                    filtered_results = filter_results_by_hierarchy(results, hierarchy_filters)
                    
                    # Display filtered results
                    st.subheader("📊 Analysis Results")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Groups", len(filtered_results))
                    with col2:
                        if filtered_results:
                            avg_consistency = np.mean([r['consistency_rate'] for r in filtered_results])
                            st.metric("Avg Consistency", f"{avg_consistency:.1f}%")
                        else:
                            st.metric("Avg Consistency", "N/A")
                    with col3:
                        if filtered_results:
                            total_inconsistent = sum(r['inconsistent_periods'] for r in filtered_results)
                            st.metric("Total Inconsistent", total_inconsistent)
                        else:
                            st.metric("Total Inconsistent", "N/A")
                    with col4:
                        if filtered_results:
                            avg_volatility = np.mean([r['volatility_score'] for r in filtered_results])
                            st.metric("Avg Volatility", f"{avg_volatility:.2f}")
                        else:
                            st.metric("Avg Volatility", "N/A")
                    
                    if filtered_results:
                        # Results table (excluding summary column for display)
                        display_results = []
                        for result in filtered_results:
                            display_result = {k: v for k, v in result.items() if k != 'summary'}
                            display_results.append(display_result)
                        
                        results_df = pd.DataFrame(display_results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Detailed summaries
                        st.subheader("📝 Detailed Group Analysis")
                        for result in filtered_results:
                            group_desc = f"SBU: {result.get('SBU')}, Dept: {result.get('Department')}, Retailer: {result.get('Retailer')}"
                            with st.expander(f"📋 {group_desc}", expanded=False):
                                st.markdown(result['summary'])
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        with col1:
                            report_data = export_hierarchy_report(filtered_results, 'Level 1 - Full Detail')
                            st.download_button(
                                label="📄 Download Report",
                                data=report_data.encode('utf-8'),
                                file_name="contribution_analysis_full_detail.txt",
                                mime="text/plain"
                            )
                        
                        with col2:
                            csv_data = pd.DataFrame(display_results).to_csv(index=False)
                            st.download_button(
                                label="📊 Download Data",
                                data=csv_data,
                                file_name="analysis_data_full_detail.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("No results match the selected filters. Please adjust your filter criteria.")
            
            with tab2:
                st.header("📈 Visualization & Export")
                
                # Check if analysis results exist
                if 'analysis_results' not in st.session_state:
                    st.info("Please run the analysis first in the 'Analysis Results' tab.")
                else:
                    results = st.session_state['analysis_results']
                    df = st.session_state['dataframe']
                    
                    # Hierarchy Filters for Visualization
                    st.subheader("🔍 Visualization Filters")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    # Get unique values for each hierarchy level
                    unique_sbu = list(df['SBU'].unique())
                    unique_dept = list(df['Department'].unique())
                    unique_retailer = list(df['Retailer'].unique())
                    unique_breakdown = list(df['Breakdown'].unique())
                    unique_data_source = list(df['Data_source'].unique())
                    
                    with col1:
                        viz_selected_sbu = st.multiselect("SBU", unique_sbu, key="viz_sbu")
                    with col2:
                        viz_selected_dept = st.multiselect("Department", unique_dept, key="viz_dept")
                    with col3:
                        viz_selected_retailer = st.multiselect("Retailer", unique_retailer, key="viz_retailer")
                    with col4:
                        viz_selected_breakdown = st.multiselect("Breakdown", unique_breakdown, key="viz_breakdown")
                    with col5:
                        viz_selected_data_source = st.multiselect("Data Source", unique_data_source, key="viz_data_source")
                    
                    # Apply filters for visualization
                    viz_hierarchy_filters = {
                        'SBU': viz_selected_sbu,
                        'Department': viz_selected_dept,
                        'Retailer': viz_selected_retailer,
                        'Breakdown': viz_selected_breakdown,
                        'Data_source': viz_selected_data_source
                    }
                    
                    viz_filtered_results = filter_results_by_hierarchy(results, viz_hierarchy_filters)
                    
                    if viz_filtered_results:
                        st.info(f"📊 {len(viz_filtered_results)} groups selected for visualization")
                        
                        if st.button("🎨 Generate Line Plots", type="primary"):
                            with st.spinner("Generating line plots..."):
                                figures = create_multiple_line_plots(df, viz_filtered_results)
                                st.session_state['generated_figures'] = figures
                                
                                st.success(f"✅ Generated {len(figures)} line plots")
                                
                                # Display plots
                                for i, fig in enumerate(figures):
                                    st.plotly_chart(fig, use_container_width=True)
                                    if i < len(figures) - 1:  # Don't add divider after last plot
                                        st.divider()
                        
                        # PDF Export section
                        if 'generated_figures' in st.session_state:
                            st.subheader("📁 Export Options")
                            
                            if PDF_EXPORT_AVAILABLE:
                                if st.button("📄 Export All Plots to PDF", type="secondary"):
                                    with st.spinner("Creating PDFs..."):
                                        figures = st.session_state['generated_figures']
                                        pdf_files = export_plots_to_pdf(figures)
                                        
                                        if pdf_files:
                                            # Create individual PDF downloads for each plot
                                            for i, pdf_bytes in enumerate(pdf_files):
                                                group_desc = f"Group_{i+1}"
                                                if i < len(viz_filtered_results):
                                                    result = viz_filtered_results[i]
                                                    group_desc = f"SBU_{result.get('SBU', 'NA')}_Dept_{result.get('Department', 'NA')}_Plot_{i+1}"
                                                
                                                st.download_button(
                                                    label=f"📄 Download {group_desc} PDF",
                                                    data=pdf_bytes,
                                                    file_name=f"contribution_plot_{group_desc}.pdf",
                                                    mime="application/pdf",
                                                    key=f"pdf_download_{i}"
                                                )
                            else:
                                st.warning("PDF export not available. Please install 'kaleido' package.")
                                
                            # HTML export as alternative
                            st.subheader("💾 Alternative Export")
                            if st.button("📱 Export Plots as HTML"):
                                figures = st.session_state['generated_figures']
                                for i, fig in enumerate(figures):
                                    group_desc = f"Group_{i+1}"
                                    if i < len(viz_filtered_results):
                                        result = viz_filtered_results[i]
                                        group_desc = f"SBU_{result.get('SBU', 'NA')}_Dept_{result.get('Department', 'NA')}_Plot_{i+1}"
                                    
                                    html_str = fig.to_html(include_plotlyjs='cdn')
                                    st.download_button(
                                        label=f"📱 Download {group_desc} HTML",
                                        data=html_str.encode('utf-8'),
                                        file_name=f"contribution_plot_{group_desc}.html",
                                        mime="text/html",
                                        key=f"html_download_{i}"
                                    )
                    else:
                        st.info("No groups match the selected filters. Please adjust your filter criteria.")
        
        else:
            st.error("❌ Failed to load data. Please check your CSV file format.")
    
    else:
        st.info("👆 Please upload a CSV file to begin analysis.")
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        sample_data = pd.DataFrame({
            'SBU': ['FOOD', 'FOOD', 'FOOD'],
            'Department': ['Beer', 'Beer', 'Beer'],
            'wm_year_month': ['2019-01', '2019-02', '2019-03'],
            'Contribution_old': [0.01149, 0.00696, 0.00996],
            'Contribution_new': [0.00959, 0.01094, 0.00683],
            'Retailer': ['Walmart', 'Walmart', 'Walmart'],
            'Breakdown': ['PB', 'PB', 'PB'],
            'Data_source': ['IRI', 'IRI', 'IRI']
        })
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()
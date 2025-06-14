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
        hierarchy_level: str = Field(..., description="Hierarchy level name")
        group_key: str = Field(..., description="Group combination key")
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
        
        # Create group key string
        if isinstance(group_key, tuple):
            group_key_str = " | ".join([f"{col}: {val}" for col, val in zip(hierarchy_cols, group_key)])
        else:
            group_key_str = f"{hierarchy_cols[0]}: {group_key}"
        
        result = {
            'hierarchy_level': level_name,
            'group_key': group_key_str,
            **metrics,
            'summary': ""
        }
        
        results.append(result)
    
    return results

def generate_basic_summary(result: Dict) -> str:
    """Generate a basic summary without AI - simplified version"""
    summary = f"""
    **Analysis Summary for {result['group_key']}:**
    
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
            analysis_text = f"""
            Group: {result['group_key']}
            
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
                st.warning(f"Using basic summary for {result['group_key']}: {str(e)}")
        
        return results
        
    except Exception as e:
        st.error(f"Error generating AI summaries: {str(e)}")
        # Fallback to basic summaries
        for result in results:
            result['summary'] = generate_basic_summary(result)
        return results

def create_plot_for_group(df: pd.DataFrame, group_result: Dict, plot_type: str) -> go.Figure:
    """Create individual plot for a specific group"""
    # Extract group information
    group_key = group_result['group_key']
    
    # Parse group key to filter data
    filters = {}
    for part in group_key.split(' | '):
        col, val = part.split(': ', 1)
        filters[col] = val
    
    # Filter dataframe
    filtered_df = df.copy()
    for col, val in filters.items():
        filtered_df = filtered_df[filtered_df[col] == val]
    
    if plot_type == "Time Series":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['wm_year_month'],
            y=filtered_df['Contribution_old'],
            mode='lines+markers',
            name='Old Contribution',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df['wm_year_month'],
            y=filtered_df['Contribution_new'],
            mode='lines+markers',
            name='New Contribution',
            line=dict(color='red')
        ))
        fig.update_layout(
            title=f'Time Series - {group_key}',
            xaxis_title='Time Period',
            yaxis_title='Contribution Value',
            height=500
        )
    
    elif plot_type == "Difference Analysis":
        differences = filtered_df['Contribution_new'] - filtered_df['Contribution_old']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['wm_year_month'],
            y=differences,
            mode='lines+markers',
            name='Difference (New - Old)',
            line=dict(color='green')
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title=f'Difference Analysis - {group_key}',
            xaxis_title='Time Period',
            yaxis_title='Difference Value',
            height=500
        )
    
    elif plot_type == "Scatter Plot":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['Contribution_old'],
            y=filtered_df['Contribution_new'],
            mode='markers',
            name='Old vs New',
            marker=dict(size=8, color='purple')
        ))
        # Add diagonal line
        min_val = min(filtered_df['Contribution_old'].min(), filtered_df['Contribution_new'].min())
        max_val = max(filtered_df['Contribution_old'].max(), filtered_df['Contribution_new'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Equal Line',
            line=dict(dash='dash', color='gray')
        ))
        fig.update_layout(
            title=f'Old vs New Contributions - {group_key}',
            xaxis_title='Old Contribution',
            yaxis_title='New Contribution',
            height=500
        )
    
    elif plot_type == "Distribution":
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Old Contribution', 'New Contribution'])
        
        fig.add_trace(go.Histogram(
            x=filtered_df['Contribution_old'],
            name='Old',
            opacity=0.7,
            marker_color='blue'
        ), row=1, col=1)
        
        fig.add_trace(go.Histogram(
            x=filtered_df['Contribution_new'],
            name='New',
            opacity=0.7,
            marker_color='red'
        ), row=1, col=2)
        
        fig.update_layout(
            title=f'Distribution Analysis - {group_key}',
            height=500
        )
    
    return fig

def create_combined_plots(df: pd.DataFrame, results: List[Dict], plot_type: str) -> List[go.Figure]:
    """Create plots for all groups"""
    figures = []
    
    for result in results:
        try:
            fig = create_plot_for_group(df, result, plot_type)
            figures.append(fig)
        except Exception as e:
            st.error(f"Error creating plot for {result['group_key']}: {str(e)}")
            continue
    
    return figures

def export_plots_to_pdf(figures: List[go.Figure], filename: str):
    """Export multiple plots to PDF"""
    if not PDF_EXPORT_AVAILABLE:
        st.error("PDF export not available. Please install 'kaleido' package.")
        return None
    
    try:
        # Create a combined PDF
        pdf_bytes = io.BytesIO()
        
        # Convert each figure to PDF bytes and combine
        for i, fig in enumerate(figures):
            img_bytes = fig.to_image(format="pdf", engine="kaleido")
            if i == 0:
                pdf_bytes.write(img_bytes)
            else:
                # For multiple figures, we'll create separate downloads
                pass
        
        pdf_bytes.seek(0)
        return pdf_bytes.getvalue()
    
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

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
        report += f"""
GROUP {i}: {result['group_key']}
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
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Groups", len(results))
                    with col2:
                        avg_consistency = np.mean([r['consistency_rate'] for r in results])
                        st.metric("Avg Consistency", f"{avg_consistency:.1f}%")
                    with col3:
                        total_inconsistent = sum(r['inconsistent_periods'] for r in results)
                        st.metric("Total Inconsistent", total_inconsistent)
                    with col4:
                        avg_volatility = np.mean([r['volatility_score'] for r in results])
                        st.metric("Avg Volatility", f"{avg_volatility:.2f}")
                    
                    # Results table
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df.drop('summary', axis=1), use_container_width=True)
                    
                    # Detailed summaries
                    st.subheader("📝 Detailed Group Analysis")
                    for result in results:
                        with st.expander(f"📋 {result['group_key']}", expanded=False):
                            st.markdown(result['summary'])
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    with col1:
                        report_data = export_hierarchy_report(results, 'Level 1 - Full Detail')
                        st.download_button(
                            label="📄 Download Report",
                            data=report_data.encode('utf-8'),
                            file_name="contribution_analysis_full_detail.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        csv_data = pd.DataFrame(results).to_csv(index=False)
                        st.download_button(
                            label="📊 Download Data",
                            data=csv_data,
                            file_name="analysis_data_full_detail.csv",
                            mime="text/csv"
                        )
            
            with tab2:
                st.header("📈 Visualization & Export")
                
                # Check if analysis results exist
                if 'analysis_results' not in st.session_state:
                    st.info("Please run the analysis first in the 'Analysis Results' tab.")
                else:
                    results = st.session_state['analysis_results']
                    df = st.session_state['dataframe']
                    
                    # Plot configuration
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        plot_type = st.selectbox(
                            "Select Plot Type",
                            ["Time Series", "Difference Analysis", "Scatter Plot", "Distribution"]
                        )
                    
                    with col2:
                        group_selection = st.selectbox(
                            "Select Groups",
                            ["All Groups"] + [r['group_key'] for r in results]
                        )
                    
                    if st.button("🎨 Generate Plots", type="primary"):
                        with st.spinner("Generating plots..."):
                            if group_selection == "All Groups":
                                figures = create_combined_plots(df, results, plot_type)
                                st.session_state['generated_figures'] = figures
                                
                                st.success(f"✅ Generated {len(figures)} plots")
                                
                                # Display plots
                                for i, fig in enumerate(figures):
                                    st.plotly_chart(fig, use_container_width=True)
                                    if i < len(figures) - 1:  # Don't add divider after last plot
                                        st.divider()
                            
                            else:
                                # Single group plot
                                selected_result = next(r for r in results if r['group_key'] == group_selection)
                                fig = create_plot_for_group(df, selected_result, plot_type)
                                st.session_state['generated_figures'] = [fig]
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # PDF Export section
                    if 'generated_figures' in st.session_state:
                        st.subheader("📁 Export Options")
                        
                        if PDF_EXPORT_AVAILABLE:
                            if st.button("📄 Export Plots to PDF", type="secondary"):
                                with st.spinner("Creating PDF..."):
                                    figures = st.session_state['generated_figures']
                                    
                                    # Create individual PDF downloads for each plot
                                    for i, fig in enumerate(figures):
                                        try:
                                            pdf_bytes = fig.to_image(format="pdf", engine="kaleido")
                                            st.download_button(
                                                label=f"📄 Download Plot {i+1} PDF",
                                                data=pdf_bytes,
                                                file_name=f"contribution_plot_{i+1}.pdf",
                                                mime="application/pdf",
                                                key=f"pdf_download_{i}"
                                            )
                                        except Exception as e:
                                            st.error(f"Error creating PDF for plot {i+1}: {str(e)}")
                        else:
                            st.warning("PDF export not available. Please install 'kaleido' package.")
                            
                        # HTML export as alternative
                        st.subheader("💾 Alternative Export")
                        if st.button("📱 Export Plots as HTML"):
                            figures = st.session_state['generated_figures']
                            for i, fig in enumerate(figures):
                                html_str = fig.to_html(include_plotlyjs='cdn')
                                st.download_button(
                                    label=f"📱 Download Plot {i+1} HTML",
                                    data=html_str.encode('utf-8'),
                                    file_name=f"contribution_plot_{i+1}.html",
                                    mime="text/html",
                                    key=f"html_download_{i}"
                                )
        
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
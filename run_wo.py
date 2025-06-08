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

# Custom Summarization Tool
if CREWAI_AVAILABLE and GROQ_AVAILABLE:
    class SummaryTool(BaseTool):
        """Custom tool for Groq API summarization"""
        name: str = "summarization_tool"
        description: str = "Tool for generating summaries using Groq API"
        
        def __init__(self, groq_client):
            super().__init__()
            self.groq_client = groq_client
        
        def _run(self, analysis_data: str) -> str:
            """Generate summary using Groq API"""
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a senior data analyst expert specializing in retail contribution analysis. 
                            Provide clear, actionable insights focusing on business implications, trends, and recommendations.
                            Be specific about numbers and percentages when available."""
                        },
                        {
                            "role": "user",
                            "content": f"Analyze this contribution data and provide detailed business insights:\n{analysis_data}"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error generating summary: {str(e)}"

# Dashboard Configuration
class DashboardConfig:
    """Configuration class for the dashboard"""
    THRESHOLD_DEFAULT = 0.000001
    GROQ_MODELS = [
        "llama3-70b-8192",
        "gemma-7b-it"
    ]
    
    # Define hierarchy levels from most granular to least
    HIERARCHY_LEVELS = {
        'Level 1 - Full Detail': ['SBU', 'Department', 'Retailer', 'Breakdown', 'Data_source'],
        'Level 2 - By Data Source': ['SBU', 'Department', 'Retailer', 'Data_source'],
        'Level 3 - By Retailer': ['SBU', 'Department', 'Retailer'],
        'Level 4 - By Department': ['SBU', 'Department'],
        'Level 5 - By SBU': ['SBU'],
        'Level 6 - Overall': []
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
    
    if not hierarchy_cols:  # Overall analysis
        metrics = calculate_detailed_metrics(df, threshold)
        result = {
            'hierarchy_level': level_name,
            'group_key': 'Overall',
            **metrics,
            'summary': ""
        }
        results.append(result)
    else:
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
    """Generate a basic summary without AI"""
    summary = f"""
    **Analysis Summary for {result['group_key']}:**
    
    • **Consistency:** {result['consistency_rate']:.1f}% of periods are consistent (threshold: {DashboardConfig.THRESHOLD_DEFAULT})
    • **Trend:** New contributions are {result['trend_direction'].lower()} than old contributions
    • **Average Difference:** {result['avg_difference']:.8f}
    • **Volatility:** {'High' if result['volatility_score'] > 1 else 'Medium' if result['volatility_score'] > 0.5 else 'Low'} (Score: {result['volatility_score']:.2f})
    • **Time Coverage:** {result['total_periods']} periods analyzed
    
    **Key Insights:**
    - {result['inconsistent_periods']} out of {result['total_periods']} periods show significant differences
    - Maximum difference observed: {result['max_difference']:.8f}
    - Standard deviation of differences: {result['std_difference']:.8f}
    """
    
    if result['inconsistent_periods'] > result['consistent_periods']:
        summary += "\n⚠️ **Attention Required:** More inconsistent periods than consistent ones detected."
    
    return summary

def generate_ai_summary_for_hierarchy(results: List[Dict], groq_client, level_name: str) -> List[Dict]:
    """Generate AI summaries for each result in a hierarchy level"""
    if not GROQ_AVAILABLE or not groq_client:
        # Use basic summaries
        for result in results:
            result['summary'] = generate_basic_summary(result)
        return results
    
    try:
        for result in results:
            # Prepare detailed analysis data
            analysis_text = f"""
            Hierarchy Level: {level_name}
            Group: {result['group_key']}
            
            METRICS:
            - Total Periods: {result['total_periods']}
            - Consistent Periods: {result['consistent_periods']} ({result['consistency_rate']:.1f}%)
            - Inconsistent Periods: {result['inconsistent_periods']}
            - Average Difference: {result['avg_difference']:.8f}
            - Maximum Difference: {result['max_difference']:.8f}
            - Standard Deviation: {result['std_difference']:.8f}
            - Trend Direction: {result['trend_direction']}
            - Volatility Score: {result['volatility_score']:.2f}
            - Average Old Contribution: {result['avg_old']:.8f}
            - Average New Contribution: {result['avg_new']:.8f}
            
            CONTEXT:
            This is a contribution analysis comparing old vs new contribution values.
            Threshold for consistency: {DashboardConfig.THRESHOLD_DEFAULT}
            """
            
            try:
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a senior retail data analyst. Analyze the contribution data and provide:
                            1. Key findings and trends
                            2. Business implications
                            3. Specific recommendations
                            4. Risk assessment if applicable
                            Be concise but comprehensive. Use bullet points for clarity."""
                        },
                        {
                            "role": "user",
                            "content": f"Provide detailed analysis for this contribution data:\n{analysis_text}"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=800
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

def create_hierarchy_visualizations(df: pd.DataFrame, results: List[Dict], level_name: str) -> go.Figure:
    """Create visualizations for a hierarchy level"""
    
    if not results:
        return go.Figure()
    
    # Create consistency rate chart
    groups = [r['group_key'] for r in results]
    consistency_rates = [r['consistency_rate'] for r in results]
    volatility_scores = [r['volatility_score'] for r in results]
    avg_differences = [r['avg_difference'] for r in results]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Consistency Rate by Group',
            'Average Difference by Group',
            'Volatility Score by Group',
            'Trend Analysis'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Consistency rate chart
    colors_consistency = ['green' if rate >= 80 else 'orange' if rate >= 60 else 'red' for rate in consistency_rates]
    fig.add_trace(
        go.Bar(x=groups, y=consistency_rates, name='Consistency Rate (%)',
               marker_color=colors_consistency, showlegend=False),
        row=1, col=1
    )
    
    # Average difference chart
    fig.add_trace(
        go.Bar(x=groups, y=avg_differences, name='Avg Difference',
               marker_color='blue', showlegend=False),
        row=1, col=2
    )
    
    # Volatility chart  
    colors_volatility = ['red' if vol > 1 else 'orange' if vol > 0.5 else 'green' for vol in volatility_scores]
    fig.add_trace(
        go.Bar(x=groups, y=volatility_scores, name='Volatility Score',
               marker_color=colors_volatility, showlegend=False),
        row=2, col=1
    )
    
    # Trend analysis scatter
    trend_colors = {'Significantly Higher': 'green', 'Significantly Lower': 'red', 'Similar': 'blue'}
    for result in results:
        color = trend_colors.get(result['trend_direction'], 'gray')
        fig.add_trace(
            go.Scatter(x=[result['avg_old']], y=[result['avg_new']], 
                      name=result['group_key'], mode='markers',
                      marker=dict(color=color, size=10),
                      showlegend=False),
            row=2, col=2
        )
    
    # Add diagonal line for reference
    if results:
        min_val = min(min(r['avg_old'] for r in results), min(r['avg_new'] for r in results))
        max_val = max(max(r['avg_old'] for r in results), max(r['avg_new'] for r in results))
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', line=dict(dash='dash', color='gray'),
                      name='Equal Line', showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text=f"Analysis Dashboard - {level_name}")
    fig.update_xaxes(tickangle=45)
    
    return fig

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
    st.title("🎯 Multi-Level Contribution Analysis Dashboard")
    st.markdown("### Hierarchical Analysis with Detailed Insights")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Keys
    groq_api_key = None
    if GROQ_AVAILABLE:
        groq_api_key = st.sidebar.text_input("Groq API Key", type="password", 
                                            help="Enter your Groq API key for AI summarization")
    else:
        st.sidebar.info("Install 'groq' package for AI-powered insights")
    
    if AGENTOPS_AVAILABLE:
        agentops_api_key = st.sidebar.text_input("AgentOps API Key", type="password",
                                               help="Enter your AgentOps API key for monitoring")
    
    # Analysis parameters
    threshold = st.sidebar.number_input("Difference Threshold", 
                                      value=DashboardConfig.THRESHOLD_DEFAULT,
                                      format="%.9f",
                                      help="Threshold for determining consistency")
    
    if GROQ_AVAILABLE:
        selected_model = st.sidebar.selectbox("Groq Model", DashboardConfig.GROQ_MODELS)
    
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
            
            # Data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
                st.write(f"**Shape:** {df.shape}")
                st.write(f"**Columns:** {list(df.columns)}")
                
                # Data quality check
                missing_cols = set(DashboardConfig.HIERARCHY_LEVELS['Level 1 - Full Detail']) - set(df.columns)
                if missing_cols:
                    st.warning(f"⚠️ Missing columns: {missing_cols}")
            
            # Hierarchy Analysis
            st.header("🔍 Multi-Level Hierarchy Analysis")
            
            # Select hierarchy levels to analyze
            selected_levels = st.multiselect(
                "Select Hierarchy Levels to Analyze",
                options=list(DashboardConfig.HIERARCHY_LEVELS.keys()),
                default=list(DashboardConfig.HIERARCHY_LEVELS.keys())[:3],
                help="Choose which hierarchy levels to analyze"
            )
            
            if st.button("🚀 Run Multi-Level Analysis", type="primary"):
                all_results = {}
                
                for level_name in selected_levels:
                    hierarchy_cols = DashboardConfig.HIERARCHY_LEVELS[level_name]
                    
                    with st.spinner(f"Analyzing {level_name}..."):
                        # Analyze this hierarchy level
                        level_results = analyze_hierarchy_level(df, hierarchy_cols, level_name, threshold)
                        
                        # Generate summaries
                        level_results = generate_ai_summary_for_hierarchy(level_results, groq_client, level_name)
                        
                        all_results[level_name] = level_results
                
                # Display results by hierarchy level
                for level_name, level_results in all_results.items():
                    st.subheader(f"📊 {level_name}")
                    
                    # Create expandable sections for each level
                    with st.expander(f"View {level_name} Analysis", expanded=True):
                        
                        # Key metrics for this level
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Groups", len(level_results))
                        with col2:
                            avg_consistency = np.mean([r['consistency_rate'] for r in level_results])
                            st.metric("Avg Consistency", f"{avg_consistency:.1f}%")
                        with col3:
                            total_inconsistent = sum(r['inconsistent_periods'] for r in level_results)
                            st.metric("Total Inconsistent", total_inconsistent)
                        with col4:
                            avg_volatility = np.mean([r['volatility_score'] for r in level_results])
                            st.metric("Avg Volatility", f"{avg_volatility:.2f}")
                        
                        # Results table
                        results_df = pd.DataFrame(level_results)
                        st.dataframe(results_df.drop('summary', axis=1), use_container_width=True)
                        
                        # Visualizations
                        fig = create_hierarchy_visualizations(df, level_results, level_name)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed summaries for each group
                        st.subheader("📝 Detailed Group Analysis")
                        for result in level_results:
                            st.markdown(f"**{result['group_key']}**")
                            st.markdown(result['summary'])
                            st.divider()
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        with col1:
                            report_data = export_hierarchy_report(level_results, level_name)
                            st.download_button(
                                label=f"📄 Download {level_name} Report",
                                data=report_data.encode('utf-8'),
                                file_name=f"contribution_analysis_{level_name.replace(' ', '_')}.txt",
                                mime="text/plain"
                            )
                        
                        with col2:
                            csv_data = pd.DataFrame(level_results).to_csv(index=False)
                            st.download_button(
                                label=f"📊 Download {level_name} Data",
                                data=csv_data,
                                file_name=f"analysis_data_{level_name.replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                    
                    st.divider()
        
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
        
        # Show hierarchy levels
        st.subheader("🔍 Analysis Hierarchy Levels")
        for level, cols in DashboardConfig.HIERARCHY_LEVELS.items():
            if cols:
                st.write(f"**{level}:** {' → '.join(cols)}")
            else:
                st.write(f"**{level}:** Overall Analysis")

if __name__ == "__main__":
    main()
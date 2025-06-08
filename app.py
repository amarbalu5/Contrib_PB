import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import os
from typing import List, Dict, Any, Optional
import asyncio
import agentops
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Contribution Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pydantic Models
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

class AnalysisResult(BaseModel):
    """Result model for analysis summary"""
    group_key: str = Field(..., description="Group combination key")
    total_periods: int = Field(..., description="Total time periods analyzed")
    consistent_periods: int = Field(..., description="Number of consistent periods")
    inconsistent_periods: int = Field(..., description="Number of inconsistent periods")
    avg_difference: float = Field(..., description="Average difference")
    trend_direction: str = Field(..., description="Whether new is generally higher or lower")
    summary: str = Field(..., description="AI-generated summary")

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
                model="mixtral-8x7b-32768",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a data analyst expert. Analyze contribution data and provide clear, 
                        actionable insights. Focus on trends, patterns, and business implications."""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this contribution data and provide insights:\n{analysis_data}"
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

# Dashboard Configuration
class DashboardConfig:
    """Configuration class for the dashboard"""
    THRESHOLD_DEFAULT = 0.000001
    GROQ_MODELS = [
        "mixtral-8x7b-32768",
        "llama2-70b-4096",
        "gemma-7b-it"
    ]
    GROUP_COLUMNS = ['SBU', 'Department', 'Retailer', 'Breakdown', 'Data_source']

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
        
        # Sort by date
        df = df.sort_values('wm_year_month')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def analyze_contributions(df: pd.DataFrame, threshold: float) -> List[AnalysisResult]:
    """Analyze contribution differences by group"""
    results = []
    
    # Group by the specified columns
    grouped = df.groupby(DashboardConfig.GROUP_COLUMNS)
    
    for group_key, group_df in grouped:
        # Calculate differences
        group_df = group_df.copy()
        group_df['difference'] = abs(group_df['Contribution_new'] - group_df['Contribution_old'])
        group_df['is_consistent'] = group_df['difference'] <= threshold
        
        # Calculate metrics
        total_periods = len(group_df)
        consistent_periods = group_df['is_consistent'].sum()
        inconsistent_periods = total_periods - consistent_periods
        avg_difference = group_df['difference'].mean()
        
        # Determine trend direction
        avg_new = group_df['Contribution_new'].mean()
        avg_old = group_df['Contribution_old'].mean()
        trend_direction = "Higher" if avg_new > avg_old else "Lower" if avg_new < avg_old else "Similar"
        
        # Create group key string
        group_key_str = " | ".join([f"{col}: {val}" for col, val in zip(DashboardConfig.GROUP_COLUMNS, group_key)])
        
        result = AnalysisResult(
            group_key=group_key_str,
            total_periods=total_periods,
            consistent_periods=consistent_periods,
            inconsistent_periods=inconsistent_periods,
            avg_difference=avg_difference,
            trend_direction=trend_direction,
            summary=""  # Will be filled by AI
        )
        
        results.append(result)
    
    return results

def create_agents_and_crew(groq_client):
    """Create CrewAI agents and crew for analysis"""
    
    # Create summarization tool
    summary_tool = SummaryTool(groq_client)
    
    # Data Analyst Agent
    data_analyst = Agent(
        role='Senior Data Analyst',
        goal='Analyze contribution data patterns and identify key insights',
        backstory="""You are an experienced data analyst specializing in retail and business intelligence.
        You excel at identifying patterns, trends, and anomalies in large datasets.""",
        verbose=True,
        allow_delegation=False,
        tools=[summary_tool]
    )
    
    # Business Intelligence Agent
    bi_specialist = Agent(
        role='Business Intelligence Specialist',
        goal='Translate data insights into actionable business recommendations',
        backstory="""You are a BI specialist with deep knowledge of retail operations.
        You can translate complex data patterns into clear business insights and recommendations.""",
        verbose=True,
        allow_delegation=False,
        tools=[summary_tool]
    )
    
    # Quality Assurance Agent
    qa_agent = Agent(
        role='Quality Assurance Analyst',
        goal='Validate analysis results and ensure data quality',
        backstory="""You are a meticulous QA analyst who ensures all analysis results are accurate
        and reliable. You identify potential data quality issues and validation concerns.""",
        verbose=True,
        allow_delegation=False
    )
    
    return data_analyst, bi_specialist, qa_agent

def generate_ai_summary(analysis_results: List[AnalysisResult], groq_client) -> str:
    """Generate AI summary using CrewAI"""
    try:
        # Create agents
        data_analyst, bi_specialist, qa_agent = create_agents_and_crew(groq_client)
        
        # Prepare analysis data
        analysis_text = ""
        for result in analysis_results:
            analysis_text += f"""
            Group: {result.group_key}
            Total Periods: {result.total_periods}
            Consistent Periods: {result.consistent_periods}
            Inconsistent Periods: {result.inconsistent_periods}
            Average Difference: {result.avg_difference:.8f}
            Trend Direction: New contributions are {result.trend_direction}
            ---
            """
        
        # Create tasks
        analysis_task = Task(
            description=f"""Analyze the following contribution data patterns:
            {analysis_text}
            
            Identify key trends, patterns, and anomalies. Focus on:
            1. Which groups show the most inconsistency
            2. Overall trend patterns
            3. Potential data quality issues
            """,
            agent=data_analyst,
            expected_output="Detailed analysis of contribution data patterns and trends"
        )
        
        insights_task = Task(
            description="""Based on the data analysis, provide business insights and recommendations.
            Focus on actionable insights that can help business stakeholders understand the implications
            of the contribution changes.""",
            agent=bi_specialist,
            expected_output="Business insights and actionable recommendations"
        )
        
        validation_task = Task(
            description="""Review the analysis and insights for accuracy and completeness.
            Identify any potential issues or areas that need further investigation.""",
            agent=qa_agent,
            expected_output="Quality validation report and recommendations"
        )
        
        # Create crew
        crew = Crew(
            agents=[data_analyst, bi_specialist, qa_agent],
            tasks=[analysis_task, insights_task, validation_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute analysis
        result = crew.kickoff()
        return str(result)
        
    except Exception as e:
        return f"Error generating AI summary: {str(e)}"

def create_visualizations(df: pd.DataFrame, selected_group: str) -> Dict[str, Any]:
    """Create visualizations for selected group"""
    
    # Filter data for selected group
    group_parts = selected_group.split(" | ")
    filter_conditions = {}
    for part in group_parts:
        if ": " in part:
            key, value = part.split(": ", 1)
            filter_conditions[key] = value
    
    filtered_df = df.copy()
    for key, value in filter_conditions.items():
        if key in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[key] == value]
    
    # Sort by date
    filtered_df = filtered_df.sort_values('wm_year_month')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Contribution Trends Over Time',
            'Difference Analysis',
            'Distribution Comparison',
            'Monthly Variance'
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Time series
    fig.add_trace(
        go.Scatter(x=filtered_df['wm_year_month'], y=filtered_df['Contribution_old'],
                  name='Old Contribution', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=filtered_df['wm_year_month'], y=filtered_df['Contribution_new'],
                  name='New Contribution', line=dict(color='red')),
        row=1, col=1
    )
    
    # Plot 2: Difference analysis
    difference = filtered_df['Contribution_new'] - filtered_df['Contribution_old']
    fig.add_trace(
        go.Bar(x=filtered_df['wm_year_month'], y=difference,
               name='Difference (New - Old)', marker_color=np.where(difference >= 0, 'green', 'red')),
        row=1, col=2
    )
    
    # Plot 3: Distribution comparison
    fig.add_trace(
        go.Histogram(x=filtered_df['Contribution_old'], name='Old Distribution',
                    opacity=0.7, marker_color='blue'),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=filtered_df['Contribution_new'], name='New Distribution',
                    opacity=0.7, marker_color='red'),
        row=2, col=1
    )
    
    # Plot 4: Monthly variance
    monthly_stats = filtered_df.groupby('wm_year_month').agg({
        'Contribution_old': ['mean', 'std'],
        'Contribution_new': ['mean', 'std']
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(x=monthly_stats['wm_year_month'], 
                  y=monthly_stats[('Contribution_old', 'std')],
                  name='Old Std Dev', line=dict(color='lightblue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=monthly_stats['wm_year_month'], 
                  y=monthly_stats[('Contribution_new', 'std')],
                  name='New Std Dev', line=dict(color='lightcoral')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text=f"Analysis for: {selected_group}")
    
    return {"main_chart": fig, "filtered_data": filtered_df}

def export_to_pdf(fig, summary_text: str, group_name: str):
    """Export charts and summary to PDF"""
    try:
        # Convert plotly figure to image
        img_bytes = fig.to_image(format="png", width=1200, height=800)
        
        # Create a simple PDF-like export (in practice, you'd use reportlab)
        # For now, we'll create a comprehensive text report
        report = f"""
CONTRIBUTION ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

GROUP: {group_name}

SUMMARY:
{summary_text}

DATA INSIGHTS:
- Analysis based on contribution comparison
- Threshold-based consistency evaluation
- Time series trend analysis
- Statistical distribution comparison

Note: Chart visualization available in the dashboard interface.
        """
        
        return report.encode('utf-8')
    except Exception as e:
        st.error(f"Error creating PDF export: {str(e)}")
        return None

# Main Streamlit Application
def main():
    st.title("🎯 Multi-Agent Contribution Analysis Dashboard")
    st.markdown("### Powered by CrewAI, Groq API, and AgentOps")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Keys
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password", 
                                        help="Enter your Groq API key for AI summarization")
    agentops_api_key = st.sidebar.text_input("AgentOps API Key", type="password",
                                           help="Enter your AgentOps API key for monitoring")
    
    # Analysis parameters
    threshold = st.sidebar.number_input("Difference Threshold", 
                                      value=DashboardConfig.THRESHOLD_DEFAULT,
                                      format="%.9f",
                                      help="Threshold for determining consistency")
    
    selected_model = st.sidebar.selectbox("Groq Model", DashboardConfig.GROQ_MODELS)
    
    # Initialize APIs
    groq_client = None
    if groq_api_key:
        try:
            groq_client = Groq(api_key=groq_api_key)
            st.sidebar.success("✅ Groq API connected")
        except Exception as e:
            st.sidebar.error(f"❌ Groq API error: {str(e)}")
    
    if agentops_api_key:
        try:
            agentops.init(api_key=agentops_api_key)
            st.sidebar.success("✅ AgentOps connected")
        except Exception as e:
            st.sidebar.error(f"❌ AgentOps error: {str(e)}")
    
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
            
            # Create tabs
            tab1, tab2 = st.tabs(["📈 Analysis Summary", "📊 Detailed Charts"])
            
            with tab1:
                st.header("🤖 AI-Powered Analysis Summary")
                
                if st.button("🚀 Run Analysis", type="primary"):
                    with st.spinner("Analyzing contribution data..."):
                        # Perform analysis
                        analysis_results = analyze_contributions(df, threshold)
                        
                        if analysis_results:
                            # Display results table
                            results_df = pd.DataFrame([result.dict() for result in analysis_results])
                            st.subheader("📋 Analysis Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Generate AI summary if Groq is available
                            if groq_client:
                                with st.spinner("Generating AI insights..."):
                                    ai_summary = generate_ai_summary(analysis_results, groq_client)
                                    st.subheader("🧠 AI-Generated Insights")
                                    st.markdown(ai_summary)
                            else:
                                st.warning("⚠️ Add Groq API key for AI-powered insights")
                            
                            # Key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Groups", len(analysis_results))
                            with col2:
                                total_inconsistent = sum(r.inconsistent_periods for r in analysis_results)
                                st.metric("Total Inconsistent Periods", total_inconsistent)
                            with col3:
                                avg_diff = np.mean([r.avg_difference for r in analysis_results])
                                st.metric("Average Difference", f"{avg_diff:.8f}")
                            with col4:
                                consistency_rate = sum(r.consistent_periods for r in analysis_results) / sum(r.total_periods for r in analysis_results) * 100
                                st.metric("Overall Consistency Rate", f"{consistency_rate:.1f}%")
                        else:
                            st.warning("No analysis results generated. Please check your data.")
            
            with tab2:
                st.header("📊 Detailed Visualizations")
                
                # Group selection
                if 'analysis_results' in locals() and analysis_results:
                    selected_group = st.selectbox(
                        "Select Group for Detailed Analysis",
                        options=[result.group_key for result in analysis_results],
                        help="Choose a group combination to view detailed charts"
                    )
                    
                    if selected_group:
                        with st.spinner("Creating visualizations..."):
                            viz_data = create_visualizations(df, selected_group)
                            
                            st.plotly_chart(viz_data["main_chart"], use_container_width=True)
                            
                            # Export options
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if st.button("📄 Download Analysis Report"):
                                    if groq_client and 'ai_summary' in locals():
                                        pdf_data = export_to_pdf(viz_data["main_chart"], ai_summary, selected_group)
                                        if pdf_data:
                                            st.download_button(
                                                label="Download Report",
                                                data=pdf_data,
                                                file_name=f"contribution_analysis_{selected_group.replace(' | ', '_')}.txt",
                                                mime="text/plain"
                                            )
                                    else:
                                        st.warning("Run analysis first to generate report")
                            
                            with col2:
                                if st.button("📊 Download Chart Data"):
                                    csv_data = viz_data["filtered_data"].to_csv(index=False)
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv_data,
                                        file_name=f"filtered_data_{selected_group.replace(' | ', '_')}.csv",
                                        mime="text/csv"
                                    )
                else:
                    st.info("👆 Run analysis in the 'Analysis Summary' tab first to view detailed charts.")
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
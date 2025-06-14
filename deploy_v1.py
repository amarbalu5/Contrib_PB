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

# Try to import optional dependencies with better error handling
AGENTOPS_AVAILABLE = False
CREWAI_AVAILABLE = False
GROQ_AVAILABLE = False
PYDANTIC_AVAILABLE = False
PDF_EXPORT_AVAILABLE = False

try:
    import agentops
    AGENTOPS_AVAILABLE = True
except ImportError:
    pass

# Skip CrewAI import entirely to avoid ChromaDB issues
# try:
#     from crewai import Agent, Task, Crew, Process
#     from crewai.tools import BaseTool
#     CREWAI_AVAILABLE = True
# except ImportError:
#     CREWAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    pass

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    pass

try:
    import plotly.io as pio
    # Remove kaleido import as it can cause issues
    # import kaleido
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    pass

# Configure Streamlit page
st.set_page_config(
    page_title="Contribution Analysis Dashboard",
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
    THRESHOLD_DEFAULT = 0.001
    GROQ_MODELS = [
        "llama3-70b-8192",
        "gemma-7b-it"
    ]
    
    # Only Level 1 - Full Detail
    HIERARCHY_LEVELS = {
        'Full Detail': ['SBU', 'Department', 'Retailer', 'Breakdown', 'Data_source']
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

def create_formatted_title(group_result: Dict) -> str:
    """Create formatted title with underscores for the plot"""
    title_parts = []
    
    # Add each hierarchy level to title if it exists
    if group_result.get('SBU'):
        title_parts.append(str(group_result['SBU']))
    if group_result.get('Department'):
        title_parts.append(str(group_result['Department']))
    if group_result.get('Retailer'):
        title_parts.append(str(group_result['Retailer']))
    if group_result.get('Breakdown'):
        title_parts.append(str(group_result['Breakdown']))
    if group_result.get('Data_source'):
        title_parts.append(str(group_result['Data_source']))
    
    # Join with underscores
    formatted_title = "_".join(title_parts)
    return formatted_title

# =============================================================================
# PART 1: USER DOCUMENTATION FUNCTION
# Add this function after the existing utility functions (around line 300-400)
# =============================================================================

def show_user_documentation():
    """Display comprehensive user documentation for the dashboard"""
    st.markdown("""
    # 📚 User Documentation - Contribution Analysis Dashboard
    
    ## 🎯 Overview
    This dashboard provides comprehensive analysis of contribution data across different business hierarchies, 
    helping you identify inconsistencies, trends, and patterns in your data.
    
    ## 📋 Getting Started
    
    ### 1. Data Requirements
    Your CSV file should contain the following columns:
    - **SBU**: Strategic Business Unit
    - **Department**: Department name
    - **wm_year_month**: Year-Month in YYYY-MM format
    - **Contribution_old**: Original contribution values
    - **Contribution_new**: Updated contribution values
    - **Retailer**: Retailer name
    - **Breakdown**: Breakdown category
    - **Data_source**: Source of the data
    
    ### 2. Configuration
    - **Groq API Key**: Optional - enables AI-powered insights and summaries
    - **Difference Threshold**: Determines what constitutes a "consistent" vs "inconsistent" period
    
    ## 🔍 Features Overview
    
    ### Tab 1: Analysis Results
    - **Data Preview**: View your uploaded data structure
    - **Hierarchy Filters**: Filter results by SBU, Department, Retailer, etc.
    - **Key Metrics**: 
      - Total groups analyzed
      - Average consistency rate
      - Total inconsistent periods
      - Average volatility score
    - **Detailed Analysis**: Group-by-group breakdown with AI summaries
    - **Export Options**: Download reports and data as text/CSV files
    
    ### Tab 2: Visualization & Export
    - **Interactive Line Plots**: Compare old vs new contributions over time
    - **Filtering**: Select specific groups for visualization
    - **Export Options**: Download plots as PDF or HTML files
    
    ### Tab 3: Chat with Data
    - **Interactive Q&A**: Ask questions about your data in natural language
    - **Smart Analysis**: Get insights, summaries, and specific data points
    - **Context-Aware**: Understands your dataset structure and content
    
    ## 📊 Understanding the Metrics
    
    ### Consistency Analysis
    - **Consistent Periods**: Time periods where |new - old| ≤ threshold
    - **Inconsistent Periods**: Time periods where |new - old| > threshold
    - **Consistency Rate**: Percentage of consistent periods
    
    ### Trend Analysis
    - **Significantly Higher**: New contributions are >1% higher than old
    - **Significantly Lower**: New contributions are >1% lower than old
    - **Similar**: New contributions are within ±1% of old values
    
    ### Volatility Metrics
    - **Volatility Score**: Coefficient of variation of differences
    - **Standard Deviation**: Measure of difference variability
    - **Min/Max Difference**: Range of differences observed
    
    ## 🚀 Best Practices
    
    ### Data Preparation
    1. Ensure all required columns are present
    2. Check for missing values in contribution columns
    3. Verify date format (YYYY-MM)
    4. Remove or handle outliers if necessary
    
    ### Analysis Workflow
    1. Upload your CSV file
    2. Configure threshold and API keys if available
    3. Run the analysis
    4. Use filters to focus on specific segments
    5. Review detailed summaries for insights
    6. Generate visualizations for key groups
    7. Export results for further analysis
    
    ### Interpretation Tips
    - **High Consistency Rate (>90%)**: Data is stable, minor differences
    - **Medium Consistency Rate (70-90%)**: Some variability, investigate patterns
    - **Low Consistency Rate (<70%)**: Significant changes, requires attention
    - **High Volatility Score (>1.0)**: Unstable differences, erratic patterns
    
    ## ❓ Troubleshooting
    
    ### Common Issues
    - **Data not loading**: Check CSV format and column names
    - **No results**: Verify threshold setting and data quality
    - **API errors**: Check Groq API key validity
    - **Export failures**: Ensure required packages are installed
    
    ### Performance Tips
    - For large datasets (>10K rows), consider filtering before analysis
    - Use appropriate threshold values based on your data scale
    - Limit visualization to <20 groups for optimal performance
    
    ## 🔧 Technical Requirements
    
    ### Required Packages
    - streamlit
    - pandas
    - numpy
    - plotly
    
    ### Optional Packages (for enhanced features)
    - groq (AI summaries)
    - reportlab (PDF export)
    - pydantic (data validation)
    
    ## 📞 Support
    If you encounter issues or need additional features, please check:
    1. Data format requirements
    2. System status in the dashboard
    3. Error messages for specific guidance
    """)

# =============================================================================
# PART 2: CHAT WITH DATA FUNCTION
# Add this function after the documentation function
# =============================================================================

def chat_with_data_tab(df: pd.DataFrame, groq_client=None):
    """Chat interface for interacting with the uploaded data"""
    st.header("💬 Chat with Your Data")
    
    if groq_client is None:
        st.warning("⚠️ Groq API key required for chat functionality. Please add your API key in the sidebar.")
        st.info("The chat feature uses AI to answer questions about your data. Without an API key, this feature is not available.")
        return
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm here to help you analyze your contribution data. You can ask me questions like:\n\n"
                    "• What's the overall consistency rate?\n\n"
                    "• Which SBUs have the highest volatility?\n\n"
                    "• Show me trends for Walmart data\n\n"
                    "• What are the main differences between old and new contributions?\n\n"
                    "What would you like to know?"
                )
            }
        ]
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_question := st.chat_input("Ask a question about your data..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                try:
                    response = generate_data_chat_response(df, user_question, groq_client)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error while analyzing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Chat cleared! How can I help you analyze your data?"}
        ]
        st.rerun()

def generate_data_chat_response(df: pd.DataFrame, user_question: str, groq_client) -> str:
    """Generate AI response based on user question and data analysis"""
    try:
        # Prepare data summary for context
        data_summary = prepare_data_context(df)
        
        # Create the prompt
        system_prompt = """You are a data analyst assistant helping users understand their contribution analysis data. 
        
        You have access to a dataset with the following structure:
        - Columns: SBU, Department, wm_year_month, Contribution_old, Contribution_new, Retailer, Breakdown, Data_source
        - The data contains contribution values across different business units, departments, and time periods
        
        Provide helpful, accurate analysis based on the data summary provided. Be specific with numbers and insights.
        If asked about specific calculations, show your work. Keep responses concise but informative.
        
        Focus on:
        1. Direct answers to user questions
        2. Relevant statistics and insights
        3. Patterns and trends in the data
        4. Actionable observations
        
        Do not make recommendations about business decisions - stick to data analysis and observations."""
        
        user_prompt = f"""
        Based on this contribution analysis dataset:
        
        {data_summary}
        
        User Question: {user_question}
        
        Please provide a helpful analysis addressing their question.
        """
        
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or check your API connection."

def prepare_data_context(df: pd.DataFrame) -> str:
    """Prepare a summary of the dataset for AI context"""
    try:
        # Basic statistics
        total_rows = len(df)
        date_range = f"{df['wm_year_month'].min()} to {df['wm_year_month'].max()}"
        unique_sbu = df['SBU'].nunique()
        unique_dept = df['Department'].nunique()
        unique_retailer = df['Retailer'].nunique()
        
        # Contribution analysis
        df['difference'] = abs(df['Contribution_new'] - df['Contribution_old'])
        avg_old = df['Contribution_old'].mean()
        avg_new = df['Contribution_new'].mean()
        avg_difference = df['difference'].mean()
        max_difference = df['difference'].max()
        
        # Top entities by volume
        top_sbu = df['SBU'].value_counts().head(3).to_dict()
        top_retailers = df['Retailer'].value_counts().head(3).to_dict()
        
        context = f"""
        DATASET SUMMARY:
        - Total Records: {total_rows:,}
        - Date Range: {date_range}
        - Unique SBUs: {unique_sbu}
        - Unique Departments: {unique_dept}
        - Unique Retailers: {unique_retailer}
        
        CONTRIBUTION ANALYSIS:
        - Average Old Contribution: {avg_old:.6f}
        - Average New Contribution: {avg_new:.6f}
        - Average Absolute Difference: {avg_difference:.6f}
        - Maximum Difference: {max_difference:.6f}
        
        TOP SBUs BY RECORD COUNT:
        {dict(list(top_sbu.items())[:3])}
        
        TOP RETAILERS BY RECORD COUNT:
        {dict(list(top_retailers.items())[:3])}
        
        KEY COLUMNS AVAILABLE:
        - SBU, Department, Retailer, Breakdown, Data_source
        - wm_year_month (time dimension)
        - Contribution_old, Contribution_new (values to compare)
        """
        
        return context
    except Exception as e:
        return f"Error preparing data context: {str(e)}"

def create_line_plot_for_group(df: pd.DataFrame, group_result: Dict) -> go.Figure:
    """Create line plot for a specific group with formatted title"""
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
    
    # Create formatted title
    plot_title = create_formatted_title(group_result)
    
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
        title={
            'text': plot_title,
            'x': 0.5,  # Center the title
            'xanchor': 'center'
        },
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

def export_plots_to_html(figures: List[go.Figure], results: List[Dict]) -> str:
    """Export multiple plots to a single HTML file"""
    try:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Contribution Analysis Plots</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .plot-container { margin-bottom: 40px; }
                .plot-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
                hr { margin: 30px 0; }
            </style>
        </head>
        <body>
            <h1>Contribution Analysis Dashboard - Plots Export</h1>
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        """
        
        for i, (fig, result) in enumerate(zip(figures, results)):
            group_desc = f"SBU: {result.get('SBU')}, Dept: {result.get('Department')}, Retailer: {result.get('Retailer')}"
            
            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">Plot {i+1}: {group_desc}</div>
                <div id="plot_{i}"></div>
            </div>
            """
            
            if i < len(figures) - 1:
                html_content += "<hr>"
        
        html_content += """
        <script>
        """
        
        for i, fig in enumerate(figures):
            plot_json = fig.to_json()
            html_content += f"""
            Plotly.newPlot('plot_{i}', {plot_json});
            """
        
        html_content += """
        </script>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        st.error(f"Error creating HTML export: {str(e)}")
        return ""

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
    st.markdown("### Full Detail Analysis")
    
    # Show status of optional dependencies
    with st.expander("📋 System Status", expanded=False):
        st.write("**Optional Dependencies Status:**")
        st.write(f"• Groq API: {'✅ Available' if GROQ_AVAILABLE else '❌ Not installed'}")
        st.write(f"• Pydantic: {'✅ Available' if PYDANTIC_AVAILABLE else '❌ Not installed'}")
        st.write(f"• AgentOps: {'✅ Available' if AGENTOPS_AVAILABLE else '❌ Not installed'}")
        st.write(f"• PDF Export: {'✅ Available' if PDF_EXPORT_AVAILABLE else '❌ Not installed'}")
        st.write("• CrewAI: ❌ Disabled (to avoid deployment issues)")
    
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

    # Documentation is always available
    st.sidebar.markdown("---")
    if st.sidebar.button("📚 View Documentation"):
        st.session_state.show_docs = True

    if st.session_state.get('show_docs', False):
        show_user_documentation()
        if st.button("← Back to Dashboard"):
            st.session_state.show_docs = False
            st.rerun()
    else:
        if uploaded_file is not None:
            # Load data
            with st.spinner("Loading and processing data..."):
                df = load_and_process_data(uploaded_file)
            
            if not df.empty:
                st.success(f"✅ Data loaded successfully! {len(df)} rows processed.")

                # Create tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Results", "📈 Visualization & Export", "💬 Chat with Data", "📚 Documentation"])
                
                with tab1:
                    # Data preview
                    with st.expander("📊 Data Preview", expanded=False):
                        st.dataframe(df.head(), use_container_width=True)
                        st.write(f"**Shape:** {df.shape}")
                        st.write(f"**Columns:** {list(df.columns)}")
                    
                    # Analysis
                    st.header("🔍 Full Detail Analysis Report")
                    
                    if st.button("🚀 Run Analysis", type="primary"):
                        hierarchy_cols = DashboardConfig.HIERARCHY_LEVELS['Full Detail']
                        
                        with st.spinner("Analyzing Level 1 - Full Detail..."):
                            # Analyze this hierarchy level
                            results = analyze_hierarchy_level(df, hierarchy_cols, 'Full Detail', threshold)
                            
                            # Generate summaries
                            results = generate_ai_summary_for_hierarchy(results, groq_client, 'Full Detail')
                            
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
                                report_data = export_hierarchy_report(filtered_results, 'Full Detail')
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

                with tab3:
                    chat_with_data_tab(df, groq_client)
                
                with tab4:
                    show_user_documentation()
                
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
                                    st.session_state['viz_results'] = viz_filtered_results
                                    
                                    st.success(f"✅ Generated {len(figures)} line plots")
                                    
                                    # Display plots
                                    for i, fig in enumerate(figures):
                                        st.plotly_chart(fig, use_container_width=True)
                                        if i < len(figures) - 1:  # Don't add divider after last plot
                                            st.divider()
                            
                            # Export section
                            if 'generated_figures' in st.session_state:
                                st.subheader("📁 Export Options")
                                
                                if PDF_EXPORT_AVAILABLE:
                                    if st.button("📄 Export All Plots to Single PDF", type="secondary"):
                                        with st.spinner("Creating PDF..."):
                                            figures = st.session_state['generated_figures']
                                            pdf_bytes = export_plots_to_single_pdf(figures)
                                            
                                            if pdf_bytes:
                                                st.download_button(
                                                    label="📄 Download All Plots PDF",
                                                    data=pdf_bytes,
                                                    file_name="contribution_plots_all.pdf",
                                                    mime="application/pdf"
                                                )
                                else:
                                    st.warning("PDF export not available. Please install 'kaleido' package.")
                                    
                                # # HTML export as alternative
                                # st.subheader("💾 Alternative Export")
                                # if st.button("📱 Export Plots as HTML"):
                                #     figures = st.session_state['generated_figures']
                                #     for i, fig in enumerate(figures):
                                #         group_desc = f"Group_{i+1}"
                                #         if i < len(viz_filtered_results):
                                #             result = viz_filtered_results[i]
                                #             group_desc = f"SBU_{result.get('SBU', 'NA')}_Dept_{result.get('Department', 'NA')}_Plot_{i+1}"
                                        
                                #         html_str = fig.to_html(include_plotlyjs='cdn')
                                #         st.download_button(
                                #             label=f"📱 Download {group_desc} HTML",
                                #             data=html_str.encode('utf-8'),
                                #             file_name=f"contribution_plot_{group_desc}.html",
                                #             mime="text/html",
                                #             key=f"html_download_{i}"
                                #         )
                        else:
                            st.info("No groups match the selected filters. Please adjust your filter criteria.")
            
            else:
                st.error("❌ Failed to load data. Please check your CSV file format.")
        
        else:
            st.info("👆 Please upload a CSV file to begin analysis.")
            
            # Show sample data format
            st.subheader("📋 Expected Data Format")
            sample_data = pd.DataFrame({
                'SBU': ['FOOD'],
                'Department': ['Beer'],
                'wm_year_month': ['2019-01'],
                'Contribution_old': [0.01149],
                'Contribution_new': [0.00959],
                'Retailer': ['Walmart'],
                'Breakdown': ['PB'],
                'Data_source': ['IRI']
            })
            st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()
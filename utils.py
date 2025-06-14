"""
Utility functions for the Multi-Agent Streamlit Dashboard
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing utilities"""
    
    @staticmethod
    def validate_csv_structure(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate if the uploaded CSV has the required structure"""
        required_columns = [
            'SBU', 'Department', 'wm_year_month', 'Contribution_old', 
            'Contribution_new', 'Retailer', 'Breakdown', 'Data_source'
        ]
        
        validation_result = {
            'valid': True,
            'missing_columns': [],
            'extra_columns': [],
            'issues': [],
            'suggestions': []
        }
        
        # Check for missing required columns
        df_columns = [col.strip().replace(' ', '_') for col in df.columns]
        missing_cols = [col for col in required_columns if col not in df_columns]
        
        if missing_cols:
            validation_result['valid'] = False
            validation_result['missing_columns'] = missing_cols
            validation_result['issues'].append(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check for extra columns
        extra_cols = [col for col in df_columns if col not in required_columns]
        if extra_cols:
            validation_result['extra_columns'] = extra_cols
            validation_result['suggestions'].append(f"Extra columns found: {', '.join(extra_cols)}")
        
        # Validate data types
        if 'Contribution_old' in df_columns:
            if not pd.api.types.is_numeric_dtype(df['Contribution_old']):
                validation_result['issues'].append("Contribution_old should be numeric")
        
        if 'Contribution_new' in df_columns:
            if not pd.api.types.is_numeric_dtype(df['Contribution_new']):
                validation_result['issues'].append("Contribution_new should be numeric")
        
        # Check for date format
        if 'wm_year_month' in df_columns:
            try:
                pd.to_datetime(df['wm_year_month'].head())
            except:
                validation_result['issues'].append("wm_year_month format may be invalid (expected YYYY-MM)")
        
        return validation_result
    
    @staticmethod
    def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        try:
            df_clean = df.copy()
            
            # Standardize column names
            df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_')
            
            # Convert numeric columns
            numeric_columns = ['Contribution_old', 'Contribution_new']
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Handle missing values
            df_clean = df_clean.dropna(subset=numeric_columns)
            
            # Sort by date
            if 'wm_year_month' in df_clean.columns:
                df_clean = df_clean.sort_values('wm_year_month')
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates()
            
            logger.info(f"Data cleaned: {len(df_clean)} rows remaining from {len(df)} original rows")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return df
    
    @staticmethod
    def calculate_summary_statistics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Calculate summary statistics for grouped data"""
        try:
            # Group and calculate statistics
            grouped = df.groupby(group_cols).agg({
                'Contribution_old': ['count', 'mean', 'std', 'min', 'max'],
                'Contribution_new': ['count', 'mean', 'std', 'min', 'max'],
                'wm_year_month': ['min', 'max']
            }).reset_index()
            
            # Flatten column names
            grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
            
            # Calculate additional metrics
            grouped['avg_difference'] = grouped['Contribution_new_mean'] - grouped['Contribution_old_mean']
            grouped['period_span'] = grouped['wm_year_month_max'] - grouped['wm_year_month_min']
            
            return grouped
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")
            return pd.DataFrame()

class VisualizationUtils:
    """Visualization utility functions"""
    
    @staticmethod
    def create_comparison_chart(df: pd.DataFrame, title: str = "Contribution Comparison") -> go.Figure:
        """Create a comparison chart for old vs new contributions"""
        fig = go.Figure()
        
        # Add old contribution trace
        fig.add_trace(go.Scatter(
            x=df['wm_year_month'],
            y=df['Contribution_old'],
            mode='lines+markers',
            name='Old Contribution',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add new contribution trace
        fig.add_trace(go.Scatter(
            x=df['wm_year_month'],
            y=df['Contribution_new'],
            mode='lines+markers',
            name='New Contribution',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="Contribution Value",
            hovermode='x unified',
            showlegend=True,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_difference_chart(df: pd.DataFrame, threshold: float) -> go.Figure:
        """Create a chart showing differences between old and new contributions"""
        differences = df['Contribution_new'] - df['Contribution_old']
        
        # Color based on threshold
        colors = ['red' if abs(diff) > threshold else 'green' for diff in differences]
        
        fig = go.Figure(data=go.Bar(
            x=df['wm_year_month'],
            y=differences,
            marker_color=colors,
            name='Difference (New - Old)'
        ))
        
        # Add threshold lines
        fig.add_hline(y=threshold, line_dash="dash", line_color="orange", 
                     annotation_text=f"Threshold: +{threshold}")
        fig.add_hline(y=-threshold, line_dash="dash", line_color="orange",
                     annotation_text=f"Threshold: -{threshold}")
        
        fig.update_layout(
            title="Contribution Differences Analysis",
            xaxis_title="Time Period",
            yaxis_title="Difference Value",
            showlegend=True,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_distribution_comparison(df: pd.DataFrame) -> go.Figure:
        """Create distribution comparison chart"""
        fig = go.Figure()
        
        # Add old contribution histogram
        fig.add_trace(go.Histogram(
            x=df['Contribution_old'],
            name='Old Contribution',
            opacity=0.7,
            nbinsx=20,
            marker_color='blue'
        ))
        
        # Add new contribution histogram
        fig.add_trace(go.Histogram(
            x=df['Contribution_new'],
            name='New Contribution',
            opacity=0.7,
            nbinsx=20,
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Distribution Comparison",
            xaxis_title="Contribution Value",
            yaxis_title="Frequency",
            barmode='overlay',
            showlegend=True,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_heatmap(df: pd.DataFrame, group_cols: List[str]) -> go.Figure:
        """Create a heatmap of average differences by group"""
        try:
            # Calculate average difference by group
            df['difference'] = abs(df['Contribution_new'] - df['Contribution_old'])
            pivot_data = df.groupby(group_cols[0:2])['difference'].mean().reset_index()
            pivot_table = pivot_data.pivot(index=group_cols[0], columns=group_cols[1], values='difference')
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='Reds',
                showscale=True
            ))
            
            fig.update_layout(
                title=f"Average Difference Heatmap by {group_cols[0]} and {group_cols[1]}",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            return go.Figure()

class ExportUtils:
    """Export and download utilities"""
    
    @staticmethod
    def create_excel_report(analysis_results: List[Dict], summary_text: str) -> bytes:
        """Create comprehensive Excel report"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame(analysis_results)
                summary_df.to_excel(writer, sheet_name='Analysis_Summary', index=False)
                
                # AI Summary sheet
                summary_sheet_df = pd.DataFrame({'AI_Summary': [summary_text]})
                summary_sheet_df.to_excel(writer, sheet_name='AI_Insights', index=False)
                
                # Metadata sheet
                metadata = {
                    'Report_Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total_Groups_Analyzed': [len(analysis_results)],
                    'Analysis_Type': ['Contribution Comparison Analysis']
                }
                metadata_df = pd.DataFrame(metadata)
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            return b""
    
    @staticmethod
    def create_pdf_report(fig: go.Figure, summary_text: str, group_name: str) -> bytes:
        """Create PDF report with charts and summary"""
        try:
            # Convert figure to image
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Contribution Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .summary {{ margin: 20px 0; padding: 20px; background-color: #f5f5f5; }}
                    .chart {{ text-align: center; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Contribution Analysis Report</h1>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Group: {group_name}</p>
                </div>
                
                <div class="summary">
                    <h2>Analysis Summary</h2>
                    <p>{summary_text}</p>
                </div>
                
                <div class="chart">
                    <h2>Visualization</h2>
                    <img src="data:image/png;base64,{base64.b64encode(img_bytes).decode()}" alt="Analysis Chart"/>
                </div>
            </body>
            </html>
            """
            
            return html_content.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error creating PDF report: {str(e)}")
            return b""

class CacheUtils:
    """Caching utilities for performance optimization"""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def cached_data_processing(file_hash: str, data: bytes) -> pd.DataFrame:
        """Cache processed data based on file hash"""
        df = pd.read_csv(io.BytesIO(data))
        return DataProcessor.clean_and_prepare_data(df)
    
    @staticmethod
    def get_file_hash(uploaded_file) -> str:
        """Generate hash for uploaded file"""
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        return hashlib.md5(file_content).hexdigest()
    
    @staticmethod
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def cached_analysis_results(data_hash: str, threshold: float) -> List[Dict]:
        """Cache analysis results"""
        # This would contain the actual analysis logic
        # For now, return empty list as placeholder
        return []

class SessionManager:
    """Session state management utilities"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables"""
        default_values = {
            'analysis_complete': False,
            'current_data': None,
            'analysis_results': None,
            'selected_group': None,
            'ai_summary': None,
            'processing_status': 'idle'
        }
        
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def clear_session_state():
        """Clear all session state variables"""
        keys_to_clear = [
            'analysis_complete', 'current_data', 'analysis_results',
            'selected_group', 'ai_summary', 'processing_status'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def update_processing_status(status: str):
        """Update processing status in session state"""
        st.session_state.processing_status = status

class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    @staticmethod
    def measure_execution_time(func):
        """Decorator to measure function execution time"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def log_memory_usage():
        """Log current memory usage"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    @staticmethod
    def create_performance_metrics() -> Dict[str, Any]:
        """Create performance metrics for monitoring"""
        import psutil
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }

class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def validate_threshold(threshold: float) -> Tuple[bool, str]:
        """Validate threshold value"""
        if threshold <= 0:
            return False, "Threshold must be positive"
        if threshold > 1:
            return False, "Threshold seems unusually large (>1)"
        return True, "Valid"
    
    @staticmethod
    def validate_date_range(df: pd.DataFrame, date_col: str) -> Tuple[bool, str]:
        """Validate date range in dataset"""
        try:
            dates = pd.to_datetime(df[date_col])
            date_range = dates.max() - dates.min()
            
            if date_range.days < 30:
                return False, "Date range is too short for meaningful analysis"
            if date_range.days > 3650:  # 10 years
                return False, "Date range is unusually long, please verify data"
            
            return True, f"Valid date range: {date_range.days} days"
            
        except Exception as e:
            return False, f"Date validation error: {str(e)}"
    
    @staticmethod
    def validate_data_completeness(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness"""
        completeness_report = {
            'total_rows': len(df),
            'missing_data': {},
            'completeness_score': 0,
            'recommendations': []
        }
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            completeness_report['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_percent
            }
            
            if missing_percent > 10:
                completeness_report['recommendations'].append(
                    f"Column '{col}' has {missing_percent:.1f}% missing data"
                )
        
        # Calculate overall completeness score
        total_missing = sum([info['count'] for info in completeness_report['missing_data'].values()])
        total_cells = len(df) * len(df.columns)
        completeness_report['completeness_score'] = ((total_cells - total_missing) / total_cells) * 100
        
        return completeness_report

# Error handling utilities
class ErrorHandler:
    """Centralized error handling"""
    
    @staticmethod
    def handle_file_upload_error(error: Exception) -> str:
        """Handle file upload errors"""
        error_messages = {
            'EmptyDataError': "The uploaded file appears to be empty",
            'ParserError': "Error parsing the CSV file. Please check the format",
            'UnicodeDecodeError': "Error reading file encoding. Try saving as UTF-8",
            'MemoryError': "File is too large. Please reduce file size"
        }
        
        error_type = type(error).__name__
        return error_messages.get(error_type, f"Upload error: {str(error)}")
    
    @staticmethod
    def handle_api_error(error: Exception, api_name: str) -> str:
        """Handle API-related errors"""
        if "rate limit" in str(error).lower():
            return f"{api_name} rate limit exceeded. Please wait and try again"
        elif "authentication" in str(error).lower():
            return f"{api_name} authentication failed. Please check your API key"
        elif "timeout" in str(error).lower():
            return f"{api_name} request timed out. Please try again"
        else:
            return f"{api_name} error: {str(error)}"
    
    @staticmethod
    def log_error(error: Exception, context: str):
        """Log error with context"""
        logger.error(f"Error in {context}: {type(error).__name__}: {str(error)}")

# Initialize session state on module import
SessionManager.initialize_session_state()
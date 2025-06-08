"""
Configuration management for the Multi-Agent Streamlit Dashboard
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """API configuration settings"""
    groq_api_key: str = ""
    agentops_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    rate_limit_requests_per_minute: int = 30
    timeout_seconds: int = 30

@dataclass
class DashboardConfig:
    """Dashboard configuration settings"""
    app_title: str = "Multi-Agent Contribution Analysis Dashboard"
    page_icon: str = "📊"
    layout: str = "wide"
    threshold_default: float = 0.000001
    max_file_size_mb: int = 200
    supported_file_types: List[str] = None
    
    def __post_init__(self):
        if self.supported_file_types is None:
            self.supported_file_types = ['csv', 'xlsx', 'xls']

@dataclass
class AnalysisConfig:
    """Analysis configuration settings"""
    group_columns: List[str] = None
    groq_models: List[str] = None
    max_groups_for_analysis: int = 100
    min_periods_for_analysis: int = 3
    confidence_threshold: float = 0.95
    
    def __post_init__(self):
        if self.group_columns is None:
            self.group_columns = ['SBU', 'Department', 'Retailer', 'Breakdown', 'Data_source']
        if self.groq_models is None:
            self.groq_models = [
                "mixtral-8x7b-32768",
                "llama2-70b-4096",
                "gemma-7b-it",
                "llama3-8b-8192",
                "llama3-70b-8192"
            ]

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "dashboard.log"
    max_log_size_mb: int = 10
    backup_count: int = 5

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    encrypt_api_keys: bool = True
    secure_cookies: bool = True

class ConfigManager:
    """Main configuration manager class"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "config.yaml"
        self.api = APIConfig()
        self.dashboard = DashboardConfig()
        self.analysis = AnalysisConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        
        self._load_from_env()
        self._load_from_file()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # API Configuration
        self.api.groq_api_key = os.getenv('GROQ_API_KEY', '')
        self.api.agentops_api_key = os.getenv('AGENTOPS_API_KEY', '')
        self.api.rate_limit_requests_per_minute = int(
            os.getenv('RATE_LIMIT_RPM', self.api.rate_limit_requests_per_minute)
        )
        
        # Dashboard Configuration
        self.dashboard.threshold_default = float(
            os.getenv('THRESHOLD_DEFAULT', self.dashboard.threshold_default)
        )
        self.dashboard.max_file_size_mb = int(
            os.getenv('MAX_FILE_SIZE_MB', self.dashboard.max_file_size_mb)
        )
        
        # Analysis Configuration
        self.analysis.max_groups_for_analysis = int(
            os.getenv('MAX_GROUPS_ANALYSIS', self.analysis.max_groups_for_analysis)
        )
        
        # Logging Configuration
        self.logging.log_level = os.getenv('LOG_LEVEL', self.logging.log_level)
        
        # Security Configuration
        self.security.session_timeout_minutes = int(
            os.getenv('SESSION_TIMEOUT_MIN', self.security.session_timeout_minutes)
        )
    
    def _load_from_file(self):
        """Load configuration from YAML file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update configurations from file
                if 'api' in config_data:
                    for key, value in config_data['api'].items():
                        if hasattr(self.api, key):
                            setattr(self.api, key, value)
                
                if 'dashboard' in config_data:
                    for key, value in config_data['dashboard'].items():
                        if hasattr(self.dashboard, key):
                            setattr(self.dashboard, key, value)
                
                if 'analysis' in config_data:
                    for key, value in config_data['analysis'].items():
                        if hasattr(self.analysis, key):
                            setattr(self.analysis, key, value)
                
                if 'logging' in config_data:
                    for key, value in config_data['logging'].items():
                        if hasattr(self.logging, key):
                            setattr(self.logging, key, value)
                
                if 'security' in config_data:
                    for key, value in config_data['security'].items():
                        if hasattr(self.security, key):
                            setattr(self.security, key, value)
            
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def save_to_file(self):
        """Save current configuration to YAML file"""
        config_data = {
            'api': {
                'groq_base_url': self.api.groq_base_url,
                'rate_limit_requests_per_minute': self.api.rate_limit_requests_per_minute,
                'timeout_seconds': self.api.timeout_seconds
            },
            'dashboard': {
                'app_title': self.dashboard.app_title,
                'page_icon': self.dashboard.page_icon,
                'layout': self.dashboard.layout,
                'threshold_default': self.dashboard.threshold_default,
                'max_file_size_mb': self.dashboard.max_file_size_mb,
                'supported_file_types': self.dashboard.supported_file_types
            },
            'analysis': {
                'group_columns': self.analysis.group_columns,
                'groq_models': self.analysis.groq_models,
                'max_groups_for_analysis': self.analysis.max_groups_for_analysis,
                'min_periods_for_analysis': self.analysis.min_periods_for_analysis,
                'confidence_threshold': self.analysis.confidence_threshold
            },
            'logging': {
                'log_level': self.logging.log_level,
                'log_format': self.logging.log_format,
                'log_file': self.logging.log_file,
                'max_log_size_mb': self.logging.max_log_size_mb,
                'backup_count': self.logging.backup_count
            },
            'security': {
                'session_timeout_minutes': self.security.session_timeout_minutes,
                'max_login_attempts': self.security.max_login_attempts,
                'encrypt_api_keys': self.security.encrypt_api_keys,
                'secure_cookies': self.security.secure_cookies
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate API configuration
        if not self.api.groq_api_key:
            validation_results['warnings'].append("Groq API key not configured")
        
        if not self.api.agentops_api_key:
            validation_results['warnings'].append("AgentOps API key not configured")
        
        # Validate thresholds
        if self.dashboard.threshold_default <= 0:
            validation_results['errors'].append("Threshold must be positive")
            validation_results['valid'] = False
        
        # Validate file size limits
        if self.dashboard.max_file_size_mb <= 0:
            validation_results['errors'].append("Max file size must be positive")
            validation_results['valid'] = False
        
        # Validate analysis parameters
        if self.analysis.min_periods_for_analysis < 1:
            validation_results['errors'].append("Minimum periods for analysis must be >= 1")
            validation_results['valid'] = False
        
        if not (0 < self.analysis.confidence_threshold <= 1):
            validation_results['errors'].append("Confidence threshold must be between 0 and 1")
            validation_results['valid'] = False
        
        return validation_results
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get configuration suitable for Streamlit page config"""
        return {
            'page_title': self.dashboard.app_title,
            'page_icon': self.dashboard.page_icon,
            'layout': self.dashboard.layout,
            'initial_sidebar_state': 'expanded'
        }

# Global configuration instance
config = ConfigManager()

# Validation on import
validation_result = config.validate_config()
if not validation_result['valid']:
    print("Configuration validation errors:")
    for error in validation_result['errors']:
        print(f"  - {error}")

if validation_result['warnings']:
    print("Configuration warnings:")
    for warning in validation_result['warnings']:
        print(f"  - {warning}")
"""
Multi-Agent System using CrewAI for Contribution Analysis
"""

import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from groq import Groq
import json
import logging
from datetime import datetime
from pydantic import BaseModel, Field
import agentops
from utils import PerformanceMonitor, ErrorHandler

# Configure logging
logger = logging.getLogger(__name__)

class AnalysisInput(BaseModel):
    """Input model for analysis"""
    group_data: Dict[str, Any] = Field(..., description="Grouped analysis data")
    threshold: float = Field(..., description="Threshold for consistency analysis")
    time_periods: List[str] = Field(..., description="List of time periods")
    
class AnalysisOutput(BaseModel):
    """Output model for analysis results"""
    summary: str = Field(..., description="Analysis summary")
    insights: List[str] = Field(..., description="Key insights")
    recommendations: List[str] = Field(..., description="Recommendations")
    risk_level: str = Field(..., description="Risk assessment level")
    confidence_score: float = Field(..., description="Confidence in analysis")

class GroqSummarizationTool(BaseTool):
    """Tool for generating summaries using Groq API"""
    
    name: str = "groq_summarization"
    description: str = "Generate AI-powered summaries and insights using Groq models"
    
    def __init__(self, groq_client: Groq, model: str = "mixtral-8x7b-32768"):
        super().__init__()
        self.groq_client = groq_client
        self.model = model
    
    def _run(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Execute summarization using Groq API"""
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert data analyst specializing in retail analytics 
                        and contribution analysis. Provide clear, actionable insights with specific 
                        business implications."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return f"Error generating summary: {ErrorHandler.handle_api_error(e, 'Groq')}"

class DataAnalysisTool(BaseTool):
    """Tool for statistical data analysis"""
    
    name: str = "data_analysis"
    description: str = "Perform statistical analysis on contribution data"
    
    def _run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis"""
        try:
            analysis_results = {
                'statistical_summary': self._calculate_statistics(data),
                'trend_analysis': self._analyze_trends(data),
                'anomaly_detection': self._detect_anomalies(data),
                'seasonality_check': self._check_seasonality(data)
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Data analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_statistics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate basic statistics"""
        if 'differences' in data:
            differences = data['differences']
            return {
                'mean_difference': sum(differences) / len(differences),
                'std_difference': (sum([(x - sum(differences)/len(differences))**2 for x in differences]) / len(differences))**0.5,
                'max_difference': max(differences),
                'min_difference': min(differences)
            }
        return {}
    
    def _analyze_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in the data"""
        if 'time_series' in data:
            # Simple trend analysis
            values = data['time_series']
            if len(values) > 1:
                trend = "increasing" if values[-1] > values[0] else "decreasing"
                return {'trend_direction': trend, 'trend_strength': abs(values[-1] - values[0])}
        return {}
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in the data"""
        anomalies = []
        if 'differences' in data:
            differences = data['differences']
            mean_diff = sum(differences) / len(differences)
            std_diff = (sum([(x - mean_diff)**2 for x in differences]) / len(differences))**0.5
            
            for i, diff in enumerate(differences):
                if abs(diff - mean_diff) > 2 * std_diff:
                    anomalies.append({
                        'index': i,
                        'value': diff,
                        'deviation': abs(diff - mean_diff),
                        'severity': 'high' if abs(diff - mean_diff) > 3 * std_diff else 'medium'
                    })
        return anomalies
    
    def _check_seasonality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for seasonal patterns"""
        if 'time_series' in data and len(data['time_series']) >= 12:
            # Simple seasonality check
            values = data['time_series']
            seasonal_strength = self._calculate_seasonal_strength(values)
            return {
                'has_seasonality': seasonal_strength > 0.3,
                'seasonal_strength': seasonal_strength,
                'pattern': 'monthly' if seasonal_strength > 0.5 else 'weak'
            }
        return {'has_seasonality': False, 'seasonal_strength': 0.0}
    
    def _calculate_seasonal_strength(self, values: List[float]) -> float:
        """Calculate seasonal strength (simplified)"""
        if len(values) < 12:
            return 0.0
        
        # Calculate monthly averages for first year
        monthly_avg = [sum(values[i::12]) / len(values[i::12]) for i in range(12)]
        overall_avg = sum(values) / len(values)
        
        # Calculate variance
        seasonal_var = sum([(avg - overall_avg)**2 for avg in monthly_avg]) / 12
        total_var = sum([(val - overall_avg)**2 for val in values]) / len(values)
        
        return seasonal_var / total_var if total_var > 0 else 0.0

class ContributionAnalysisSystem:
    """Main system for contribution analysis using CrewAI"""
    
    def __init__(self, groq_api_key: str, agentops_api_key: Optional[str] = None):
        """Initialize the analysis system"""
        self.groq_client = Groq(api_key=groq_api_key)
        self.performance_monitor = PerformanceMonitor()
        
        if agentops_api_key:
            agentops.init(agentops_api_key)
        
        # Initialize tools
        self.groq_tool = GroqSummarizationTool(self.groq_client)
        self.data_analysis_tool = DataAnalysisTool()
        
        # Initialize agents
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup the analysis agents"""
        
        # Data Analyst Agent
        self.data_analyst = Agent(
            role='Senior Data Analyst',
            goal='Perform comprehensive statistical analysis of contribution data',
            backstory="""You are a senior data analyst with expertise in retail analytics 
            and statistical modeling. You excel at identifying patterns, trends, and 
            anomalies in complex datasets.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.data_analysis_tool]
        )
        
        # Business Intelligence Agent
        self.bi_analyst = Agent(
            role='Business Intelligence Specialist',
            goal='Translate data insights into actionable business recommendations',
            backstory="""You are a business intelligence specialist with deep understanding 
            of retail operations and strategic planning. You excel at connecting data 
            insights to business outcomes.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.groq_tool]
        )
        
        # Risk Assessment Agent
        self.risk_analyst = Agent(
            role='Risk Assessment Expert',
            goal='Evaluate risks and provide confidence assessments',
            backstory="""You are a risk assessment expert specializing in data quality 
            and business risk evaluation. You provide thorough risk assessments and 
            confidence scores for analytical findings.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.groq_tool, self.data_analysis_tool]
        )
        
        # Strategic Advisor Agent
        self.strategic_advisor = Agent(
            role='Strategic Business Advisor',
            goal='Synthesize all analyses into comprehensive strategic recommendations',
            backstory="""You are a strategic business advisor with extensive experience 
            in retail strategy and data-driven decision making. You excel at synthesizing 
            complex analyses into clear, actionable strategic guidance.""",
            verbose=True,
            allow_delegation=True,
            tools=[self.groq_tool]
        )
    
    def _create_tasks(self, analysis_input: AnalysisInput) -> List[Task]:
        """Create analysis tasks"""
        
        # Task 1: Statistical Analysis
        statistical_analysis_task = Task(
            description=f"""
            Perform comprehensive statistical analysis on the contribution data:
            - Analyze group data: {analysis_input.group_data}
            - Apply threshold analysis: {analysis_input.threshold}
            - Examine time periods: {analysis_input.time_periods}
            
            Focus on:
            1. Statistical summaries and distributions
            2. Trend identification and quantification
            3. Anomaly detection and classification
            4. Seasonality and pattern recognition
            
            Provide detailed statistical findings with quantitative metrics.
            """,
            agent=self.data_analyst,
            expected_output="Detailed statistical analysis report with metrics, trends, and anomalies identified"
        )
        
        # Task 2: Business Intelligence Analysis
        bi_analysis_task = Task(
            description="""
            Based on the statistical analysis, provide business intelligence insights:
            
            1. Interpret statistical findings in business context
            2. Identify key performance indicators and their implications
            3. Highlight areas of concern and opportunity
            4. Connect data patterns to potential business drivers
            
            Focus on actionable business insights rather than technical details.
            """,
            agent=self.bi_analyst,
            expected_output="Business intelligence report with key insights and their business implications",
            context=[statistical_analysis_task]
        )
        
        # Task 3: Risk Assessment
        risk_assessment_task = Task(
            description="""
            Evaluate the risks and provide confidence assessments:
            
            1. Assess data quality and reliability
            2. Evaluate the confidence level of findings
            3. Identify potential risks and their severity
            4. Provide uncertainty quantification
            
            Consider both analytical and business risks.
            """,
            agent=self.risk_analyst,
            expected_output="Risk assessment report with confidence scores and risk categorization",
            context=[statistical_analysis_task, bi_analysis_task]
        )
        
        # Task 4: Strategic Synthesis
        strategic_synthesis_task = Task(
            description="""
            Synthesize all previous analyses into comprehensive strategic recommendations:
            
            1. Integrate statistical, business, and risk perspectives
            2. Develop prioritized recommendations
            3. Create implementation roadmap
            4. Provide executive summary
            
            Ensure recommendations are specific, measurable, and actionable.
            """,
            agent=self.strategic_advisor,
            expected_output="Comprehensive strategic analysis with prioritized recommendations and implementation plan",
            context=[statistical_analysis_task, bi_analysis_task, risk_assessment_task]
        )
        
        return [statistical_analysis_task, bi_analysis_task, risk_assessment_task, strategic_synthesis_task]
    
    async def analyze(self, analysis_input: AnalysisInput) -> AnalysisOutput:
        """Execute the complete analysis workflow"""
        
        with self.performance_monitor.measure_execution():
            try:
                logger.info("Starting contribution analysis workflow")
                
                # Create tasks
                tasks = self._create_tasks(analysis_input)
                
                # Create crew
                crew = Crew(
                    agents=[self.data_analyst, self.bi_analyst, self.risk_analyst, self.strategic_advisor],
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=True
                )
                
                # Execute workflow
                result = crew.kickoff()
                
                # Parse and structure results
                analysis_output = self._parse_results(result, analysis_input)
                
                logger.info("Analysis workflow completed successfully")
                return analysis_output
                
            except Exception as e:
                logger.error(f"Analysis workflow failed: {str(e)}")
                return AnalysisOutput(
                    summary=f"Analysis failed: {str(e)}",
                    insights=["Error occurred during analysis"],
                    recommendations=["Review data quality and retry analysis"],
                    risk_level="high",
                    confidence_score=0.0
                )
    
    def _parse_results(self, crew_result: Any, analysis_input: AnalysisInput) -> AnalysisOutput:
        """Parse crew results into structured output"""
        
        try:
            # Extract key information from crew result
            result_text = str(crew_result)
            
            # Use Groq to structure the final output
            structuring_prompt = f"""
            Based on the following analysis results, create a structured summary:
            
            {result_text}
            
            Please provide:
            1. A concise executive summary (2-3 sentences)
            2. Top 3-5 key insights
            3. Top 3-5 specific recommendations
            4. Overall risk level (low/medium/high)
            5. Confidence score (0.0 to 1.0)
            
            Format as JSON with keys: summary, insights, recommendations, risk_level, confidence_score
            """
            
            structured_result = self.groq_tool._run(structuring_prompt, max_tokens=1500)
            
            try:
                parsed_result = json.loads(structured_result)
                return AnalysisOutput(
                    summary=parsed_result.get('summary', 'Analysis completed'),
                    insights=parsed_result.get('insights', []),
                    recommendations=parsed_result.get('recommendations', []),
                    risk_level=parsed_result.get('risk_level', 'medium'),
                    confidence_score=float(parsed_result.get('confidence_score', 0.7))
                )
            except json.JSONDecodeError:
                # Fallback parsing
                return self._fallback_parsing(result_text)
                
        except Exception as e:
            logger.error(f"Result parsing failed: {str(e)}")
            return AnalysisOutput(
                summary="Analysis completed with parsing issues",
                insights=["Analysis completed but results could not be fully structured"],
                recommendations=["Review analysis logs for detailed findings"],
                risk_level="medium",
                confidence_score=0.5
            )
    
    def _fallback_parsing(self, result_text: str) -> AnalysisOutput:
        """Fallback parsing when structured parsing fails"""
        
        # Simple text analysis for key information
        lines = result_text.split('\n')
        insights = []
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['insight', 'finding', 'pattern']):
                insights.append(line)
            elif any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should']):
                recommendations.append(line)
        
        return AnalysisOutput(
            summary="Multi-agent analysis completed successfully",
            insights=insights[:5] if insights else ["Analysis provided comprehensive findings"],
            recommendations=recommendations[:5] if recommendations else ["Implement data-driven improvements"],
            risk_level="medium",
            confidence_score=0.7
        )

# Usage Example
async def main():
    """Example usage of the contribution analysis system"""
    
    # Initialize system
    system = ContributionAnalysisSystem(
        groq_api_key="your-groq-api-key",
        agentops_api_key="your-agentops-api-key"  # Optional
    )
    
    # Prepare analysis input
    analysis_input = AnalysisInput(
        group_data={
            'differences': [0.05, -0.02, 0.08, -0.01, 0.12, -0.05, 0.03],
            'time_series': [100, 105, 103, 108, 107, 119, 114, 117],
            'categories': ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        },
        threshold=0.05,
        time_periods=['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4']
    )
    
    # Execute analysis
    result = await system.analyze(analysis_input)
    
    # Display results
    print("=== Contribution Analysis Results ===")
    print(f"Summary: {result.summary}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Confidence: {result.confidence_score:.2f}")
    
    print("\nKey Insights:")
    for i, insight in enumerate(result.insights, 1):
        print(f"{i}. {insight}")
    
    print("\nRecommendations:")
    for i, recommendation in enumerate(result.recommendations, 1):
        print(f"{i}. {recommendation}")

if __name__ == "__main__":
    asyncio.run(main())
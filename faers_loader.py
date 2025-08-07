import pandas as pd
import requests
from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FAERSLoader:
    """
    Loader for FDA Adverse Event Reporting System (FAERS) data.
    
    Provides functionality to:
    - Download FAERS data
    - Filter by drug and gender
    - Generate summary statistics
    """
    
    def __init__(self):
        """Initialize FAERS loader."""
        self.base_url = "https://api.fda.gov/drug/event.json"
        self.max_results = 100
        
    def search_faers(self, drug_name: str, limit: int = 100) -> Dict[str, Any]:
        """
        Search FAERS database for adverse events related to a drug.
        
        Args:
            drug_name: Name of the drug to search for
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        params = {
            'search': f'drug.medicinalproduct:"{drug_name}"',
            'limit': min(limit, self.max_results),
            'skip': 0
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error searching FAERS: {e}")
            raise
    
    def filter_by_gender(self, results: Dict[str, Any], gender: str) -> pd.DataFrame:
        """
        Filter FAERS results by patient gender.
        
        Args:
            results: FAERS search results
            gender: Gender to filter by ("F" or "M")
            
        Returns:
            DataFrame containing filtered results
        """
        events = []
        
        for result in results.get('results', []):
            patient = result.get('patient', {})
            patient_gender = patient.get('patientsex', '')
            
            if patient_gender == gender:
                events.append({
                    'case_id': patient.get('caseid', ''),
                    'drug_name': patient.get('drug', [{}])[0].get('medicinalproduct', ''),
                    'gender': patient_gender,
                    'age': patient.get('patientonsetage', ''),
                    'reaction': patient.get('reaction', [{}])[0].get('reactionmeddrapt', ''),
                    'severity': patient.get('reaction', [{}])[0].get('reactionoutcome', ''),
                    'date': patient.get('patientreactions', [{}])[0].get('reactionmeddraversionpt', '')
                })
        
        return pd.DataFrame(events)
    
    def generate_summary(self, drug_name: str) -> Dict[str, Any]:
        """
        Generate summary statistics for a drug's adverse events.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Dictionary containing summary statistics
        """
        try:
            # Search FAERS
            results = self.search_faers(drug_name)
            
            # Filter by gender
            female_events = self.filter_by_gender(results, 'F')
            male_events = self.filter_by_gender(results, 'M')
            
            # Generate summary
            summary = {
                'drug_name': drug_name,
                'total_events': len(female_events) + len(male_events),
                'female_events': len(female_events),
                'male_events': len(male_events),
                'female_percentage': (len(female_events) / (len(female_events) + len(male_events)) * 100) if (len(female_events) + len(male_events)) > 0 else 0,
                'top_female_reactions': female_events['reaction'].value_counts().head(5).to_dict(),
                'top_male_reactions': male_events['reaction'].value_counts().head(5).to_dict(),
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating FAERS summary: {e}")
            raise
    
    def create_demo_data(self, drug_name: str) -> pd.DataFrame:
        """
        Create a demo dataset with FAERS-like structure.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            DataFrame containing demo data
        """
        data = {
            'case_id': [f"CASE{str(i).zfill(6)}" for i in range(100)],
            'drug_name': [drug_name] * 100,
            'gender': np.random.choice(['F', 'M'], 100, p=[0.55, 0.45]),  # Slight female bias
            'age': np.random.normal(55, 15, 100).astype(int),
            'reaction': np.random.choice([
                'Bleeding',
                'Dizziness',
                'Nausea',
                'Headache',
                'Fatigue'
            ], 100),
            'severity': np.random.choice(['Serious', 'Non-serious'], 100, p=[0.2, 0.8]),
            'date': [datetime.now() - timedelta(days=random.randint(0, 365)) for _ in range(100)]
        }
        
        return pd.DataFrame(data)

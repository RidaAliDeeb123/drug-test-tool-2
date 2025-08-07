import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Generator for synthetic healthcare data with controlled gender bias.
    
    This class can generate:
    - Patient demographics
    - Drug administration records
    - Adverse events
    - Lab results
    """
    
    def __init__(self):
        """Initialize the synthetic data generator."""
        self.gender_distribution = {
            'female': 0.5,  # Base distribution
            'male': 0.5
        }
        
        self.drug_risk_profiles = {
            'warfarin': {
                'female_risk_multiplier': 1.5,
                'male_risk_multiplier': 1.0,
                'base_risk': 0.05,
                'lab_impact': ['INR', 'PT']
            },
            'zolpidem': {
                'female_risk_multiplier': 2.0,
                'male_risk_multiplier': 1.0,
                'base_risk': 0.03,
                'lab_impact': ['Sedation', 'Respiratory Rate']
            },
            'lisinopril': {
                'female_risk_multiplier': 1.2,
                'male_risk_multiplier': 1.0,
                'base_risk': 0.04,
                'lab_impact': ['Creatinine', 'Potassium']
            }
        }
    
    def generate_patient_data(self, n_patients: int = 100, bias: float = 0.0) -> pd.DataFrame:
        """
        Generate synthetic patient data with controlled gender bias.
        
        Args:
            n_patients: Number of patients to generate
            bias: Gender bias factor (-1 to 1, where 0 is no bias)
            
        Returns:
            DataFrame containing synthetic patient data
        """
        # Adjust gender distribution based on bias
        female_prob = 0.5 + (bias * 0.5)
        male_prob = 1 - female_prob
        
        # Generate patient data
        data = {
            'patient_id': [f"PAT{str(i).zfill(5)}" for i in range(n_patients)],
            'gender': np.random.choice(['female', 'male'], 
                                     p=[female_prob, male_prob],
                                     size=n_patients),
            'age': np.random.normal(55, 15, n_patients).astype(int),
            'weight': np.random.normal(70, 15, n_patients).astype(int),
            'height': np.random.normal(165, 10, n_patients).astype(int)
        }
        
        return pd.DataFrame(data)
    
    def generate_drug_administration(self, patients: pd.DataFrame, drugs: List[str]) -> pd.DataFrame:
        """
        Generate synthetic drug administration records.
        
        Args:
            patients: DataFrame containing patient data
            drugs: List of drugs to administer
            
        Returns:
            DataFrame containing drug administration records
        """
        data = []
        
        for _, patient in patients.iterrows():
            for drug in drugs:
                # Randomize start date
                start_date = datetime.now() - timedelta(days=random.randint(0, 30))
                
                # Calculate dose based on gender and drug
                base_dose = self._calculate_base_dose(drug, patient['gender'])
                
                # Add some variation
                dose = base_dose * (1 + np.random.normal(0, 0.1))
                
                data.append({
                    'patient_id': patient['patient_id'],
                    'drug': drug,
                    'dose': dose,
                    'start_date': start_date,
                    'end_date': start_date + timedelta(days=random.randint(7, 30)),
                    'gender': patient['gender']
                })
        
        return pd.DataFrame(data)
    
    def generate_adverse_events(self, drug_admin: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic adverse events based on drug-gender interactions.
        
        Args:
            drug_admin: DataFrame containing drug administration records
            
        Returns:
            DataFrame containing adverse event records
        """
        events = []
        
        for _, admin in drug_admin.iterrows():
            drug = admin['drug']
            gender = admin['gender']
            
            # Get risk profile
            if drug not in self.drug_risk_profiles:
                continue
                
            profile = self.drug_risk_profiles[drug]
            
            # Calculate risk based on gender
            if gender == 'female':
                risk = profile['base_risk'] * profile['female_risk_multiplier']
            else:
                risk = profile['base_risk'] * profile['male_risk_multiplier']
            
            # Randomly generate adverse event
            if np.random.random() < risk:
                event_date = admin['start_date'] + timedelta(days=random.randint(1, 14))
                severity = random.choice(['mild', 'moderate', 'severe'])
                
                events.append({
                    'patient_id': admin['patient_id'],
                    'drug': drug,
                    'event_date': event_date,
                    'severity': severity,
                    'gender': gender
                })
        
        return pd.DataFrame(events)
    
    def _calculate_base_dose(self, drug: str, gender: str) -> float:
        """
        Calculate base dose for a drug based on gender.
        
        Args:
            drug: Name of the drug
            gender: Patient gender
            
        Returns:
            Base dose recommendation
        """
        # Default dose
        base_dose = 5.0  # mg
        
        # Adjust based on drug-gender interaction
        if drug in self.drug_risk_profiles:
            profile = self.drug_risk_profiles[drug]
            
            if gender == 'female':
                # Reduce dose for higher risk drugs in females
                if profile['female_risk_multiplier'] > 1.2:
                    base_dose *= 0.75
        
        return base_dose
    
    def create_demo_dataset(self, n_patients: int = 100, drugs: List[str] = None, bias: float = 0.0):
        """
        Create a complete demo dataset with synthetic data.
        
        Args:
            n_patients: Number of patients to generate
            drugs: List of drugs to include
            bias: Gender bias factor (-1 to 1)
            
        Returns:
            Tuple of (patients, drug_admin, adverse_events)
        """
        if drugs is None:
            drugs = list(self.drug_risk_profiles.keys())
            
        # Generate patient data
        patients = self.generate_patient_data(n_patients, bias)
        
        # Generate drug administration
        drug_admin = self.generate_drug_administration(patients, drugs)
        
        # Generate adverse events
        adverse_events = self.generate_adverse_events(drug_admin)
        
        return patients, drug_admin, adverse_events

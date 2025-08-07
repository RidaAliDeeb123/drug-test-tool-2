import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DrugResponseDataPreparer:
    """
    Class for preparing drug response data for machine learning models.
    Handles:
    - Feature engineering
    - Data cleaning
    - Creating synthetic data
    - Generating gender-specific features
    """
    
    def __init__(self):
        """
        Initialize the data preparer.
        """
        self.feature_columns = [
            'gender', 'age', 'dosage', 'drug',
            'weight', 'height', 'bmi', 'genetic_marker',
            'medical_history', 'concomitant_medications'
        ]
        
    def create_synthetic_data(
        self,
        n_samples: int = 1000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Create synthetic drug response data for training models.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame containing synthetic data
        """
        np.random.seed(random_state)
        
        # Generate synthetic features
        data = {
            'gender': np.random.choice(['male', 'female'], n_samples),
            'age': np.random.normal(50, 15, n_samples).astype(int),
            'dosage': np.random.uniform(0.1, 100, n_samples),
            'drug': np.random.choice([
                'DrugA', 'DrugB', 'DrugC', 'DrugD', 'DrugE'
            ], n_samples),
            'weight': np.random.normal(70, 15, n_samples),
            'height': np.random.normal(170, 10, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'genetic_marker': np.random.choice([
                'AA', 'AG', 'GG', 'TT', 'TC', 'CC'
            ], n_samples),
            'medical_history': np.random.choice([
                'None', 'Hypertension', 'Diabetes', 'Asthma'
            ], n_samples),
            'concomitant_medications': np.random.choice([
                'None', 'Aspirin', 'Statins', 'Anticoagulants'
            ], n_samples)
        }
        
        # Generate synthetic response based on features
        # Example: Higher dosage and male gender increase response probability
        response_prob = 0.5 + 0.1 * (data['dosage'] / 100) + 0.1 * (data['gender'] == 'male')
        response_prob = np.clip(response_prob, 0, 1)
        data['response'] = np.random.binomial(1, response_prob)
        
        return pd.DataFrame(data)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from existing data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Create interaction features
        df['age_dosage'] = df['age'] * df['dosage']
        df['bmi_dosage'] = df['bmi'] * df['dosage']
        
        # Create gender-specific features
        df['is_male'] = (df['gender'] == 'male').astype(int)
        df['male_dosage'] = df['is_male'] * df['dosage']
        df['female_dosage'] = (1 - df['is_male']) * df['dosage']
        
        # Create categorical interactions
        df['drug_gender'] = df['drug'] + '_' + df['gender']
        
        # Create risk scores
        df['risk_score'] = (
            df['age'] * 0.1 +
            df['dosage'] * 0.2 +
            df['bmi'] * 0.1 +
            df['is_male'] * 0.1
        )
        
        return df
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Prepare data for model training.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train/test splits and preprocessing info
        """
        # Engineer features
        df = self.engineer_features(df)
        
        # Split data
        X = df.drop(columns=['response'])
        y = df['response']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': X.columns.tolist()
        }
    
    def generate_gender_specific_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate gender-specific features for analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with gender-specific features
        """
        df_gender = pd.DataFrame()
        
        # Gender-specific statistics
        for gender in ['male', 'female']:
            gender_df = df[df['gender'] == gender]
            df_gender[f'{gender}_mean_dosage'] = [gender_df['dosage'].mean()]
            df_gender[f'{gender}_std_dosage'] = [gender_df['dosage'].std()]
            df_gender[f'{gender}_response_rate'] = [
                gender_df['response'].mean()
            ]
            
        # Gender differences
        df_gender['dosage_difference'] = (
            df_gender['male_mean_dosage'] - df_gender['female_mean_dosage']
        )
        df_gender['response_difference'] = (
            df_gender['male_response_rate'] - df_gender['female_response_rate']
        )
        
        return df_gender

import pandas as pd
from fhir.resources.bundle import Bundle
from fhir.resources.patient import Patient
from fhir.resources.medicationstatement import MedicationStatement
from typing import Union, Dict, Any
import json

class DataIngestor:
    def __init__(self):
        """Initialize the data ingestor with various data source handlers."""
        self.handlers = {
            'csv': self._handle_csv,
            'ehr': self._handle_ehr,
            'manual': self._handle_manual
        }

    def load_medical_data(self, source_type: str, source: Union[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Load medical data from various sources.
        
        Args:
            source_type: Type of data source ('csv', 'ehr', or 'manual')
            source: Source data (path for CSV, FHIR bundle for EHR, dict for manual)
            
        Returns:
            pd.DataFrame: Cleaned and standardized data
        """
        if source_type not in self.handlers:
            raise ValueError(f"Unsupported source type: {source_type}")
            
        return self.handlers[source_type](source)

    def _handle_csv(self, file_path: str) -> pd.DataFrame:
        """Handle CSV data source."""
        df = pd.read_csv(file_path)
        return self._standardize_columns(df)

    def _handle_ehr(self, fhir_bundle: Dict[str, Any]) -> pd.DataFrame:
        """Handle EHR data source using FHIR format."""
        bundle = Bundle(**fhir_bundle)
        patients = []
        
        for entry in bundle.entry:
            resource = entry.resource
            if isinstance(resource, Patient):
                patient_data = {
                    'patient_id': resource.id,
                    'gender': resource.gender,
                    'birth_date': resource.birthDate
                }
            elif isinstance(resource, MedicationStatement):
                patient_data.update({
                    'medication_code': resource.medicationCodeableConcept.coding[0].code,
                    'dosage': resource.dosage[0].doseAndRate[0].doseQuantity.value if resource.dosage else None
                })
                patients.append(patient_data)
        
        return pd.DataFrame(patients)

    def _handle_manual(self, data_dict: Dict[str, Any]) -> pd.DataFrame:
        """Handle manual data entry."""
        df = pd.DataFrame([data_dict])
        return self._standardize_columns(df)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types."""
        column_mappings = {
            'sex': 'gender',
            'dose': 'dosage',
            'drug': 'medication_code'
        }
        
        df = df.rename(columns=column_mappings)
        df['gender'] = df['gender'].str.lower()
        return df

    def detect_bias(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect gender bias in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict: Gender distribution and bias metrics
        """
        gender_counts = df['gender'].value_counts(normalize=True)
        male_ratio = gender_counts.get('male', 0)
        female_ratio = gender_counts.get('female', 0)
        
        return {
            'male_ratio': male_ratio,
            'female_ratio': female_ratio,
            'bias_score': abs(male_ratio - female_ratio),
            'is_biased': abs(male_ratio - female_ratio) > 0.2
        }

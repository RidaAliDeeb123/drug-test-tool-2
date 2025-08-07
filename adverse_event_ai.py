import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class AdverseEventPredictor:
    """
    AI-powered adverse event prediction system using BioClinicalBERT.
    
    This model predicts adverse events based on:
    - Drug-gender interactions
    - Patient demographics
    - Medical history
    """
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        """
        Initialize the adverse event predictor.
        
        Args:
            model_name: Name of the transformer model to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def prepare_model(self):
        """Prepare the transformer model architecture."""
        if self.model is None:
            self.model = nn.Sequential(
                AutoModel.from_pretrained(self.model_name),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)  # Binary classification for adverse event
            )
            
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess clinical text for model input.
        
        Args:
            text: Clinical text to process
            
        Returns:
            Dictionary of tokenized input
        """
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    
    def predict_adverse_event(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict adverse event risk for a patient.
        
        Args:
            patient_data: Dictionary containing patient information
                e.g., {
                    "gender": "female",
                    "age": 45,
                    "drug": "warfarin",
                    "medical_history": "Hypertension, diabetes"
                }
                
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Prepare input text
            input_text = self._format_input_text(patient_data)
            inputs = self.preprocess_text(input_text)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                risk_score = torch.sigmoid(outputs).item()
                
            return {
                'risk_score': risk_score,
                'risk_category': 'High' if risk_score > 0.7 else 'Low',
                'confidence': f"{risk_score:.2%}"
            }
            
        except Exception as e:
            logger.error(f"Error in adverse event prediction: {e}")
            raise
    
    def _format_input_text(self, patient_data: Dict[str, Any]) -> str:
        """
        Format patient data into a structured clinical text.
        
        Args:
            patient_data: Patient information dictionary
            
        Returns:
            Formatted clinical text
        """
        return f"""
        Patient Profile:
        Gender: {patient_data['gender']}
        Age: {patient_data['age']}
        Drug: {patient_data['drug']}
        Medical History: {patient_data.get('medical_history', 'None')}
        """
    
    def create_interactive_ai(self):
        """Create an interactive adverse event prediction UI component."""
        st.header("Adverse Event AI")
        
        # Patient inputs
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=18, max_value=100, value=45)
        with col2:
            drug = st.text_input("Drug Name", "warfarin")
            medical_history = st.text_area("Medical History", "")
        
        # Predict adverse event risk
        if st.button("Predict Adverse Event Risk"):
            try:
                patient_data = {
                    'gender': gender,
                    'age': age,
                    'drug': drug,
                    'medical_history': medical_history
                }
                
                prediction = self.predict_adverse_event(patient_data)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Adverse Event Risk",
                        prediction['risk_category'],
                        delta=prediction['confidence']
                    )
                with col2:
                    st.write("### Risk Factors")
                    st.write(f"- Gender: {gender}")
                    st.write(f"- Age: {age}")
                    st.write(f"- Drug: {drug}")
                    if medical_history:
                        st.write(f"- Medical History: {medical_history}")
                
            except Exception as e:
                logger.error(f"Error in adverse event prediction UI: {e}")
                st.error("Error predicting adverse event risk. Please check your inputs.")

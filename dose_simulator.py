import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)

class DoseSimulator:
    """
    Interactive dose simulator that calculates optimal drug dosages based on:
    - Patient gender
    - Age
    - Weight
    - Drug-specific pharmacokinetic parameters
    """
    
    def __init__(self, drug_params: Dict[str, Dict[str, Any]]):
        """
        Initialize the dose simulator with drug-specific parameters.
        
        Args:
            drug_params: Dictionary containing drug-specific parameters
                e.g., {"warfarin": {"half_life": 36, "clearance": 0.15, "volume_distribution": 10}}
        """
        self.drug_params = drug_params
        self.models = {}
        
    def train_dose_response_model(self, drug_name: str, df: pd.DataFrame):
        """
        Train a dose-response model for a specific drug.
        
        Args:
            drug_name: Name of the drug
            df: DataFrame containing dose-response data
        """
        # Prepare features: gender, age, weight, dose
        features = ['gender', 'age', 'weight', 'dose']
        X = df[features]
        y = df['response']
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        self.models[drug_name] = {
            'model': model,
            'poly_features': poly,
            'features': features
        }
        
    def simulate_dose(self, drug_name: str, patient_data: Dict[str, Any]) -> float:
        """
        Simulate optimal dose for a patient.
        
        Args:
            drug_name: Name of the drug
            patient_data: Dictionary containing patient information
                e.g., {"gender": "female", "age": 45, "weight": 65}
                
        Returns:
            float: Optimal dose recommendation
        """
        if drug_name not in self.models:
            raise ValueError(f"No model trained for drug: {drug_name}")
            
        model_data = self.models[drug_name]
        features = model_data['features']
        
        # Prepare patient data
        patient_df = pd.DataFrame([patient_data])[features]
        X_poly = model_data['poly_features'].transform(patient_df)
        
        # Predict response
        predicted_response = model_data['model'].predict(X_poly)[0]
        
        # Adjust dose based on response prediction
        base_dose = patient_data['dose']
        adjusted_dose = base_dose * (1 + predicted_response)
        
        return adjusted_dose
    
    def create_interactive_simulator(self, drug_name: str):
        """
        Create an interactive dose simulator UI component.
        
        Args:
            drug_name: Name of the drug to simulate
        """
        st.header(f"{drug_name} Dose Simulator")
        
        # Patient inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        with col2:
            age = st.slider("Age", 18, 100, 45)
        with col3:
            weight = st.slider("Weight (kg)", 40, 120, 70)
        
        # Base dose input
        base_dose = st.slider("Base Dose", 0.1, 10.0, 5.0, 0.1)
        
        # Calculate optimal dose
        if st.button("Calculate Optimal Dose"):
            try:
                patient_data = {
                    'gender': gender.lower(),
                    'age': age,
                    'weight': weight,
                    'dose': base_dose
                }
                
                optimal_dose = self.simulate_dose(drug_name, patient_data)
                
                st.metric(
                    "Optimal Dose",
                    f"{optimal_dose:.2f} mg",
                    delta=f"{optimal_dose - base_dose:.2f} mg",
                    delta_color="inverse" if optimal_dose < base_dose else "normal"
                )
                
                # Show dose-response curve
                self._plot_dose_response(drug_name, patient_data)
                
            except Exception as e:
                logger.error(f"Error in dose simulation: {e}")
                st.error("Error calculating optimal dose. Please check your inputs.")
    
    def _plot_dose_response(self, drug_name: str, patient_data: Dict[str, Any]):
        """
        Plot the dose-response curve for visualization.
        
        Args:
            drug_name: Name of the drug
            patient_data: Patient data dictionary
        """
        if drug_name not in self.models:
            return
            
        # Generate dose range
        doses = np.linspace(0.1, 10.0, 100)
        responses = []
        
        # Calculate response for each dose
        for dose in doses:
            patient_data['dose'] = dose
            response = self.simulate_dose(drug_name, patient_data)
            responses.append(response)
            
        # Create plot
        fig = go.Figure()
        
        # Add dose-response curve
        fig.add_trace(go.Scatter(
            x=doses,
            y=responses,
            mode='lines',
            name='Dose-Response Curve'
        ))
        
        # Add optimal dose marker
        optimal_dose = patient_data['dose']
        fig.add_trace(go.Scatter(
            x=[optimal_dose],
            y=[responses[np.argmin(np.abs(doses - optimal_dose))]],
            mode='markers',
            name='Optimal Dose'
        ))
        
        # Update layout
        fig.update_layout(
            title='Dose-Response Curve',
            xaxis_title='Dose (mg)',
            yaxis_title='Response',
            showlegend=True
        )
        
        st.plotly_chart(fig)

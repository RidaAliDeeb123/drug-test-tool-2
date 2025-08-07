import requests
import pandas as pd
from typing import Dict, Any, List, Optional, Set, Union, Tuple
from datetime import datetime, timedelta
import os
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from requests.exceptions import RequestException
from typing_extensions import TypedDict

# Constants
CACHE_TTL = timedelta(days=7)  # Cache expiration time
API_BASE_URL = "https://api.fda.gov/drug/label.json"
DEFAULT_LIMIT = 100

@dataclass
class DrugMetadata:
    """Data structure for storing drug metadata."""
    manufacturer: str
    approval_date: str
    indications: List[str]
    contraindications: List[str]
    clinical_trials: List[str]
    pharmacology: Dict[str, Any]
    mechanism_of_action: List[str]


class GenderSpecificInfo(TypedDict):
    """Typed dictionary for storing gender-specific drug information."""
    warnings: List[str]
    dose_adjustments: List[str]
    adverse_events: List[str]
    pharmacokinetics: List[str]
    pharmacodynamics: List[str]
    clinical_studies: List[str]
    risk_factors: List[str]
    pregnancy_category: Optional[str]
    lactation_risk: Optional[str]


class FDADrugData:
    """
    A comprehensive handler for FDA drug data with caching and advanced analysis capabilities.
    
    This class provides methods to:
    - Search and retrieve drug information from FDA's openFDA API
    - Cache API responses for improved performance
    - Extract and analyze gender-specific drug information
    - Calculate gender bias scores
    - Handle drug interactions and dosing guidelines
    
    Attributes:
        api_key: Optional API key for openFDA API
        cache_dir: Directory for storing cached data
        cache_ttl: Time-to-live for cached data
        _cache: Internal cache storage
    """

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = '.fda_cache'):
        """
        Initialize the FDA drug data handler.
        
        Args:
            api_key: Optional API key for openFDA API
            cache_dir: Directory to store cached data
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = CACHE_TTL
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached data from disk."""
        try:
            cache_file = self.cache_dir / 'fda_cache.pkl'
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            cache_file = self.cache_dir / 'fda_cache.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _load_cache(self):
        """Load cached data from disk"""
        try:
            cache_file = self.cache_dir / 'fda_cache.pkl'
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
            else:
                self._cache = {}
        except:
            self._cache = {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            cache_file = self.cache_dir / 'fda_cache.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
        except:
            pass

    def _is_cached(self, key: str) -> bool:
        """
        Check if an item is in cache and not expired.
        
        Args:
            key: Cache key to check
            
        Returns:
            bool: True if item is cached and not expired, False otherwise
        """
        if key not in self._cache:
            return False
            
        cached_time = self._cache[key]['timestamp']
        return datetime.now() - cached_time < self.cache_ttl

    def search_drug(self, drug_name: str, limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
        """
        Search for drug information by name with caching.
        
        Args:
            drug_name: Name of the drug to search for
            limit: Maximum number of results to return (default: 100)
            
        Returns:
            List of drug information dictionaries
            
        Raises:
            RequestException: If the API request fails
        """
        cache_key = f"search_{drug_name}_{limit}"
        
        if self._is_cached(cache_key):
            return self._cache[cache_key]['data']
            
        try:
            response = requests.get(
                API_BASE_URL,
                params={
                    'search': f'drug_name:{drug_name}',
                    'limit': limit,
                    'api_key': self.api_key or ''
                }
            )
            response.raise_for_status()
            results = response.json().get('results', [])
            
            # Cache the results
            self._cache[cache_key] = {
                'data': results,
                'timestamp': datetime.now()
            }
            self._save_cache()
            
            return results
            
        except RequestException as e:
            print(f"Error searching drug: {e}")
            raise

    def get_gender_specific_info(self, drug_data: Dict[str, Any]) -> GenderSpecificInfo:
        """
        Extract comprehensive gender-specific information from drug data.
        
        Args:
            drug_data: Dictionary containing drug information
            
        Returns:
            GenderSpecificInfo: Dictionary with detailed gender-specific information
        """
        gender_info: GenderSpecificInfo = {
            'gender_warnings': [],
            'dose_adjustments': [],
            'adverse_events': [],
            'pharmacokinetics': [],
            'pharmacodynamics': [],
            'clinical_studies': [],
            'risk_factors': [],
            'pregnancy_category': None,
            'lactation_risk': None
        }
        
        # Extract gender-specific information from various sections
        for section, key in [
            ('warnings', 'gender_warnings'),
            ('dosage_and_administration', 'dose_adjustments'),
            ('adverse_reactions', 'adverse_events'),
            ('pharmacokinetics', 'pharmacokinetics'),
            ('pharmacodynamics', 'pharmacodynamics'),
            ('clinical_studies', 'clinical_studies'),
            ('risk_factors', 'risk_factors')
        ]:
            if section in drug_data:
                for item in drug_data[section]:
                    if 'gender' in str(item).lower():
                        gender_info[key].append(str(item))
        
        # Extract pregnancy and lactation information
        if 'pregnancy' in drug_data:
            gender_info['pregnancy_category'] = drug_data['pregnancy'].get('category')
        
        if 'lactation' in drug_data:
            gender_info['lactation_risk'] = drug_data['lactation'].get('risk')
        
        return gender_info

    def get_drug_interactions(self, drug_name: str) -> List[Dict[str, Any]]:
        """
        Get drug interactions for a specific drug
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            List of drug interactions
        """
        interactions = []
        drug_data = self.search_drug(drug_name)
        
        if drug_data:
            drug_info = drug_data[0]
            if 'drug_interactions' in drug_info:
                interactions = drug_info['drug_interactions']
        
        return interactions

    def create_drug_profile(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Create a comprehensive drug profile with gender-specific information.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Dict[str, Any]: Drug profile containing:
                - Basic drug information
                - Gender-specific information
                - Dosing guidelines
                - Drug interactions
                - Gender bias score
                - Metadata
            None: If drug information cannot be retrieved
        """
        try:
            drug_data = self.search_drug(drug_name)
            if not drug_data:
                return None
                
            drug_info = drug_data[0]
            gender_info = self.get_gender_specific_info(drug_info)
            
            # Analyze drug interactions
            interactions = self.get_drug_interactions(drug_name)
            
            # Analyze dosing guidelines
            dosing = self.get_drug_dosing_guidelines(drug_name)
            
            # Calculate gender bias score
            bias_score = self._calculate_gender_bias_score(drug_info)
            
            profile: Dict[str, Any] = {
                'drug_name': drug_name,
                'manufacturer': drug_info.get('manufacturer_name', 'Unknown'),
                'approval_date': drug_info.get('approval_date', 'Unknown'),
                'gender_specific_info': gender_info,
                'indications': drug_info.get('indications_and_usage', []),
                'contraindications': drug_info.get('contraindications', []),
                'drug_interactions': interactions,
                'dosing_guidelines': dosing,
                'gender_bias_score': bias_score,
                'last_updated': datetime.now().isoformat(),
                'metadata': {
                    'clinical_trials': drug_info.get('clinical_trials', []),
                    'pharmacology': drug_info.get('pharmacology', {}),
                    'mechanism_of_action': drug_info.get('mechanism_of_action', [])
                }
            }
            
            return profile
            
        except Exception as e:
            print(f"Error creating drug profile: {e}")
            return None

    def _calculate_gender_bias_score(self, drug_info: Dict[str, Any]) -> float:
        """
        Calculate a gender bias score for the drug based on available data
        
        Returns:
            float: Gender bias score (0-1) where 1 is highly biased
        """
        score = 0.0
        max_score = 5
        
        # Check for gender-specific warnings
        if any('gender' in w.lower() for w in drug_info.get('warnings', [])):
            score += 1
        
        # Check for gender-specific dose adjustments
        if any('gender' in d.lower() for d in drug_info.get('dosage_and_administration', [])):
            score += 1
        
        # Check for gender differences in pharmacokinetics
        if any('gender' in pk.lower() for pk in drug_info.get('pharmacokinetics', [])):
            score += 1
        
        # Check for gender differences in pharmacodynamics
        if any('gender' in pd.lower() for pd in drug_info.get('pharmacodynamics', [])):
            score += 1
        
        # Check for gender differences in clinical trials
        if any('gender' in s.lower() for s in drug_info.get('clinical_studies', [])):
            score += 1
        
        return round(score / max_score, 2)

    def get_drug_dosing_guidelines(self, drug_name: str) -> Dict[str, Any]:
        """
        Get dosing guidelines for a specific drug
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Dictionary containing dosing guidelines
        """
        drug_data = self.search_drug(drug_name)
        if not drug_data:
            return None
            
        drug_info = drug_data[0]
        
        dosing = {
            'standard_dose': drug_info.get('dosage_and_administration', []),
            'gender_adjustments': self.get_gender_specific_info(drug_info)['dose_adjustments'],
            'renal_adjustments': drug_info.get('renal_dosage', []),
            'hepatic_adjustments': drug_info.get('hepatic_dosage', [])
        }
        
        return dosing

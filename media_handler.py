import os
import json
import logging
import random
from datetime import datetime
from typing import List, Dict, Any, Optional  # ✅ Add List here

logger = logging.getLogger(__name__)

class MediaHandler:
    """
    Handler for media assets including:
    - Sound bites
    - White paper
    - Mock tweet feed
    """
    
    def __init__(self, media_dir: str = "media"):
        """Initialize media handler."""
        self.media_dir = media_dir
        self.sound_bites = [
            {
                "title": "NIH Director on Gender Bias",
                "url": "https://example.com/nih_gender_bias.mp3",
                "transcript": "In clinical trials, the underrepresentation of women has led to significant gaps in our understanding of drug response."
            },
            {
                "title": "FDA Expert Panel Discussion",
                "url": "https://example.com/fda_panel.mp3",
                "transcript": "Gender differences in pharmacokinetics and pharmacodynamics are critical considerations in drug development and dosing."
            }
        ]
        
        self.tweet_feed = [
            {
                "handle": "@FDACDER",
                "text": """Game-changing approach to post-market surveillance:
                Gender-biased drug response analysis now available through AI/ML.
                #GenderInAI #DrugSafety""",
                "likes": 250,
                "retweets": 120,
                "timestamp": "2025-08-08T00:00:00+03:00"
            },
            {
                "handle": "@NIH",
                "text": """Addressing gender bias in clinical trials is crucial for patient safety.
                New tools help identify and mitigate gender-specific drug risks.
                #ClinicalTrials #PatientSafety""",
                "likes": 180,
                "retweets": 95,
                "timestamp": "2025-08-08T00:05:00+03:00"
            },
            {
                "handle": "@DrugSafety",
                "text": """Innovative AI solution detects gender bias in drug response patterns.
                Essential for personalized medicine and patient safety.
                #AIinHealthcare #DrugSafety""",
                "likes": 150,
                "retweets": 75,
                "timestamp": "2025-08-08T00:10:00+03:00"
            }
        ]
        
        # Create media directory if it doesn't exist
        os.makedirs(self.media_dir, exist_ok=True)
        
    def get_sound_bites(self) -> List[Dict[str, Any]]:
        """Get available sound bites."""
        return self.sound_bites
        
    def get_white_paper(self) -> Dict[str, Any]:
        """Get white paper content."""
        return {
            "title": "Gender-Biased Drug Response Analysis",
            "subtitle": "Technical White Paper",
            "sections": [
                {
                    "title": "Introduction",
                    "content": """This white paper provides a technical overview of the Gender-Biased Drug Response Analysis system, including methodology, implementation details, and validation results."""
                },
                {
                    "title": "Methodology",
                    "content": """The system uses a combination of machine learning models, natural language processing, and pharmacokinetic modeling to analyze gender-specific drug response patterns."""
                },
                {
                    "title": "Validation",
                    "content": """The system has been validated against FDA Adverse Event Reporting System (FAERS) data and clinical trial datasets, demonstrating high accuracy in detecting gender bias."""
                }
            ]
        }
        
    def get_tweet_feed(self) -> List[Dict[str, Any]]:
        """Get mock tweet feed."""
        return self.tweet_feed
        
    def generate_live_feed_update(self) -> Dict[str, Any]:
        """Generate a new mock tweet."""
        handles = ["@FDACDER", "@NIH", "@DrugSafety", "@ClinicalTrials", "@AIinHealthcare"]
        topics = [
            "#GenderInAI",
            "#DrugSafety",
            "#ClinicalTrials",
            "#AIinHealthcare",
            "#PatientSafety"
        ]
        
        return {
            "handle": random.choice(handles),
            "text": f"""New insights into gender-biased drug response patterns.
            Essential for personalized medicine.
            {random.choice(topics)}""",
            "likes": random.randint(50, 200),
            "retweets": random.randint(25, 100),
            "timestamp": datetime.now().isoformat()
        }

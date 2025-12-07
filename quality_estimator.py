"""
Translation Quality Estimation using COMET-QE

Uses Unbabel's COMET-QE model for reference-free quality estimation.
Provides a single quality score (0-100) comparing source text to translation.
Works across multiple language pairs without needing human reference translations.
"""

from typing import Dict, Optional
import torch


class QualityEstimator:
    """Reference-free translation quality estimator using COMET-QE."""
    
    def __init__(self, model_name: str = "Unbabel/wmt22-cometkiwi-da"):
        """
        Initialize quality estimator with COMET-QE model.
        
        Args:
            model_name: COMET model identifier
                - "Unbabel/wmt22-cometkiwi-da" (recommended, latest QE model - requires HF login)
                - "Unbabel/wmt20-comet-qe-da" (older alternative, publicly available)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load COMET-QE model (downloads on first run)."""
        try:
            from comet import download_model, load_from_checkpoint
            
            print(f"Loading COMET-QE model '{self.model_name}'...")
            model_path = download_model(self.model_name)
            self.model = load_from_checkpoint(model_path)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                print("✓ Using GPU for quality estimation")
            else:
                print("✓ Using CPU for quality estimation (slower)")
            
            print("✓ COMET-QE model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "COMET library not found. Install with: pip install unbabel-comet"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load COMET-QE model: {str(e)}")
    
    def estimate_quality(self, source: str, translation: str) -> float:
        """
        Estimate translation quality without reference.
        
        Args:
            source: Source text (original language)
            translation: Translated text (target language)
            
        Returns:
            Quality score (0-100, higher is better)
        """
        if not source or not translation:
            return 0.0
        
        if self.model is None:
            raise RuntimeError("COMET-QE model not loaded")
        
        # Remove ALL line breaks from both inputs to eliminate formatting differences
        # This ensures COMET only evaluates translation quality, not formatting alignment
        source_cleaned = source.replace('\n', ' ').strip()
        translation_cleaned = translation.replace('\n', ' ').strip()
        
        # Prepare data for COMET (expects list of dicts)
        data = [{
            "src": source_cleaned,
            "mt": translation_cleaned
        }]
        
        # Get predictions (returns scores in 0-1 range)
        # Use GPU if available, otherwise CPU
        gpus = 1 if torch.cuda.is_available() else 0
        
        try:
            scores = self.model.predict(data, batch_size=8, gpus=gpus)
            # Convert to 0-100 scale
            return scores.scores[0] * 100
        except Exception as e:
            print(f"Warning: Quality estimation failed: {e}")
            return 0.0
    
    @staticmethod
    def interpret_score(score: float) -> Dict[str, str]:
        """
        Provide human-readable interpretation of quality score.
        
        Args:
            score: Quality score (0-100)
            
        Returns:
            Dictionary with quality level and description
        """
        if score >= 85:
            return {
                "level": "Excellent",
                "description": "Professional-grade translation with minimal issues",
                "color": "#2e7d32"  # Green
            }
        elif score >= 70:
            return {
                "level": "Good",
                "description": "High-quality translation with minor imperfections",
                "color": "#558b2f"  # Light green
            }
        elif score >= 55:
            return {
                "level": "Fair",
                "description": "Acceptable translation that may need review",
                "color": "#f9a825"  # Yellow
            }
        elif score >= 40:
            return {
                "level": "Poor",
                "description": "Low-quality translation requiring significant revision",
                "color": "#ef6c00"  # Orange
            }
        else:
            return {
                "level": "Very Poor",
                "description": "Inadequate translation with major quality issues",
                "color": "#c62828"  # Red
            }
    
    def evaluate_with_interpretation(self, source: str, translation: str) -> Dict:
        """
        Estimate quality and provide interpretation.
        
        Args:
            source: Source text (original language)
            translation: Translated text (target language)
            
        Returns:
            Dictionary with score and interpretation:
            {
                'score': 78.5,
                'level': 'Good',
                'description': 'High-quality translation...',
                'color': '#558b2f'
            }
        """
        score = self.estimate_quality(source, translation)
        interpretation = self.interpret_score(score)
        
        return {
            'score': score,
            **interpretation
        }

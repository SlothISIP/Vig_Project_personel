"""
Defect Explainer - XAI for Defect Detection with Root Cause Analysis.

This module provides comprehensive explanations for defect detection,
linking visual evidence to potential root causes in the manufacturing process.

Key Features:
1. Grad-CAM visualization of defect regions
2. Defect pattern classification (scratch, stain, crack, etc.)
3. Spatial analysis (location, size, shape)
4. Root cause hypothesis generation
5. Confidence-based filtering

Innovation: Bridges Vision AI results with Digital Twin for root cause inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage import measure

from .grad_cam import SwinGradCAM, GradCAM, CAMResult, overlay_heatmap
from src.core.logging import get_logger

logger = get_logger(__name__)


class DefectType(Enum):
    """Types of manufacturing defects."""

    SCRATCH = "scratch"
    STAIN = "stain"
    CRACK = "crack"
    DENT = "dent"
    CONTAMINATION = "contamination"
    MISALIGNMENT = "misalignment"
    DISCOLORATION = "discoloration"
    MISSING_COMPONENT = "missing_component"
    SURFACE_ROUGHNESS = "surface_roughness"
    UNKNOWN = "unknown"


class DefectSeverity(Enum):
    """Severity levels for defects."""

    MINOR = "minor"  # Cosmetic, acceptable
    MODERATE = "moderate"  # Should be reviewed
    MAJOR = "major"  # Requires rework
    CRITICAL = "critical"  # Scrap or major repair


class DefectLocation(Enum):
    """General location categories."""

    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"
    EDGE = "edge"
    CORNER = "corner"


@dataclass
class DefectRegion:
    """Information about a detected defect region."""

    region_id: int
    centroid: Tuple[float, float]  # (y, x) normalized 0-1
    bounding_box: Tuple[int, int, int, int]  # (y_min, x_min, y_max, x_max)
    area: float  # Normalized area (0-1)
    intensity: float  # Average CAM intensity in region
    shape_features: Dict[str, float]  # Eccentricity, solidity, etc.
    location: DefectLocation


@dataclass
class RootCauseHypothesis:
    """Hypothesis about the root cause of a defect."""

    hypothesis_id: str
    description: str
    probability: float  # 0-1 confidence
    related_station: Optional[str]  # Manufacturing station that might cause this
    related_process: Optional[str]  # Process step
    evidence: List[str]  # List of supporting evidence
    recommended_actions: List[str]


@dataclass
class ExplanationResult:
    """Complete explanation for a defect detection."""

    # Detection info
    is_defect: bool
    confidence: float
    defect_probability: float

    # Visual explanation
    heatmap: np.ndarray  # Grad-CAM heatmap
    overlay_image: Optional[np.ndarray]  # Heatmap overlaid on original

    # Defect analysis
    defect_type: DefectType
    defect_type_confidence: float
    severity: DefectSeverity
    regions: List[DefectRegion]

    # Spatial analysis
    primary_location: DefectLocation
    affected_area_percent: float

    # Root cause analysis
    root_cause_hypotheses: List[RootCauseHypothesis]

    # Metadata
    processing_time_ms: float
    model_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_defect": self.is_defect,
            "confidence": self.confidence,
            "defect_probability": self.defect_probability,
            "defect_type": self.defect_type.value,
            "defect_type_confidence": self.defect_type_confidence,
            "severity": self.severity.value,
            "primary_location": self.primary_location.value,
            "affected_area_percent": self.affected_area_percent,
            "num_regions": len(self.regions),
            "regions": [
                {
                    "id": r.region_id,
                    "centroid": r.centroid,
                    "area": r.area,
                    "intensity": r.intensity,
                    "location": r.location.value,
                }
                for r in self.regions
            ],
            "root_causes": [
                {
                    "id": h.hypothesis_id,
                    "description": h.description,
                    "probability": h.probability,
                    "station": h.related_station,
                    "process": h.related_process,
                    "actions": h.recommended_actions,
                }
                for h in self.root_cause_hypotheses
            ],
            "processing_time_ms": self.processing_time_ms,
        }


class DefectPatternClassifier:
    """
    Classifies defect patterns based on spatial and visual features.

    Uses heuristics and learned patterns to categorize defect types
    from the Grad-CAM heatmap and original image features.
    """

    # Pattern characteristics for each defect type
    PATTERN_PROFILES = {
        DefectType.SCRATCH: {
            "elongation_min": 0.7,  # Very elongated
            "aspect_ratio_min": 3.0,
            "edge_density_high": True,
        },
        DefectType.STAIN: {
            "circularity_min": 0.6,  # Relatively round
            "uniformity_min": 0.5,
            "edge_density_low": True,
        },
        DefectType.CRACK: {
            "elongation_min": 0.6,
            "branching": True,
            "edge_density_high": True,
        },
        DefectType.DENT: {
            "circularity_min": 0.5,
            "gradient_pattern": "concentric",
        },
        DefectType.CONTAMINATION: {
            "multiple_regions": True,
            "irregular_shape": True,
        },
        DefectType.MISALIGNMENT: {
            "edge_location": True,
            "linear_pattern": True,
        },
        DefectType.DISCOLORATION: {
            "large_area": True,
            "low_intensity_variance": True,
        },
    }

    def classify(
        self,
        heatmap: np.ndarray,
        regions: List[DefectRegion],
        image: Optional[np.ndarray] = None,
    ) -> Tuple[DefectType, float]:
        """
        Classify the defect pattern.

        Args:
            heatmap: Grad-CAM heatmap
            regions: Detected defect regions
            image: Original image (optional, for additional analysis)

        Returns:
            Tuple of (DefectType, confidence)
        """
        if not regions:
            return DefectType.UNKNOWN, 0.0

        # Extract features from primary region
        primary_region = max(regions, key=lambda r: r.intensity * r.area)
        features = self._extract_pattern_features(heatmap, primary_region, regions)

        # Score each defect type
        scores = {}
        for defect_type, profile in self.PATTERN_PROFILES.items():
            score = self._match_profile(features, profile)
            scores[defect_type] = score

        # Select best match
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            return best_type, confidence

        return DefectType.UNKNOWN, 0.0

    def _extract_pattern_features(
        self,
        heatmap: np.ndarray,
        primary_region: DefectRegion,
        all_regions: List[DefectRegion],
    ) -> Dict[str, float]:
        """Extract pattern features for classification."""
        features = {}

        # Shape features from primary region
        shape = primary_region.shape_features
        features["elongation"] = shape.get("eccentricity", 0.5)
        features["circularity"] = shape.get("circularity", 0.5)
        features["solidity"] = shape.get("solidity", 0.8)

        # Compute aspect ratio from bounding box
        bbox = primary_region.bounding_box
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        features["aspect_ratio"] = max(height, width) / (min(height, width) + 1e-6)

        # Area features
        features["area"] = primary_region.area
        features["num_regions"] = len(all_regions)

        # Intensity features
        features["intensity"] = primary_region.intensity
        features["intensity_variance"] = np.var(heatmap[heatmap > 0.1]) if np.any(heatmap > 0.1) else 0

        # Location features
        features["is_edge"] = primary_region.location in [
            DefectLocation.EDGE, DefectLocation.CORNER
        ]

        # Edge density (high values in heatmap gradient)
        gradient = np.gradient(heatmap)
        edge_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
        features["edge_density"] = np.mean(edge_magnitude[heatmap > 0.3])

        return features

    def _match_profile(
        self,
        features: Dict[str, float],
        profile: Dict[str, Any],
    ) -> float:
        """Score how well features match a defect type profile."""
        score = 0.0
        matches = 0

        if "elongation_min" in profile:
            if features.get("elongation", 0) >= profile["elongation_min"]:
                score += 0.3
            matches += 1

        if "aspect_ratio_min" in profile:
            if features.get("aspect_ratio", 1) >= profile["aspect_ratio_min"]:
                score += 0.3
            matches += 1

        if "circularity_min" in profile:
            if features.get("circularity", 0) >= profile["circularity_min"]:
                score += 0.3
            matches += 1

        if "edge_density_high" in profile:
            if features.get("edge_density", 0) > 0.1:
                score += 0.2
            matches += 1

        if "edge_density_low" in profile:
            if features.get("edge_density", 1) < 0.1:
                score += 0.2
            matches += 1

        if "multiple_regions" in profile:
            if features.get("num_regions", 1) > 2:
                score += 0.3
            matches += 1

        if "large_area" in profile:
            if features.get("area", 0) > 0.1:
                score += 0.3
            matches += 1

        if "edge_location" in profile:
            if features.get("is_edge", False):
                score += 0.3
            matches += 1

        # Normalize score
        if matches > 0:
            score = score / (matches * 0.3)

        return min(1.0, score)


class RootCauseAnalyzer:
    """
    Analyzes defect patterns to generate root cause hypotheses.

    Links defect characteristics to potential manufacturing issues
    based on domain knowledge encoded as rules.
    """

    # Root cause rules based on defect type and location
    CAUSE_RULES = {
        DefectType.SCRATCH: [
            {
                "condition": lambda f: f["is_linear"] and f["is_horizontal"],
                "cause": "Material handling issue - conveyor belt scratch",
                "station": "loading",
                "process": "material_transfer",
                "actions": ["Inspect conveyor belt", "Check material guides"],
            },
            {
                "condition": lambda f: f["is_linear"] and f["is_vertical"],
                "cause": "Tool contact during processing",
                "station": "assembly",
                "process": "machining",
                "actions": ["Check tool alignment", "Inspect tool wear"],
            },
        ],
        DefectType.STAIN: [
            {
                "condition": lambda f: f["area"] > 0.05,
                "cause": "Contaminated cleaning solution or lubricant",
                "station": "assembly",
                "process": "cleaning",
                "actions": ["Replace cleaning fluid", "Check filtration system"],
            },
            {
                "condition": lambda f: f["area"] <= 0.05,
                "cause": "Localized oil/grease contamination",
                "station": "assembly",
                "process": "lubrication",
                "actions": ["Check lubrication system", "Clean work area"],
            },
        ],
        DefectType.CRACK: [
            {
                "condition": lambda f: f["at_edge"],
                "cause": "Stress concentration at edge - material or process issue",
                "station": "assembly",
                "process": "forming",
                "actions": ["Check material specifications", "Review forming parameters"],
            },
            {
                "condition": lambda f: not f["at_edge"],
                "cause": "Internal stress or material defect",
                "station": None,
                "process": "material",
                "actions": ["Inspect incoming material", "Check storage conditions"],
            },
        ],
        DefectType.DENT: [
            {
                "condition": lambda f: True,
                "cause": "Impact damage during handling or assembly",
                "station": "loading",
                "process": "handling",
                "actions": ["Review handling procedures", "Check fixture alignment"],
            },
        ],
        DefectType.CONTAMINATION: [
            {
                "condition": lambda f: f["multiple_spots"],
                "cause": "Airborne particles or dust in environment",
                "station": None,
                "process": "environment",
                "actions": ["Check air filtration", "Increase cleaning frequency"],
            },
            {
                "condition": lambda f: not f["multiple_spots"],
                "cause": "Tool or fixture contamination",
                "station": "assembly",
                "process": "tooling",
                "actions": ["Clean tools and fixtures", "Inspect tool condition"],
            },
        ],
        DefectType.MISALIGNMENT: [
            {
                "condition": lambda f: True,
                "cause": "Fixture positioning error or component misplacement",
                "station": "assembly",
                "process": "positioning",
                "actions": ["Calibrate fixtures", "Check component feeding system"],
            },
        ],
        DefectType.DISCOLORATION: [
            {
                "condition": lambda f: True,
                "cause": "Temperature variation or chemical exposure",
                "station": "assembly",
                "process": "heat_treatment",
                "actions": ["Check temperature controls", "Verify process timing"],
            },
        ],
    }

    def analyze(
        self,
        defect_type: DefectType,
        regions: List[DefectRegion],
        heatmap: np.ndarray,
    ) -> List[RootCauseHypothesis]:
        """
        Generate root cause hypotheses based on defect analysis.

        Args:
            defect_type: Classified defect type
            regions: Detected defect regions
            heatmap: Grad-CAM heatmap

        Returns:
            List of root cause hypotheses ranked by probability
        """
        hypotheses = []

        if defect_type not in self.CAUSE_RULES:
            # Generic hypothesis for unknown types
            return [
                RootCauseHypothesis(
                    hypothesis_id="generic_001",
                    description="Unidentified manufacturing defect - requires manual inspection",
                    probability=0.5,
                    related_station=None,
                    related_process="unknown",
                    evidence=["Novel defect pattern detected"],
                    recommended_actions=["Manual quality inspection", "Document defect pattern"],
                )
            ]

        # Extract features for rule matching
        features = self._extract_cause_features(regions, heatmap)

        # Apply rules
        rules = self.CAUSE_RULES[defect_type]
        for i, rule in enumerate(rules):
            try:
                if rule["condition"](features):
                    # Calculate probability based on feature confidence
                    probability = self._calculate_cause_probability(features, rule)

                    hypothesis = RootCauseHypothesis(
                        hypothesis_id=f"{defect_type.value}_{i:03d}",
                        description=rule["cause"],
                        probability=probability,
                        related_station=rule.get("station"),
                        related_process=rule.get("process"),
                        evidence=self._generate_evidence(features, defect_type),
                        recommended_actions=rule.get("actions", []),
                    )
                    hypotheses.append(hypothesis)
            except Exception as e:
                logger.debug(f"Rule evaluation failed: {e}")
                continue

        # Sort by probability
        hypotheses.sort(key=lambda h: h.probability, reverse=True)

        return hypotheses[:3]  # Return top 3 hypotheses

    def _extract_cause_features(
        self,
        regions: List[DefectRegion],
        heatmap: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract features relevant for root cause analysis."""
        if not regions:
            return {}

        primary = max(regions, key=lambda r: r.area * r.intensity)

        # Orientation analysis
        bbox = primary.bounding_box
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]

        features = {
            "is_linear": primary.shape_features.get("eccentricity", 0) > 0.8,
            "is_horizontal": width > height * 2,
            "is_vertical": height > width * 2,
            "at_edge": primary.location in [
                DefectLocation.EDGE,
                DefectLocation.TOP_CENTER,
                DefectLocation.BOTTOM_CENTER,
                DefectLocation.CENTER_LEFT,
                DefectLocation.CENTER_RIGHT,
            ],
            "at_corner": primary.location == DefectLocation.CORNER,
            "area": primary.area,
            "multiple_spots": len(regions) > 2,
            "high_intensity": primary.intensity > 0.7,
            "centroid_y": primary.centroid[0],
            "centroid_x": primary.centroid[1],
        }

        return features

    def _calculate_cause_probability(
        self,
        features: Dict[str, Any],
        rule: Dict[str, Any],
    ) -> float:
        """Calculate probability score for a root cause hypothesis."""
        base_prob = 0.6  # Base probability for matching rule

        # Adjust based on feature strength
        if features.get("high_intensity", False):
            base_prob += 0.1

        if features.get("area", 0) > 0.05:
            base_prob += 0.1

        # Cap at 0.95
        return min(0.95, base_prob)

    def _generate_evidence(
        self,
        features: Dict[str, Any],
        defect_type: DefectType,
    ) -> List[str]:
        """Generate evidence statements for the hypothesis."""
        evidence = []

        evidence.append(f"Defect classified as {defect_type.value}")

        if features.get("is_linear"):
            evidence.append("Linear defect pattern detected")
        if features.get("at_edge"):
            evidence.append("Defect located at edge region")
        if features.get("multiple_spots"):
            evidence.append("Multiple defect regions detected")
        if features.get("high_intensity"):
            evidence.append("High-confidence defect region")

        return evidence


class DefectExplainer:
    """
    Main interface for defect explanation with XAI.

    Combines Grad-CAM visualization, pattern classification,
    and root cause analysis into a unified pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str] = None,
        defect_threshold: float = 0.5,
        use_cuda: bool = True,
    ):
        """
        Initialize DefectExplainer.

        Args:
            model: Trained defect detection model
            class_names: Names for each class (default: ["good", "defect"])
            defect_threshold: Threshold for defect classification
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.class_names = class_names or ["good", "defect"]
        self.defect_threshold = defect_threshold
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        # Initialize components
        self.grad_cam = SwinGradCAM(model, use_cuda=use_cuda)
        self.pattern_classifier = DefectPatternClassifier()
        self.root_cause_analyzer = RootCauseAnalyzer()

        logger.info("DefectExplainer initialized")

    def explain(
        self,
        image: torch.Tensor,
        original_image: Optional[np.ndarray] = None,
        return_overlay: bool = True,
    ) -> ExplanationResult:
        """
        Generate comprehensive explanation for an image.

        Args:
            image: Preprocessed image tensor
            original_image: Original image for overlay (optional)
            return_overlay: Whether to generate overlay visualization

        Returns:
            ExplanationResult with full analysis
        """
        import time
        start_time = time.time()

        # Ensure correct tensor format
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Get Grad-CAM result
        cam_result = self.grad_cam(image, target_class=1)  # Target defect class

        # Determine if defect
        defect_prob = cam_result.class_scores.get(1, 0.0)
        is_defect = defect_prob >= self.defect_threshold

        # Analyze regions if defect detected
        if is_defect:
            regions = self._extract_regions(cam_result.heatmap)
            defect_type, type_confidence = self.pattern_classifier.classify(
                cam_result.heatmap, regions
            )
            root_causes = self.root_cause_analyzer.analyze(
                defect_type, regions, cam_result.heatmap
            )
        else:
            regions = []
            defect_type = DefectType.UNKNOWN
            type_confidence = 0.0
            root_causes = []

        # Determine severity based on area and confidence
        severity = self._determine_severity(regions, defect_prob)

        # Primary location
        primary_location = self._get_primary_location(regions, cam_result.heatmap)

        # Affected area
        affected_area = np.mean(cam_result.heatmap > 0.3)

        # Generate overlay if requested
        overlay = None
        if return_overlay and original_image is not None:
            overlay = overlay_heatmap(original_image, cam_result.heatmap)

        processing_time = (time.time() - start_time) * 1000

        return ExplanationResult(
            is_defect=is_defect,
            confidence=cam_result.confidence,
            defect_probability=defect_prob,
            heatmap=cam_result.heatmap,
            overlay_image=overlay,
            defect_type=defect_type,
            defect_type_confidence=type_confidence,
            severity=severity,
            regions=regions,
            primary_location=primary_location,
            affected_area_percent=affected_area * 100,
            root_cause_hypotheses=root_causes,
            processing_time_ms=processing_time,
        )

    def _extract_regions(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.3,
        min_area: int = 50,
    ) -> List[DefectRegion]:
        """Extract defect regions from heatmap."""
        # Threshold heatmap
        binary = (heatmap > threshold).astype(np.uint8)

        # Label connected components
        labeled, num_features = ndimage.label(binary)

        regions = []
        for region_id in range(1, num_features + 1):
            mask = labeled == region_id

            # Skip small regions
            area_pixels = np.sum(mask)
            if area_pixels < min_area:
                continue

            # Get region properties
            props = measure.regionprops(mask.astype(int))[0]

            # Normalized coordinates
            h, w = heatmap.shape
            centroid = (props.centroid[0] / h, props.centroid[1] / w)
            bbox = props.bbox  # (min_row, min_col, max_row, max_col)
            area = area_pixels / (h * w)

            # Shape features
            shape_features = {
                "eccentricity": props.eccentricity if hasattr(props, 'eccentricity') else 0.5,
                "solidity": props.solidity if hasattr(props, 'solidity') else 0.8,
                "circularity": 4 * np.pi * props.area / (props.perimeter ** 2 + 1e-6)
                if hasattr(props, 'perimeter') else 0.5,
            }

            # Location classification
            location = self._classify_location(centroid)

            # Average intensity in region
            intensity = np.mean(heatmap[mask])

            regions.append(DefectRegion(
                region_id=region_id,
                centroid=centroid,
                bounding_box=bbox,
                area=area,
                intensity=intensity,
                shape_features=shape_features,
                location=location,
            ))

        # Sort by intensity * area
        regions.sort(key=lambda r: r.intensity * r.area, reverse=True)

        return regions

    def _classify_location(self, centroid: Tuple[float, float]) -> DefectLocation:
        """Classify location based on normalized centroid."""
        y, x = centroid

        # Check edges first
        if y < 0.1 or y > 0.9 or x < 0.1 or x > 0.9:
            if (y < 0.1 or y > 0.9) and (x < 0.1 or x > 0.9):
                return DefectLocation.CORNER
            return DefectLocation.EDGE

        # 3x3 grid classification
        if y < 0.33:
            if x < 0.33:
                return DefectLocation.TOP_LEFT
            elif x > 0.66:
                return DefectLocation.TOP_RIGHT
            else:
                return DefectLocation.TOP_CENTER
        elif y > 0.66:
            if x < 0.33:
                return DefectLocation.BOTTOM_LEFT
            elif x > 0.66:
                return DefectLocation.BOTTOM_RIGHT
            else:
                return DefectLocation.BOTTOM_CENTER
        else:
            if x < 0.33:
                return DefectLocation.CENTER_LEFT
            elif x > 0.66:
                return DefectLocation.CENTER_RIGHT
            else:
                return DefectLocation.CENTER

    def _determine_severity(
        self,
        regions: List[DefectRegion],
        defect_prob: float,
    ) -> DefectSeverity:
        """Determine defect severity based on analysis."""
        if not regions or defect_prob < self.defect_threshold:
            return DefectSeverity.MINOR

        # Total affected area
        total_area = sum(r.area for r in regions)
        max_intensity = max(r.intensity for r in regions) if regions else 0

        # Severity logic
        if total_area > 0.2 or max_intensity > 0.9:
            return DefectSeverity.CRITICAL
        elif total_area > 0.1 or max_intensity > 0.7:
            return DefectSeverity.MAJOR
        elif total_area > 0.03 or max_intensity > 0.5:
            return DefectSeverity.MODERATE
        else:
            return DefectSeverity.MINOR

    def _get_primary_location(
        self,
        regions: List[DefectRegion],
        heatmap: np.ndarray,
    ) -> DefectLocation:
        """Get primary defect location."""
        if regions:
            # Use primary region's location
            primary = max(regions, key=lambda r: r.area * r.intensity)
            return primary.location
        else:
            # Use heatmap maximum
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            h, w = heatmap.shape
            centroid = (max_idx[0] / h, max_idx[1] / w)
            return self._classify_location(centroid)

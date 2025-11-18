"""ONNX inference engine for fast inference."""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time
import numpy as np
import cv2
import onnxruntime as ort

from src.core.constants import IMAGENET_MEAN, IMAGENET_STD
from src.core.logging import get_logger
from src.core.exceptions import ModelLoadError, ModelInferenceError

logger = get_logger(__name__)


class ONNXInferenceEngine:
    """
    Fast inference using ONNX Runtime.

    Provides optimized inference with support for:
    - FP32, FP16 precision
    - CPU and CUDA execution providers
    - Batch and single image inference
    """

    def __init__(
        self,
        model_path: Path,
        providers: Optional[list[str]] = None,
        num_classes: int = 2,
        image_size: int = 224,
    ):
        """
        Initialize ONNX inference engine.

        Args:
            model_path: Path to ONNX model
            providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            num_classes: Number of output classes
            image_size: Input image size
        """
        self.model_path = Path(model_path)
        self.num_classes = num_classes
        self.image_size = image_size

        if not self.model_path.exists():
            raise ModelLoadError(f"Model not found: {self.model_path}")

        # Default providers
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Log settings
        logger.info(f"Loading ONNX model from: {model_path}")
        logger.info(f"Providers: {providers}")

        try:
            # Create inference session
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers,
            )

            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            # Get actual provider used
            actual_provider = self.session.get_providers()[0]
            logger.info(f"✓ Model loaded successfully using {actual_provider}")

            # Warmup
            self._warmup()

        except Exception as e:
            raise ModelLoadError(f"Failed to load ONNX model: {e}")

    def _warmup(self, num_iterations: int = 5):
        """Warmup the model with dummy inputs."""
        logger.info("Warming up model...")

        dummy_input = np.random.randn(1, 3, self.image_size, self.image_size).astype(
            np.float32
        )

        for _ in range(num_iterations):
            self.session.run([self.output_name], {self.input_name: dummy_input})

        logger.info("✓ Warmup complete")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: Input image (H, W, C) in RGB format, uint8

        Returns:
            Preprocessed image (1, C, H, W) in float32
        """
        # Resize
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))

        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        image = (image - mean) / std

        # HWC to CHW
        image = image.transpose(2, 0, 1)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def postprocess(
        self, outputs: np.ndarray
    ) -> Tuple[int, float, np.ndarray]:
        """
        Postprocess model outputs.

        Args:
            outputs: Model outputs (logits)

        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        # Softmax
        exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

        # Get prediction
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0, predicted_class]

        return int(predicted_class), float(confidence), probabilities[0]

    def predict(
        self,
        image: np.ndarray,
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on single image.

        Args:
            image: Input image (H, W, C) in RGB format, uint8
            return_probabilities: Whether to return class probabilities

        Returns:
            Dictionary containing:
                - predicted_class: int
                - confidence: float
                - defect_type: str ('good' or 'defect')
                - inference_time_ms: float
                - probabilities: Optional[np.ndarray]
        """
        start_time = time.time()

        try:
            # Preprocess
            input_tensor = self.preprocess(image)

            # Inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor},
            )[0]

            # Postprocess
            predicted_class, confidence, probabilities = self.postprocess(outputs)

            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # ms

            # Build result
            result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "defect_type": "defect" if predicted_class == 1 else "good",
                "inference_time_ms": inference_time,
            }

            if return_probabilities:
                result["probabilities"] = probabilities

            return result

        except Exception as e:
            raise ModelInferenceError(f"Inference failed: {e}")

    def predict_batch(
        self,
        images: list[np.ndarray],
        return_probabilities: bool = False,
    ) -> list[Dict[str, Any]]:
        """
        Run inference on batch of images.

        Args:
            images: List of images (H, W, C) in RGB format, uint8
            return_probabilities: Whether to return probabilities

        Returns:
            List of prediction dictionaries
        """
        start_time = time.time()

        try:
            # Preprocess all images
            input_tensors = [self.preprocess(img) for img in images]
            batch_input = np.concatenate(input_tensors, axis=0)

            # Batch inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: batch_input},
            )[0]

            # Postprocess each output
            results = []
            for i in range(len(images)):
                predicted_class, confidence, probabilities = self.postprocess(
                    outputs[i : i + 1]
                )

                result = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "defect_type": "defect" if predicted_class == 1 else "good",
                }

                if return_probabilities:
                    result["probabilities"] = probabilities

                results.append(result)

            # Calculate average inference time
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(images)

            # Add timing info to each result
            for result in results:
                result["inference_time_ms"] = avg_time

            return results

        except Exception as e:
            raise ModelInferenceError(f"Batch inference failed: {e}")

    def benchmark(
        self,
        num_iterations: int = 100,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            num_iterations: Number of iterations
            batch_size: Batch size

        Returns:
            Performance metrics
        """
        logger.info(f"Benchmarking with {num_iterations} iterations, batch_size={batch_size}")

        # Create dummy input
        dummy_input = np.random.randn(
            batch_size, 3, self.image_size, self.image_size
        ).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: dummy_input})

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            self.session.run([self.output_name], {self.input_name: dummy_input})
            elapsed = (time.time() - start_time) * 1000  # ms
            times.append(elapsed)

        times = np.array(times)

        metrics = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "fps": float(1000 / np.mean(times) * batch_size),
        }

        logger.info(f"Mean latency: {metrics['mean_ms']:.2f}ms")
        logger.info(f"P95 latency: {metrics['p95_ms']:.2f}ms")
        logger.info(f"Throughput: {metrics['fps']:.1f} FPS")

        return metrics

#!/usr/bin/env python3
"""Export PyTorch model to ONNX format."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import onnx

from src.vision.models.swin_transformer import create_swin_tiny
from src.core.constants import CHECKPOINTS_DIR, ONNX_DIR
from src.core.logging import setup_logging, get_logger

setup_logging(level="INFO", log_format="text")
logger = get_logger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 14,
    simplify: bool = True,
):
    """
    Export PyTorch model to ONNX.

    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        input_shape: Input tensor shape
        opset_version: ONNX opset version
        simplify: Whether to simplify the model
    """
    logger.info(f"Exporting model to ONNX: {output_path}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Opset version: {opset_version}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Export
    logger.info("Exporting...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info("✓ Export complete")

    # Verify ONNX model
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("✓ ONNX model is valid")

    # Simplify (optional, requires onnx-simplifier)
    if simplify:
        try:
            import onnxsim

            logger.info("Simplifying ONNX model...")
            model_simplified, check = onnxsim.simplify(onnx_model)

            if check:
                simplified_path = output_path.parent / f"{output_path.stem}_simplified.onnx"
                onnx.save(model_simplified, str(simplified_path))
                logger.info(f"✓ Simplified model saved to: {simplified_path}")
            else:
                logger.warning("Simplification failed validation")

        except ImportError:
            logger.warning("onnx-simplifier not installed, skipping simplification")
            logger.info("Install with: pip install onnx-simplifier")

    # Print model info
    logger.info(f"\nModel info:")
    logger.info(f"  Input: {onnx_model.graph.input[0].name}")
    logger.info(f"  Output: {onnx_model.graph.output[0].name}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def benchmark_onnx(onnx_path: Path):
    """Benchmark ONNX model."""
    logger.info(f"\nBenchmarking ONNX model...")

    from src.vision.inference.onnx_infer import ONNXInferenceEngine

    # Create engine
    engine = ONNXInferenceEngine(
        model_path=onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Benchmark
    metrics = engine.benchmark(num_iterations=100, batch_size=1)

    logger.info("\nBenchmark results:")
    logger.info(f"  Mean latency: {metrics['mean_ms']:.2f}ms")
    logger.info(f"  P50 latency: {metrics['p50_ms']:.2f}ms")
    logger.info(f"  P95 latency: {metrics['p95_ms']:.2f}ms")
    logger.info(f"  P99 latency: {metrics['p99_ms']:.2f}ms")
    logger.info(f"  Throughput: {metrics['fps']:.1f} FPS")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "best_model.pth",
        help="PyTorch checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ONNX_DIR / "swin_defect.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Don't simplify ONNX model",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark ONNX model after export",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PYTORCH TO ONNX EXPORT")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)

    # Check checkpoint exists
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.error("Train a model first with: python scripts/train_baseline.py")
        return 1

    # Load model
    logger.info("\nLoading PyTorch model...")
    model = create_swin_tiny(num_classes=2, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info("✓ Model loaded")

    # Export to ONNX
    export_to_onnx(
        model=model,
        output_path=args.output,
        input_shape=(1, 3, args.image_size, args.image_size),
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )

    logger.info(f"\n✓ Export complete: {args.output}")

    # Benchmark if requested
    if args.benchmark:
        benchmark_onnx(args.output)

    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 80)
    logger.info("Next steps:")
    logger.info(f"  1. Test inference: python -c 'from src.vision.inference.onnx_infer import ONNXInferenceEngine; e = ONNXInferenceEngine(\"{args.output}\"); print(e)'")
    logger.info("  2. Run API server with ONNX model")

    return 0


if __name__ == "__main__":
    sys.exit(main())

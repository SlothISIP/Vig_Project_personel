#!/usr/bin/env python3
"""Download and verify MVTec AD dataset."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import urllib.request
import tarfile
from tqdm import tqdm

from src.core.constants import RAW_DATA_DIR
from src.core.logging import setup_logging, get_logger

setup_logging(level="INFO", log_format="text")
logger = get_logger(__name__)


# MVTec AD categories
MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

# Base URL (official MVTec server)
BASE_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download"


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path):
    """Download file with progress bar."""
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=output_path.name
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_category(category: str, data_dir: Path) -> bool:
    """
    Download single category.

    Args:
        category: Category name
        data_dir: Output directory

    Returns:
        True if successful
    """
    category_dir = data_dir / category

    # Skip if already exists
    if category_dir.exists():
        logger.info(f"✓ {category} already exists, skipping")
        return True

    logger.info(f"Downloading {category}...")

    # Create URLs (MVTec provides tar.xz files)
    # Note: You may need to update these URLs based on actual MVTec server
    url = f"{BASE_URL}/{category}.tar.xz"
    output_file = data_dir / f"{category}.tar.xz"

    try:
        # Download
        download_url(url, output_file)

        # Extract
        logger.info(f"Extracting {category}...")
        with tarfile.open(output_file, "r:xz") as tar:
            tar.extractall(data_dir)

        # Remove archive
        output_file.unlink()

        logger.info(f"✓ {category} downloaded and extracted")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download {category}: {e}")
        if output_file.exists():
            output_file.unlink()
        return False


def verify_dataset(data_dir: Path) -> dict:
    """
    Verify dataset structure and count samples.

    Args:
        data_dir: Dataset directory

    Returns:
        Dictionary with verification results
    """
    logger.info("\nVerifying dataset...")

    results = {}

    for category in MVTEC_CATEGORIES:
        category_dir = data_dir / category

        if not category_dir.exists():
            results[category] = {"status": "missing"}
            continue

        # Check structure
        train_good = category_dir / "train" / "good"
        test_dir = category_dir / "test"

        if not train_good.exists() or not test_dir.exists():
            results[category] = {"status": "invalid_structure"}
            continue

        # Count samples
        train_count = len(list(train_good.glob("*.png")))

        test_good = test_dir / "good"
        test_good_count = len(list(test_good.glob("*.png"))) if test_good.exists() else 0

        # Count defect types
        defect_types = [
            d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good"
        ]
        defect_count = sum(len(list((test_dir / dt).glob("*.png"))) for dt in defect_types)

        results[category] = {
            "status": "ok",
            "train_good": train_count,
            "test_good": test_good_count,
            "test_defect": defect_count,
            "defect_types": len(defect_types),
        }

    return results


def print_verification_results(results: dict):
    """Print verification results in table format."""
    logger.info("\n" + "=" * 80)
    logger.info("DATASET VERIFICATION RESULTS")
    logger.info("=" * 80)

    # Header
    logger.info(
        f"{'Category':<15} {'Status':<12} {'Train':<8} {'Test Good':<12} {'Test Defect':<12} {'Defect Types':<12}"
    )
    logger.info("-" * 80)

    total_train = 0
    total_test_good = 0
    total_test_defect = 0
    ok_count = 0

    for category, data in sorted(results.items()):
        status = data["status"]

        if status == "ok":
            logger.info(
                f"{category:<15} {'✓ OK':<12} "
                f"{data['train_good']:<8} "
                f"{data['test_good']:<12} "
                f"{data['test_defect']:<12} "
                f"{data['defect_types']:<12}"
            )
            total_train += data["train_good"]
            total_test_good += data["test_good"]
            total_test_defect += data["test_defect"]
            ok_count += 1
        elif status == "missing":
            logger.info(f"{category:<15} {'✗ MISSING':<12}")
        else:
            logger.info(f"{category:<15} {'✗ INVALID':<12}")

    logger.info("-" * 80)
    logger.info(
        f"{'TOTAL':<15} {f'{ok_count}/15':<12} "
        f"{total_train:<8} "
        f"{total_test_good:<12} "
        f"{total_test_defect:<12}"
    )
    logger.info("=" * 80)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Download MVTec AD dataset")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Categories to download (default: all)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset, don't download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR / "mvtec_ad",
        help="Output directory",
    )

    args = parser.parse_args()

    # Create output directory
    data_dir = args.output_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MVTec AD DATASET DOWNLOADER")
    logger.info("=" * 80)
    logger.info(f"Output directory: {data_dir}")

    # Select categories
    categories = args.categories if args.categories else MVTEC_CATEGORIES
    logger.info(f"Categories: {len(categories)}")

    # Download or verify
    if args.verify_only:
        logger.info("\nVerification mode (no downloads)")
    else:
        logger.info("\n⚠️  NOTE: MVTec AD dataset requires manual download")
        logger.info("Please visit: https://www.mvtec.com/company/research/datasets/mvtec-ad")
        logger.info("Download the categories you need and extract to:")
        logger.info(f"  {data_dir}")
        logger.info("\nExpected structure:")
        logger.info("  mvtec_ad/")
        logger.info("    bottle/")
        logger.info("      train/good/*.png")
        logger.info("      test/good/*.png")
        logger.info("      test/broken_large/*.png")
        logger.info("      ...")
        logger.info("\nAfter downloading, run with --verify-only to check")

        # For automated download (if URLs are available)
        user_input = input("\nAttempt automated download? (experimental) [y/N]: ")
        if user_input.lower() == "y":
            logger.info("\nStarting downloads...")
            success_count = 0
            for category in categories:
                if download_category(category, data_dir):
                    success_count += 1

            logger.info(
                f"\nDownload complete: {success_count}/{len(categories)} categories"
            )

    # Verify dataset
    results = verify_dataset(data_dir)
    print_verification_results(results)

    # Summary
    ok_count = sum(1 for r in results.values() if r["status"] == "ok")
    if ok_count == len(MVTEC_CATEGORIES):
        logger.info("\n✓ All categories verified successfully!")
        logger.info("You can now run: python scripts/train_baseline.py")
        return 0
    elif ok_count > 0:
        logger.warning(f"\n⚠️  Only {ok_count}/15 categories available")
        logger.info("You can still train on available categories")
        return 0
    else:
        logger.error("\n✗ No valid categories found")
        logger.error("Please download the dataset manually")
        return 1


if __name__ == "__main__":
    sys.exit(main())

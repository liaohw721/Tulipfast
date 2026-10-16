"""Profile pretrained ResNet-18 feature-extraction latency on one image."""

import argparse
import statistics
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure CPU latency for pretrained ResNet-18 feature extraction."
    )
    parser.add_argument("--image", default="test_image.jpg", help="Image path; relative paths use the project folder.")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations (default: 10).")
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations (default: 100).")
    parser.add_argument("--num-threads", type=int, default=2, help="PyTorch CPU threads (default: 2).")
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup cannot be negative.")
    if args.iterations < 2:
        parser.error("--iterations must be at least 2 to report standard deviation.")
    if args.num_threads < 1:
        parser.error("--num-threads must be at least 1.")
    return args


def main():
    args = parse_args()
    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    torch.set_num_threads(args.num_threads)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = weights.transforms()(image).unsqueeze(0)

    timings = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(input_tensor)

        for _ in range(args.iterations):
            start = time.perf_counter()
            features = model(input_tensor)
            timings.append((time.perf_counter() - start) * 1000)

    print("Model: Pretrained ResNet-18 (classification head replaced with Identity)")
    print("Device: CPU")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"Image: {image_path}")
    print(f"CPU threads: {args.num_threads}")
    print(f"Warm-up iterations: {args.warmup}")
    print(f"Timed iterations: {args.iterations}")
    print(f"Feature shape: {tuple(features.shape)}")
    print(f"Median latency: {statistics.median(timings):.3f} ms")
    print(f"Mean latency: {statistics.mean(timings):.3f} ms")
    print(f"Latency standard deviation: {statistics.stdev(timings):.3f} ms")
    print("Note: this measures one image's encoder latency; it does not benchmark caching.")


if __name__ == "__main__":
    main()

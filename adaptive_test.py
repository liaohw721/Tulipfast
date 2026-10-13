
import time
import statistics
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# ===================================
# tulipfast Step 14: Fair Benchmark
# ===================================

torch.manual_seed(42)
torch.set_num_threads(2)

NUM_FRAMES = 30
NUM_TRIALS = 5
REFRESH_INTERVAL = 10
THRESHOLD = 0.02

# 1. Load pretrained vision encoder
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()

# 2. Load a real photo
image = Image.open("test_image.jpg").convert("RGB")
base = weights.transforms()(image)

# 3. Create a sequence of changing frames
frames = []

for i in range(NUM_FRAMES):
    noise = torch.randn_like(base) * (i / NUM_FRAMES) * 0.1
    frame = (base + noise).unsqueeze(0)
    frames.append(frame)

STRATEGIES = [
    "Always Refresh",
    "Fixed Rate",
    "Adaptive Cache"
]

# 4. Compute reference features OUTSIDE timing
print("Computing reference features...")

with torch.inference_mode():
    reference_features = [
        model(frame) for frame in frames
    ]

# 5. Execute one caching strategy
def execute_strategy(strategy):
    last_frame = None
    cached_features = None
    refreshes = 0
    outputs = []

    # Only time actual strategy operations
    start = time.perf_counter()

    with torch.inference_mode():
        for i, frame in enumerate(frames):

            if strategy == "Always Refresh":
                refresh = True

            elif cached_features is None:
                refresh = True

            elif strategy == "Fixed Rate":
                refresh = i % REFRESH_INTERVAL == 0

            elif strategy == "Adaptive Cache":
                difference = torch.mean(
                    torch.abs(frame - last_frame)
                ).item()

                refresh = difference > THRESHOLD

            else:
                raise ValueError("Unknown strategy")

            if refresh:
                cached_features = model(frame)
                last_frame = frame
                refreshes += 1

            outputs.append(cached_features)

    total_ms = (time.perf_counter() - start) * 1000

    # Correctness measurement OUTSIDE timing
    frame_errors = []
    max_errors = []

    with torch.inference_mode():
        for output, reference in zip(
            outputs, reference_features
        ):
            diff = torch.abs(output - reference)

            frame_errors.append(diff.mean().item())
            max_errors.append(diff.max().item())

    return {
        "Strategy": strategy,
        "Total_ms": total_ms,
        "Per_Frame_ms": total_ms / NUM_FRAMES,
        "Refreshes": refreshes,
        "Cache_Hits": NUM_FRAMES - refreshes,
        "Hit_Rate": 1 - refreshes / NUM_FRAMES,
        "Mean_Feature_Error": statistics.mean(frame_errors),
        "Max_Feature_Error": max(max_errors)
    }


# 6. Warm up CPU and model
print("Warming up model...")

with torch.inference_mode():
    for _ in range(20):
        model(frames[0])

# 7. Multiple trials with rotated execution order
results = []

print("Running fair benchmark...")

for trial in range(NUM_TRIALS):
    print(f"Trial {trial + 1}/{NUM_TRIALS}")

    # Rotate strategy order across trials
    shift = trial % len(STRATEGIES)
    order = STRATEGIES[shift:] + STRATEGIES[:shift]

    for strategy in order:
        result = execute_strategy(strategy)
        result["Trial"] = trial + 1
        results.append(result)

df = pd.DataFrame(results)

# 8. Calculate statistics across trials
summary = (
    df.groupby("Strategy")
    .agg(
        Median_Total_ms=("Total_ms", "median"),
        Mean_Total_ms=("Total_ms", "mean"),
        Std_Total_ms=("Total_ms", "std"),
        Median_Per_Frame_ms=("Per_Frame_ms", "median"),
        Refreshes=("Refreshes", "median"),
        Hit_Rate=("Hit_Rate", "mean"),
        Mean_Feature_Error=("Mean_Feature_Error", "mean"),
        Max_Feature_Error=("Max_Feature_Error", "max")
    )
    .reset_index()
)

baseline_ms = summary.loc[
    summary["Strategy"] == "Always Refresh",
    "Median_Total_ms"
].iloc[0]

summary["Speedup_vs_Baseline"] = (
    baseline_ms / summary["Median_Total_ms"]
)

# 9. Display results
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

print("\n===== tulipfast BENCHMARK RESULTS =====")
print(summary.round(5).to_string(index=False))

print("\nExperimental settings:")
print("Model: Pretrained ResNet-18")
print("Device: CPU")
print("Frames:", NUM_FRAMES)
print("Trials:", NUM_TRIALS)
print("Adaptive threshold:", THRESHOLD)
print("Fixed refresh interval:", REFRESH_INTERVAL)

# 10. Save all experiment results
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

df.to_csv(
    results_dir / f"fair_raw_{timestamp}.csv",
    index=False
)

summary.to_csv(
    results_dir / f"fair_summary_{timestamp}.csv",
    index=False
)

print("\nCSV files saved to results/")
print("Benchmark complete!")

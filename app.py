
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import time
import statistics
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="tulipfast", layout="wide")
st.title("🌷 TulipFast")
st.subheader("Synthetic VLA-Style Inference Optimization")
st.caption("PyTorch | CPU | Fixed synthetic inputs | Vision-feature reuse")

torch.set_num_threads(2)


# Synthetic VLA-style model
class MiniVLA(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU()
        )

        self.language = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.backbone = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.action_head = nn.Linear(256, 7)

    def encode_vision(self, image):
        return self.vision(image)

    def predict_action(self, visual_features, instruction):
        language_features = self.language(instruction)

        combined = torch.cat(
            [visual_features, language_features], dim=1
        )

        features = self.backbone(combined)
        return self.action_head(features)

    def forward(self, image, instruction):
        visual_features = self.encode_vision(image)
        return self.predict_action(visual_features, instruction)


@st.cache_resource
def load_model():
    torch.manual_seed(42)
    return MiniVLA().eval()


model = load_model()

# Fixed, synthetic inputs
torch.manual_seed(123)
image = torch.randn(1, 3, 32, 32)
instruction = torch.randn(1, 128)

iterations = st.slider(
    "Iterations per trial", 100, 1000, 300, 100
)

trials = st.slider(
    "Number of trials", 4, 20, 10
)


def measure_latency(operation, repeats):
    samples = []

    for _ in range(30):
        operation()

    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(
            (time.perf_counter() - start) * 1000
        )

    return statistics.median(samples)


if st.button("🚀 Run Optimization Experiment", type="primary"):
    results = []

    with st.spinner("Running optimization benchmark..."):
        with torch.inference_mode():

            def baseline():
                return model(image, instruction)

            # Cache is valid for this fixed input image
            cached_features = model.encode_vision(image)

            def optimized():
                return model.predict_action(
                    cached_features, instruction
                )

            baseline_output = baseline()
            optimized_output = optimized()

            outputs_match = torch.allclose(
                baseline_output, optimized_output,
                atol=1e-6, rtol=1e-5
            )

            for trial in range(trials):
                modes = [
                    ("Baseline", baseline),
                    ("Cached Vision", optimized)
                ]

                # Reverse order every other trial
                if trial % 2 == 1:
                    modes.reverse()

                for name, operation in modes:
                    latency = measure_latency(
                        operation, iterations
                    )

                    results.append({
                        "Trial": trial + 1,
                        "Mode": name,
                        "Median_Latency_ms": latency
                    })

    df = pd.DataFrame(results)

    summary = (
        df.groupby("Mode")["Median_Latency_ms"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    baseline_ms = summary.loc[
        summary["Mode"] == "Baseline", "median"
    ].iloc[0]

    cached_ms = summary.loc[
        summary["Mode"] == "Cached Vision", "median"
    ].iloc[0]

    speedup = baseline_ms / cached_ms

    st.success("Experiment completed!")

    col1, col2, col3 = st.columns(3)

    col1.metric("Baseline", f"{baseline_ms:.4f} ms")
    col2.metric("Cached Vision", f"{cached_ms:.4f} ms")
    col3.metric("Speedup (median latency)", f"{speedup:.2f}x")

    st.subheader("Before vs After")
    st.bar_chart(summary.set_index("Mode")[["median"]])

    st.subheader("Results")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Output Verification")

    if outputs_match:
        st.success("Outputs match within numerical tolerance.")
    else:
        st.error("Outputs do not match!")

    st.info(
        "The same image is intentionally reused across "
        "all iterations. The cached method skips vision "
        "encoding. Cache lookup and invalidation overhead "
        "are not included in this controlled experiment."
    )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df.to_csv(
        results_dir / f"optimization_raw_{timestamp}.csv",
        index=False
    )

    summary.to_csv(
        results_dir / f"optimization_summary_{timestamp}.csv",
        index=False
    )

    st.success("CSV results saved!")

    st.warning(
        "This is a synthetic VLA-style model with random "
        "weights, not a pretrained robot policy. The "
        "benchmark demonstrates repeated-input feature reuse, "
        "not proven acceleration of a real robot."
    )

st.divider()
st.subheader("🔬 Cache Correctness Test")

if st.button("Test Changing Images"):
    with torch.inference_mode():
        # Original image and its cached features
        original_features = model.encode_vision(image)

        # Simulate a changed camera observation
        changed_image = image + 0.5 * torch.randn_like(image)

        # Correct output: recompute vision features
        correct_action = model(changed_image, instruction)

        # Stale cache: reuse old image features
        stale_action = model.predict_action(
            original_features, instruction
        )

        # Refresh the cache after the image changes
        refreshed_features = model.encode_vision(changed_image)
        refreshed_action = model.predict_action(
            refreshed_features, instruction
        )

        stale_error = torch.max(
            torch.abs(correct_action - stale_action)
        ).item()

        refreshed_error = torch.max(
            torch.abs(correct_action - refreshed_action)
        ).item()

    st.metric("Stale Cache Max Error", f"{stale_error:.8f}")
    st.metric("Refreshed Cache Max Error", f"{refreshed_error:.8f}")

    st.write(
        "Stale cache output matches:",
        torch.allclose(correct_action, stale_action,
                       atol=1e-6, rtol=1e-5)
    )

    st.write(
        "Refreshed cache output matches:",
        torch.allclose(correct_action, refreshed_action,
                       atol=1e-6, rtol=1e-5)
    )

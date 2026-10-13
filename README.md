# TulipFast

A lightweight inference profiling and optimization prototype for robotics-inspired AI workloads.

## Features
- Synthetic VLA-style neural network
- CPU inference benchmarking
- Vision feature caching experiment
- Before/after latency comparison
- CSV result export

## Current Experimental Results
- Baseline: 0.1649 ms
- Cached Vision: 0.0741 ms
- Measured ratio: 2.22x

These results were measured on a synthetic model with repeated identical image inputs. They do not represent performance of a pretrained VLA model.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Future Work
- Real VLA model integration
- Cache invalidation for changing observations
- Humanoid robot simulation
- ASUS UGen hardware evaluation

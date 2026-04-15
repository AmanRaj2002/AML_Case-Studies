# AML Case Study: Uber ETA Prediction

**Goal:** End-to-end mental simulation of traditional regression use cases representing common AI/ML applications in practice. 

This repository models a hyper-scale Estimated Time of Arrival (ETA) prediction system. It bridges the gap between algorithmic experimentation and production-grade MLOps by simulating a distributed, latency-optimized system.

It closely follows the following case study: [DeeprETA: An ETA Post-processing System at Scale](https://www.uber.com/in/en/blog/deepeta-how-uber-predicts-arrival-times/)

### Key Architectural Features

* **Route-Level Correction:** Refines baseline estimates from a physical routing engine by predicting an overarching trip residual rather than calculating segment-by-segment ETAs.
* **Linear Transformer:** Implements a linearized attention mechanism to reduce computational complexity from quadratic to linear, satisfying strict sub-millisecond inference budgets.
* **Asymmetric Huber Loss:** Uses a custom loss function to heavily penalize under-predictions (arriving late) versus over-predictions (arriving early), aligning mathematical optimization with marketplace realities.

---

### Repository Structure

The codebase utilizes a modular MLOps structure:

* **`data.py`**: Simulates telemetry data, feature embeddings, and handles tensor dimensionality.
* **`model.py`**: Houses the deep learning intelligence, including the Linear Attention block and the residual network.
* **`loss.py`**: Contains the asymmetric loss logic and weighting parameters.
* **`train.py`**: Simulates the training optimization pipeline.
* **`evaluate.py`**: Executes the real-time scoring flow and generates business-critical metrics.
* **`pipeline.ipynb`**: The consolidated master notebook that executes the end-to-end workflow and logs results.

---

### Quick Start

To execute the pipeline and view the evaluation metrics:

1. Clone the repository.
2. Ensure `torch` and `numpy` are installed in your environment.
3. Open and run all cells in `pipeline.ipynb`.
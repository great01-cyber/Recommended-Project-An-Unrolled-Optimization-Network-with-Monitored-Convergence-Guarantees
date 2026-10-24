# Deep Unrolled Optimization for Computational Imaging with Agentic Convergence Supervision

This repository presents an implementation of **Algorithm Unrolling (Model-Based Deep Learning)** for computational inverse problems, bridging formal optimization theory and deep learning.

## Key Features
1. **Physical Operator Constrained Model:** Implements a Learned Iterative Shrinkage-Thresholding Algorithm (LISTA) unrolling $K=6$ optimization layers.
2. **Convergence Guarantees & Lipschitz Analysis:** Calculates operator spectral norms ($L$) via power iteration to establish convergence bounds ($t < 1/L$).
3. **Agentic Optimization Supervisor:** Includes a runtime diagnostic agent (`AgenticOptimizationSupervisor`) that monitors layer-wise residuals and dynamically flags divergence or stagnation conditions.
4. **Physically Interpretable Internal Trajectories:** Tracks intermediate convergence layers to verify monotonic objective reduction.

## Mathematical Formulation
The objective function solved by the unrolled architecture is:
$$\min_{\mathbf{x}} \frac{1}{2} \|\mathbf{y} - \mathbf{A}\mathbf{x}\|_2^2 + \lambda \|\mathbf{x}\|_1$$

Where $\mathbf{A}$ represents the physical forward imaging matrix and $\mathbf{x}$ is the target sparse signal.

## Execution
To run the full training, evaluation, and supervisor diagnostic pipeline:
```bash
pip install -r requirements.txt
python run_demo.py
```
# Recommended-Project-An-Unrolled-Optimization-Network-with-Monitored-Convergence-Guarantees

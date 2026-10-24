import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_comparison(classical_res: list, unrolled_res: list, save_path: str = "convergence_comparison.png"):
    """Plots and saves the convergence trajectory comparison."""
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(classical_res) + 1), classical_res, 'r--o', label="Classical ISTA (50 Iterations)")
    plt.semilogy(range(1, len(unrolled_res) + 1), unrolled_res, 'b-s', label="Unrolled LISTA (6 Layers)")
    plt.xlabel("Layer / Iteration Index", fontsize=12)
    plt.ylabel("Data Fidelity Residual ||Ax - y||_2", fontsize=12)
    plt.title("Convergence Comparison: Classical vs. Deep Unrolled Network", fontsize=13)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_ground_truth_reconstruction(gt: np.ndarray, lista_rec: np.ndarray, save_path: str = "reconstruction.png"):
    """Plots original signal vs unrolled network recovery."""
    plt.figure(figsize=(10, 4))
    plt.plot(gt[0], label="Ground Truth (Sparse Signal)", color="black", alpha=0.7, linewidth=2)
    plt.plot(lista_rec[0], label="LISTA Reconstruction", color="crimson", linestyle="--", linewidth=1.5)
    plt.title("Sparse Signal Reconstruction Performance", fontsize=13)
    plt.xlabel("Signal Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

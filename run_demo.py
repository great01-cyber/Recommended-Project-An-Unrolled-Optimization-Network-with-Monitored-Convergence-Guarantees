import torch
import torch.optim as optim
from src.operators import InverseProblemOperator
from src.models import UnrolledLISTANetwork, classical_ista
from src.agent import AgenticOptimizationSupervisor
from src.utils import plot_convergence_comparison, plot_ground_truth_reconstruction

def generate_sparse_dataset(num_samples: int, signal_dim: int, sparsity: int = 10):
    """Generates synthetic K-sparse ground truth signals."""
    X = torch.zeros(num_samples, signal_dim)
    for i in range(num_samples):
        indices = torch.randperm(signal_dim)[:sparsity]
        X[i, indices] = torch.randn(sparsity)
    return X

def main():
    print("--- 1. Initializing Physical Imaging System ---")
    in_features, out_features = 128, 64
    operator = InverseProblemOperator(in_features=in_features, out_features=out_features)
    L = operator.compute_spectral_norm()
    print(f"Operator bounds computed. Spectral Norm (Lipschitz Constant L) = {L:.4f}")

    # Generate synthetic sparse dataset
    X_gt = generate_sparse_dataset(num_samples=500, signal_dim=in_features, sparsity=8)
    Y_meas = operator.forward(X_gt) + 0.01 * torch.randn(X_gt.shape[0], out_features) # Add noise

    print("\n--- 2. Building & Training Unrolled LISTA Network ---")
    num_layers = 6
    model = UnrolledLISTANetwork(operator_matrix=operator.A, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        x_pred, _ = model(Y_meas)
        loss = criterion(x_pred, X_gt)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/100] | Loss: {loss.item():.6f}")

    print("\n--- 3. Running Convergence & Layer Interpretability Analysis ---")
    model.eval()
    with torch.no_grad():
        test_gt = X_gt[:1]
        test_y = Y_meas[:1]
        lista_rec, trajectory = model(test_y)
        
        # Calculate residuals across unrolled layers
        unrolled_residuals = [torch.norm(operator.forward(x_k) - test_y).item() for x_k in trajectory[1:]]

    # Run classical ISTA for comparison
    _, classical_residuals = classical_ista(test_y, operator.A, num_iterations=50)

    # Save visual outputs
    plot_convergence_comparison(classical_residuals, unrolled_residuals)
    plot_ground_truth_reconstruction(test_gt.numpy(), lista_rec.numpy())
    print("Reconstruction plots saved to disk.")

    print("\n--- 4. Invoking Agentic Optimization Supervisor ---")
    agent = AgenticOptimizationSupervisor()
    agent_report = agent.InspectAndEvaluate(layer_residuals=unrolled_residuals, operator_lipschitz=L)

if __name__ == "__main__":
    main()

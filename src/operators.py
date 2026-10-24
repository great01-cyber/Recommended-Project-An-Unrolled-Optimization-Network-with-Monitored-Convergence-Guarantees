import torch

class InverseProblemOperator:
    """
    Simulates a linear forward imaging operator A: R^N -> R^M.
    Includes power iteration to compute the spectral norm (Lipschitz constant)
    needed for convergence bounds.
    """
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        torch.manual_seed(seed)
        # Random measurement matrix representing a compressive imaging system
        self.A = torch.randn(out_features, in_features) / (out_features ** 0.5)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward operator y = A x."""
        return torch.matmul(x, self.A.T)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Applies adjoint operator x = A^T y."""
        return torch.matmul(y, self.A)

    def compute_spectral_norm(self, num_iterations: int = 100) -> float:
        """
        Computes L = ||A||_2^2 using Power Iteration to establish 
        the theoretical step-size bound for ISTA (t < 1/L).
        """
        v = torch.randn(1, self.in_features)
        v = v / torch.norm(v)
        
        for _ in range(num_iterations):
            v_next = self.adjoint(self.forward(v))
            v = v_next / torch.norm(v_next)
            
        spectral_norm = torch.norm(self.forward(v))**2
        return spectral_norm.item()

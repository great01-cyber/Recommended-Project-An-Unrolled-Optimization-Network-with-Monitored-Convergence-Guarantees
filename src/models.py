
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class SoftThresholding(nn.Module):
    """Proximal operator for L1 regularization."""

    def __init__(self, init_thresh: float = 0.05):
        super().__init__()

        if init_thresh <= 0:
            raise ValueError("init_thresh must be positive.")

        # Inverse softplus ensures the effective threshold starts
        # at init_thresh, rather than softplus(init_thresh).
        raw_init = torch.log(torch.expm1(torch.tensor(float(init_thresh))))
        self.threshold = nn.Parameter(raw_init.reshape(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = F.softplus(self.threshold)
        return torch.sign(x) * F.relu(torch.abs(x) - threshold)


class UnrolledLISTANetwork(nn.Module):
    """
    Learned Iterative Shrinkage-Thresholding Algorithm (LISTA).
    Physics-informed initialization with trainable weights and thresholds.
    """

    def __init__(
        self,
        operator_matrix: torch.Tensor,
        num_layers: int = 6,
        lmbda: float = 0.1,
    ):
        super().__init__()

        self.num_layers = num_layers
        out_features, in_features = operator_matrix.shape

        # ISTA step-size: 1 / ||A||_2^2
        L = torch.linalg.matrix_norm(operator_matrix, ord=2) ** 2
        step_size = 1.0 / L

        # Physics-informed initialization
        W1_init = (
            torch.eye(
                in_features,
                device=operator_matrix.device,
                dtype=operator_matrix.dtype,
            )
            - step_size * (operator_matrix.T @ operator_matrix)
        )
        W2_init = step_size * operator_matrix.T

        self.W1 = nn.ParameterList([
            nn.Parameter(W1_init.clone()) for _ in range(num_layers)
        ])
        self.W2 = nn.ParameterList([
            nn.Parameter(W2_init.clone()) for _ in range(num_layers)
        ])

        # Match the threshold used by classical ISTA:
        # step_size * lambda = lambda / ||A||_2^2
        initial_threshold = float((step_size * lmbda).item())

        self.thresholds = nn.ModuleList([
            SoftThresholding(init_thresh=initial_threshold)
            for _ in range(num_layers)
        ])

    def forward(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        batch_size = y.shape[0]
        in_features = self.W1[0].shape[0]

        x_k = torch.zeros(
            batch_size,
            in_features,
            device=y.device,
            dtype=y.dtype,
        )

        trajectory = [x_k]

        for k in range(self.num_layers):
            linear_step = (
                x_k @ self.W1[k].T
                + y @ self.W2[k].T
            )
            x_k = self.thresholds[k](linear_step)
            trajectory.append(x_k)

        return x_k, trajectory


def classical_ista(
    y: torch.Tensor,
    A: torch.Tensor,
    num_iterations: int = 50,
    lmbda: float = 0.1,
) -> Tuple[torch.Tensor, List[float]]:
    """Classical ISTA baseline for L1-regularized reconstruction."""

    L = torch.linalg.matrix_norm(A, ord=2) ** 2
    step_size = 1.0 / L

    x = torch.zeros(
        y.shape[0],
        A.shape[1],
        device=y.device,
        dtype=y.dtype,
    )

    residuals = []

    for _ in range(num_iterations):
        grad = (x @ A.T - y) @ A
        x_unc = x - step_size * grad

        threshold = step_size * lmbda
        x = torch.sign(x_unc) * torch.clamp(
            torch.abs(x_unc) - threshold,
            min=0.0,
        )

        residual = torch.norm(x @ A.T - y).item()
        residuals.append(residual)

    return x, residuals

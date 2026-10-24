import torch
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="[Agentic Supervisor] %(levelname)s: %(message)s")

class AgenticOptimizationSupervisor:
    """
    An agentic controller that inspects theoretical properties and 
    convergence trajectories of the optimization algorithm during runtime.
    """
    def __init__(self, tolerance: float = 1e-4, divergence_threshold: float = 1e3):
        self.tolerance = tolerance
        self.divergence_threshold = divergence_threshold

    def InspectAndEvaluate(self, layer_residuals: list[float], operator_lipschitz: float) -> Dict[str, Any]:
        """
        Analyzes the trajectory of iterative residuals and applies diagnostic policies.
        """
        actions = []
        status = "STABLE"

        if len(layer_residuals) < 2:
            return {"status": "INSUFFICIENT_DATA", "actions": []}

        initial_res = layer_residuals[0]
        latest_res = layer_residuals[-1]
        res_change = layer_residuals[-2] - latest_res

        # Diagnostic Policy 1: Divergence Check
        if latest_res > self.divergence_threshold or latest_res > initial_res * 10:
            status = "DIVERGED"
            actions.append("Reduce step-size parameter; upper bound 1/L exceeded.")

        # Diagnostic Policy 2: Stagnation / Premature Convergence
        elif abs(res_change) < self.tolerance:
            status = "STAGNATED"
            actions.append("Decrease threshold parameter or apply adaptive momentum.")

        # Diagnostic Policy 3: Monotonic Decrease Check
        elif res_change > 0:
            status = "CONVERGING"
            actions.append("Descent condition satisfied. Monotonicity maintained.")

        logging.info(f"Evaluation Complete | Status: {status} | Lipschitz Constant (L): {operator_lipschitz:.4f}")
        for action in actions:
            logging.info(f"Recommended Policy Action -> {action}")

        return {
            "status": status,
            "latest_residual": latest_res,
            "lipschitz_constant": operator_lipschitz,
            "actions": actions
        }

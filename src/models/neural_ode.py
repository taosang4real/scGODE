import torch
import torch.nn as nn
from .node_dynamics import NODEDynamics

try:
    from torchdiffeq import odeint_adjoint, odeint
except ImportError:
    print("torchdiffeq not found. Please install it to use NeuralODE: pip install torchdiffeq")
    odeint_adjoint = None
    odeint = None


class NeuralODE(nn.Module):
    def __init__(self, main_model_params, node_specific_config):
        super(NeuralODE, self).__init__()

        if odeint is None and odeint_adjoint is None:
            raise ImportError("torchdiffeq is required but not installed or failed to import.")

        self.latent_dim = int(main_model_params.latent_dim)

        self.dynamics_fn = NODEDynamics(
            latent_dim=self.latent_dim,
            hidden_dims=[int(d) for d in node_specific_config.dynamics_hidden_dims],
            gnn_layer_type=str(node_specific_config.dynamics_gnn_type),
            activation=str(node_specific_config.dynamics_activation),
            dropout_rate=float(node_specific_config.dynamics_dropout_rate),
            time_dependent=bool(node_specific_config.get("time_dependent", False))
        )

        self.solver_method = str(node_specific_config.solver_method)
        self.solver_integrator = str(node_specific_config.solver_integrator)
        self.solver_options_config = node_specific_config.solver_options if hasattr(node_specific_config,
                                                                                    'solver_options') else {}

        if not isinstance(self.solver_options_config, dict):
            try:
                self.solver_options_config = dict(self.solver_options_config)
            except:
                print("Warning: Could not convert solver_options to dict for NeuralODE. Using empty dict.")
                self.solver_options_config = {}

    def forward(self, z_t0_nodes, t_eval_points, edge_index, edge_weight=None, batch_vector=None):
        if z_t0_nodes.size(0) > 0 and z_t0_nodes.size(1) != self.latent_dim:
            raise ValueError(
                f"Initial state z_t0_nodes feature dim {z_t0_nodes.size(1)} "
                f"does not match NODE latent_dim {self.latent_dim}"
            )

        if z_t0_nodes.size(0) == 0:
            return torch.empty((len(t_eval_points), 0, self.latent_dim), device=z_t0_nodes.device,
                               dtype=z_t0_nodes.dtype)

        ode_func_with_graph = lambda t, y: self.dynamics_fn(t, y, edge_index, edge_weight, batch_vector)

        rtol = float(self.solver_options_config.get('rtol', 1e-5))
        atol = float(self.solver_options_config.get('atol', 1e-7))

        additional_solver_options = {
            k: v for k, v in self.solver_options_config.items()
            if k not in ['rtol', 'atol']
        }
        options_to_pass = additional_solver_options if additional_solver_options else None
        solver_method_to_pass = self.solver_method

        if self.solver_integrator == "odeint_adjoint":
            if odeint_adjoint is None:
                raise RuntimeError("odeint_adjoint from torchdiffeq is not available.")
            solution = odeint_adjoint(
                ode_func_with_graph,
                z_t0_nodes,
                t_eval_points,
                method=solver_method_to_pass,
                rtol=rtol,
                atol=atol,
                adjoint_params=tuple(self.dynamics_fn.parameters())
            )
        elif self.solver_integrator == "odeint":
            if odeint is None:
                raise RuntimeError("odeint from torchdiffeq is not available.")
            solution = odeint(
                ode_func_with_graph,
                z_t0_nodes,
                t_eval_points,
                method=solver_method_to_pass,
                rtol=rtol,
                atol=atol,
                options=options_to_pass
            )
        else:
            raise ValueError(f"Unsupported ODE integrator: {self.solver_integrator}")

        return solution

    def solve_to_t1(self, z_t0_nodes, t0, t1, edge_index, edge_weight=None, batch_vector=None):
        _t0 = float(t0.item() if isinstance(t0, torch.Tensor) else t0)
        _t1 = float(t1.item() if isinstance(t1, torch.Tensor) else t1)

        tensor_dtype = z_t0_nodes.dtype if isinstance(z_t0_nodes,
                                                      torch.Tensor) and z_t0_nodes.is_floating_point() else torch.float32
        t_eval_points = torch.tensor([_t0, _t1],
                                     device=z_t0_nodes.device if isinstance(z_t0_nodes, torch.Tensor) else 'cpu',
                                     dtype=tensor_dtype)

        solution = self.forward(z_t0_nodes, t_eval_points, edge_index, edge_weight, batch_vector)

        if solution.shape[0] > 1:
            return solution[1]
        elif solution.shape[0] == 1:
            return solution[0]
        else:
            return solution

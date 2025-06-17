import torch
import torch.nn as nn
from .cnf_dynamics import CNFDynamicsGNN, hutchinson_trace_estimator

try:
    from torchdiffeq import odeint_adjoint, odeint
except ImportError:
    print("torchdiffeq not found. Please install it to use CNFIntegrator: pip install torchdiffeq")
    odeint_adjoint = None
    odeint = None


class CNFIntegrator(nn.Module):
    def __init__(self, main_model_params, cnf_node_config):
        super(CNFIntegrator, self).__init__()

        if odeint is None:
            raise ImportError(
                "torchdiffeq's odeint is required for CNFIntegrator training but not installed or failed to import.")

        self.latent_dim = int(main_model_params.latent_dim)

        self.dynamics_fn = CNFDynamicsGNN(
            latent_dim=self.latent_dim,
            hidden_dims=[int(d) for d in cnf_node_config.dynamics_hidden_dims],
            gnn_layer_type=str(cnf_node_config.dynamics_gnn_type),
            activation=str(cnf_node_config.dynamics_activation),
            dropout_rate=float(cnf_node_config.dynamics_dropout_rate),
            time_dependent=bool(cnf_node_config.get("time_dependent", False)),
            hutchinson_samples=int(cnf_node_config.get("hutchinson_samples_divergence", 1))
        )

        self.solver_method = str(cnf_node_config.solver_method)
        self.solver_integrator_config = str(cnf_node_config.get("solver_integrator", "odeint_adjoint"))
        self.solver_options_config = cnf_node_config.solver_options if hasattr(cnf_node_config,
                                                                               'solver_options') else {}

        if not isinstance(self.solver_options_config, dict):
            try:
                self.solver_options_config = dict(self.solver_options_config)
            except:
                self.solver_options_config = {}

        self.hutchinson_samples_for_div = int(cnf_node_config.get("hutchinson_samples_divergence", 1))

    def _augmented_dynamics(self, t, augmented_state_tuple, edge_index, edge_weight, batch_vector):
        # Unpack augmented state: [z, delta_logp, kinetic_energy, jacobian_frobenius_sq]
        z_nodes, _, _, _ = augmented_state_tuple

        if not z_nodes.requires_grad: z_nodes.requires_grad_(True)

        dz_dt = self.dynamics_fn.get_dz_dt(t, z_nodes, edge_index, edge_weight, batch_vector)

        # Divergence for d(delta_logp)/dt
        d_delta_logp_dt = torch.tensor(0.0, device=dz_dt.device, dtype=dz_dt.dtype)
        divergence = torch.tensor(0.0, device=dz_dt.device, dtype=dz_dt.dtype)
        if self.training:
            func_for_hutchinson = lambda z_prime: self.dynamics_fn.get_dz_dt(t, z_prime, edge_index, edge_weight,
                                                                             batch_vector)
            divergence = hutchinson_trace_estimator(func_for_hutchinson, z_nodes,
                                                    num_samples=self.hutchinson_samples_for_div)
            d_delta_logp_dt = -divergence
            if d_delta_logp_dt.numel() != 1 and dz_dt.numel() > 0:
                if d_delta_logp_dt.dim() > 0: d_delta_logp_dt = d_delta_logp_dt.sum()

        # Kinetic energy for d(kinetic_energy)/dt
        d_kinetic_energy_dt = (dz_dt.pow(2)).sum(dim=-1).mean()

        # Jacobian Frobenius norm squared for d(jacobian_reg)/dt
        # This can be estimated with Hutchinson as well: E[||J_f v||_2^2]
        # For simplicity, we can regularize the divergence squared instead, which we already have.
        # This is a common and effective alternative to full Frobenius norm regularization.
        d_jacobian_reg_dt = divergence.pow(2)

        return (dz_dt, d_delta_logp_dt, d_kinetic_energy_dt, d_jacobian_reg_dt)

    def forward(self, z_t0_nodes, t_eval_points, edge_index, edge_weight=None, batch_vector=None):
        if z_t0_nodes.size(0) > 0 and z_t0_nodes.size(1) != self.latent_dim:
            raise ValueError(
                f"Initial state z_t0_nodes feature dim {z_t0_nodes.size(1)} does not match CNF latent_dim {self.latent_dim}")

        # Initial augmented state: (z, delta_logp, kinetic_energy, jacobian_reg)
        initial_delta_logp = torch.tensor(0.0, device=z_t0_nodes.device, dtype=z_t0_nodes.dtype)
        initial_kinetic_energy = torch.tensor(0.0, device=z_t0_nodes.device, dtype=z_t0_nodes.dtype)
        initial_jacobian_reg = torch.tensor(0.0, device=z_t0_nodes.device, dtype=z_t0_nodes.dtype)
        y0_augmented = (z_t0_nodes, initial_delta_logp, initial_kinetic_energy, initial_jacobian_reg)

        ode_func_augmented_with_graph = lambda t, y_aug_tuple: self._augmented_dynamics(
            t, y_aug_tuple, edge_index, edge_weight, batch_vector
        )

        rtol = float(self.solver_options_config.get('rtol', 1e-5))
        atol = float(self.solver_options_config.get('atol', 1e-7))
        options_to_pass = {k: v for k, v in self.solver_options_config.items() if k not in ['rtol', 'atol']} or None
        solver_method_to_pass = self.solver_method

        effective_solver_integrator = "odeint" if self.training else self.solver_integrator_config

        if z_t0_nodes.size(0) == 0:
            return (
            torch.empty((len(t_eval_points), 0, self.latent_dim), device=z_t0_nodes.device, dtype=z_t0_nodes.dtype),
            torch.zeros((len(t_eval_points),), device=z_t0_nodes.device, dtype=z_t0_nodes.dtype),
            torch.zeros((len(t_eval_points),), device=z_t0_nodes.device, dtype=z_t0_nodes.dtype),
            torch.zeros((len(t_eval_points),), device=z_t0_nodes.device, dtype=z_t0_nodes.dtype))

        # Always use odeint for training CNF due to create_graph=True
        if odeint is None: raise RuntimeError("odeint from torchdiffeq is not available.")
        solution_tuple = odeint(
            ode_func_augmented_with_graph, y0_augmented, t_eval_points,
            method=solver_method_to_pass, rtol=rtol, atol=atol, options=options_to_pass
        )

        return solution_tuple  # (z_traj, d_logp_traj, ke_traj, jac_reg_traj)

    def solve_to_t1(self, z_t0_nodes, t0, t1, edge_index, edge_weight=None, batch_vector=None):
        _t0 = float(t0.item() if isinstance(t0, torch.Tensor) else t0)
        _t1 = float(t1.item() if isinstance(t1, torch.Tensor) else t1)

        tensor_dtype = z_t0_nodes.dtype if isinstance(z_t0_nodes,
                                                      torch.Tensor) and z_t0_nodes.is_floating_point() else torch.float32
        t_eval_points = torch.tensor([_t0, _t1],
                                     device=z_t0_nodes.device if isinstance(z_t0_nodes, torch.Tensor) else 'cpu',
                                     dtype=tensor_dtype)

        z_traj, d_logp_traj, ke_traj, jac_reg_traj = self.forward(
            z_t0_nodes, t_eval_points, edge_index, edge_weight, batch_vector
        )

        z_t1 = z_traj[-1] if z_traj.shape[0] > 0 and z_traj.numel() > 0 else torch.empty_like(z_t0_nodes)

        output_device = z_t1.device
        output_dtype = z_t1.dtype

        delta_logp_t1 = d_logp_traj[-1] if d_logp_traj.numel() > 0 else torch.tensor(0.0, device=output_device,
                                                                                     dtype=output_dtype)
        kinetic_energy_t1 = ke_traj[-1] if ke_traj.numel() > 0 else torch.tensor(0.0, device=output_device,
                                                                                 dtype=output_dtype)
        jacobian_reg_t1 = jac_reg_traj[-1] if jac_reg_traj.numel() > 0 else torch.tensor(0.0, device=output_device,
                                                                                         dtype=output_dtype)

        return z_t1, delta_logp_t1, kinetic_energy_t1, jacobian_reg_t1

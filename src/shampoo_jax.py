import enum
import itertools

import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
from flax.optim.base import OptimizerDef
from flax.optim.base import OptimizerState
from jax import lax

# Precision to use for matrix inverse pth root. Switch to f64 if you have
# hardware that supports it.
_INVERSE_PTH_ROOT_DATA_TYPE = jnp.float32

# Numerics are hard. Inverses fail sometimes. We determine that using this
# threshold.
_INVERSE_PTH_ROOT_FAILURE_THRESHOLD = 0.1

# Inverse pth root precision (XLA related) flag.
#
# Options are:
# a. lax.Precision.DEFAULT (Better step time, but not precise)
# b. lax.Precision.HIGH (Increased precision, slower)
# c. lax.Precision.HIGHEST (Best possible precision, slowest)
#
_INVERSE_PTH_ROOT_PRECISION = lax.Precision.HIGHEST


# Grafting is a technique to fix the layerwise scale of Shampoo optimizer.
# https://arxiv.org/pdf/2002.11803.pdf studies this in detail. Moreover this
# allows us to plugin the Shampoo optimizer into settings where SGD/AdaGrad
# is already well tuned.
class LayerwiseGrafting(enum.IntEnum):
    SGD = 1
    ADAGRAD = 2


@struct.dataclass
class _ShampooHyperParams:
    """Shampoo hyperparameters."""

    learning_rate: float
    # Momentum (in Heavy-Ball or Nesterov, if nesterov is True).
    beta1: onp.ndarray
    # Parameter for exponential moving average of Shampoo second moment statistics
    # if set == 1.0, then sums statistics instead of moving average.
    beta2: onp.ndarray
    # Only set if using Layerwise grafting mode to adagrad. This is the epsilon
    # for adagrad update.
    diagonal_eps: float

    # Epsilon to add to statistics before computing inverse pth root. If you are
    # running in f32 precision for inverse pth root (recommended today)
    # this can go upto 1e-6. If you have latest hardware with native f64 precision
    # set this upto 1e-12.
    matrix_eps: float

    # Weight decay parameter for regularization.
    weight_decay: float

    # When to start Shampoo update before which diagonal update is used. This is
    # because we do not have enough information to compute a stable inverse.
    start_preconditioning_step: int

    # Performance tuning params for controlling memory and compute requirements.
    # How often to compute preconditioner. Ideally set both params to 1.
    preconditioning_compute_steps: int
    # How often to compute statistics.
    statistics_compute_steps: int

    # Block size for large layers (if > 0).
    block_size: int

    # if there are some small dimensions, collapse them:
    # e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if block = 1024
    # [1, 2, 768, 1, 2048] --> [2, 768, 2048]
    best_effort_shape_interpretation: bool

    # Type of grafting (SGD or AdaGrad).
    # https://arxiv.org/pdf/2002.11803.pdf
    graft_type: int

    # Avoids preconditioning large layers to reduce overall memory usage if any
    # of the dimensions are greater than this value.
    no_preconditioning_for_layers_with_dim_gt: int

    # Nesterov momentum
    nesterov: bool
    # Exponent override (if > 0):
    exponent_override: int
    # Batch axis name (for data parallel code).
    batch_axis_name: str


class BlockPartitioner:
    """Partitions a tensor into smaller tensors."""

    def __init__(self, param, hps):
        self._shape = param.shape
        self.splits = []
        split_sizes = []
        # We split params into smaller blocks. Here we store the metadata to make that split.
        for i, d in enumerate(param.shape):
            if 0 < hps.block_size < d:
                # d-1, otherwise split appends a 0-size array.
                nsplit = (d - 1) // hps.block_size
                indices = (onp.arange(nsplit, dtype=onp.int32) + 1) * hps.block_size
                sizes = onp.ones(nsplit + 1, dtype=onp.int32) * hps.block_size
                sizes[-1] = d - indices[-1]
                self.splits.append((i, indices))
                split_sizes.append(sizes)
            else:
                split_sizes.append(onp.array([d], dtype=onp.int32))
        self.num_splits = len(split_sizes)
        self.preconditioner_shapes = [[d, d] for t in itertools.product(*split_sizes) for d in t]

    def partition(self, tensor):
        """Partition tensor into blocks."""
        tensors = [tensor]
        for i, indices in self.splits:
            tensors = [local for t in tensors for local in jnp.split(t, indices_or_sections=indices, axis=i)]
        return tensors


@struct.dataclass
class _ShampooDefaultParamState:
    """Shampoo default parameter state."""

    # Accumulator for diagonal preconditioner
    diagonal_statistics: onp.ndarray
    # Statistics
    statistics: onp.ndarray
    # Preconditioners
    preconditioners: onp.ndarray
    # Momentum for the diagonal preconditioner
    diagonal_momentum: onp.ndarray
    # Momentum for the shampoo preconditioner
    momentum: onp.ndarray


def matrix_inverse_pth_root(mat_g,
                            prev_inverse,
                            iter_count=100,
                            error_tolerance=1e-6,
                            ridge_epsilon=1e-6):
    """Computes mat_g^(-1/p), where p is a positive integer.

    Coupled newton iterations for matrix inverse pth root.

    Args:
      mat_g: the symmetric PSD matrix whose power it to be computed
      prev_inverse: alternative if error too high
      iter_count: Maximum number of iterations.
      error_tolerance: Error indicator, useful for early termination.
      ridge_epsilon: Ridge epsilon added to make the matrix positive definite.

    Returns:
      mat_g^(-1/p)
    """
    mat_g_size = mat_g.shape[0]
    alpha = jnp.asarray(-1.0 / 4, _INVERSE_PTH_ROOT_DATA_TYPE)
    identity = jnp.eye(mat_g_size, dtype=_INVERSE_PTH_ROOT_DATA_TYPE)
    new_v = onp.random.uniform(-1.0, 1.0, mat_g.shape[-1]).astype(mat_g.dtype)

    more = True
    s = jnp.zeros([], dtype=mat_g.dtype)
    while more:
        new_v = new_v / jnp.linalg.norm(new_v)
        s_v = jnp.einsum('ij,j->i', mat_g, new_v, precision=_INVERSE_PTH_ROOT_PRECISION)
        s_new = jnp.einsum('i,i->', new_v, s_v, precision=_INVERSE_PTH_ROOT_PRECISION)
        more = jnp.greater(jnp.abs(s_new - s), error_tolerance)
        s = s_new

    ridge_epsilon = ridge_epsilon * jnp.maximum(s, 1e-16)

    if mat_g_size == 1:
        resultant_mat_h = (mat_g + ridge_epsilon) ** alpha
        error = 0
    else:
        damped_mat_g = mat_g + ridge_epsilon * identity
        z = (1 + 4) / (2 * jnp.linalg.norm(damped_mat_g))
        new_mat_m_0 = damped_mat_g * z
        new_error = jnp.max(jnp.abs(new_mat_m_0 - identity))
        new_mat_h_0 = identity * jnp.power(z, 1.0 / 4)
        more = True
        error = new_error
        mat_h = new_mat_h_0
        mat_m = new_mat_m_0
        convergence = 0
        while more:
            mat_m_i = (1 - alpha) * identity + alpha * mat_m
            mat_pow_2 = jnp.matmul(mat_m, mat_m, precision=_INVERSE_PTH_ROOT_PRECISION)
            new_mat_m = jnp.matmul(jnp.matmul(mat_pow_2, mat_pow_2, precision=_INVERSE_PTH_ROOT_PRECISION), mat_m,
                                   precision=_INVERSE_PTH_ROOT_PRECISION)
            new_mat_h = jnp.matmul(mat_h, mat_m_i, precision=_INVERSE_PTH_ROOT_PRECISION)
            new_error = jnp.max(jnp.abs(new_mat_m - identity))
            convergence = new_error < error * 1.2
            more = jnp.logical_and(error > error_tolerance, convergence)
            error = new_error
            mat_h = new_mat_h
            mat_m = new_mat_m
        error = jnp.max(jnp.abs(mat_m - identity))
        is_converged = jnp.asarray(convergence, new_mat_h_0.dtype)
        resultant_mat_h = is_converged * mat_h + (1 - is_converged) * new_mat_h_0
        resultant_mat_h = jnp.asarray(resultant_mat_h, mat_g.dtype)
    if jnp.isnan(error) or error > _INVERSE_PTH_ROOT_FAILURE_THRESHOLD:
        return prev_inverse
    return resultant_mat_h


def compute_shampoo_statistics(hps, param, state, grad):
    """Compute statistics."""
    partitioner = BlockPartitioner(param, hps)
    new_stats = [jnp.tensordot(grad, grad, axes=(list(range(i)) + list(range(i + 1, len(grad.shape))),) * 2)
                 for grad in partitioner.partition(grad) for i in range(len(grad.shape))]
    new_statistics = [hps.beta2 * stat_accumulator + (1 - hps.beta2) * stat
                      for stat, stat_accumulator in zip(new_stats, state.statistics)]
    return _ShampooDefaultParamState(state.diagonal_statistics, new_statistics, state.preconditioners,
                                     state.diagonal_momentum, state.momentum)


class Shampoo(OptimizerDef):
    """Shampoo optimizer.

      Scalable Second Order Optimization for Deep Learning,
      Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer

      Preprint: https://arxiv.org/abs/2002.09018
    """

    def __init__(self,
                 learning_rate=None,
                 beta1=0.9,
                 beta2=0.999,
                 diagonal_epsilon=1e-10,
                 matrix_epsilon=1e-6,
                 weight_decay=0.0,
                 start_preconditioning_step=1,
                 preconditioning_compute_steps=1,
                 statistics_compute_steps=1,
                 block_size=128,
                 best_effort_shape_interpretation=True,
                 graft_type=LayerwiseGrafting.SGD,
                 no_preconditioning_for_layers_with_dim_gt=8192,
                 nesterov=True,
                 exponent_override=0,
                 batch_axis_name=None):
        """Constructor for the Shampoo optimizer.

        Args:
          learning_rate: the step size used to update the parameters.
          beta1: momentum parameter.
          beta2: second moment averaging parameter.
          diagonal_epsilon: epsilon for diagonal adagrad (only if layerwise grafting
            to AdaGrad is enabled).
          matrix_epsilon: epsilon to add to statistics before computing inverse pth
            root. If you are running in f32 precision for inverse pth root
            (recommended today) this can go upto 1e-6. If you have latest hardware
            with native f64 precision, set this upto 1e-12.
          weight_decay: Weight decay for regularization.
          start_preconditioning_step: When to start Shampoo update before which
            diagonal update is used. This is because we dont have enough information
            to do stable inverse.
          preconditioning_compute_steps: How often to compute preconditioner.
            Performance tuning params for controlling memory and compute
            requirements. Ideally set both params to 1.
          statistics_compute_steps: How often to compute statistics.
          block_size: Block size for large layers (if > 0). Preconditioning compute
            operation is cubic in the dimension of the tensor. Block size allows us
            to chunk the layers into sub-layers of maximal dimension dictated by
            this value. Use 128 as default (increase if you have compute budget).
          best_effort_shape_interpretation:
          graft_type: Options are: LayerwiseGrafting.SGD, LayerwiseGrafting.ADAGRAD
          no_preconditioning_for_layers_with_dim_gt: Avoids preconditioning large
            layers to reduce overall memory usage.
          nesterov: Nesterov momentum.
          exponent_override: Override the exponent used in matrix inverse.
          batch_axis_name: labeled axis over pmap for dataparallel training the
            optimizer used for.
        """
        hps = _ShampooHyperParams(
            learning_rate,
            beta1,
            beta2,
            diagonal_epsilon,
            matrix_epsilon,
            weight_decay,
            start_preconditioning_step,
            preconditioning_compute_steps,
            statistics_compute_steps,
            block_size,
            best_effort_shape_interpretation,
            graft_type=graft_type,
            no_preconditioning_for_layers_with_dim_gt=no_preconditioning_for_layers_with_dim_gt,
            nesterov=nesterov,
            exponent_override=exponent_override,
            batch_axis_name=batch_axis_name)
        print(hps)
        super().__init__(hps)

    def init_param_state(self, param):
        """Initialize parameter state."""
        hps = self.hyper_params
        statistics = []
        preconditioners = []
        if param.shape:
            partitioner = BlockPartitioner(param, hps)
            statistics = [self.hyper_params.matrix_eps * jnp.eye(s[0]) for s in partitioner.preconditioner_shapes]
            preconditioners = [jnp.eye(s[0]) for s in partitioner.preconditioner_shapes]

        return _ShampooDefaultParamState([], statistics,
                                         preconditioners, jnp.zeros_like(param),
                                         jnp.zeros_like(param))

    def apply_per_param_gradient(self, step, hps, param, state, grad):
        """Apply per-parameter gradients."""
        precond_grad = grad
        if param.shape:
            partitioner = BlockPartitioner(param, hps)
            partitioned_grads = partitioner.partition(grad)
            preconditioned_partitioned_grads = []
            num_splits = partitioner.num_splits
            for i, grad in enumerate(partitioned_grads):
                preconditioners_for_grad = state.preconditioners[i * num_splits:(i + 1) * num_splits]
                rank = len(grad.shape)
                precond_grad = grad
                for j in range(rank):
                    precond_grad = jnp.tensordot(precond_grad, preconditioners_for_grad[j], axes=[[0], [0]])
                preconditioned_partitioned_grads.append(precond_grad)
            partitions = preconditioned_partitioned_grads
            for i, indices in reversed(partitioner.splits):
                n = len(indices) + 1
                partial_merged_tensors = []
                idx = 0
                while idx < len(partitions):
                    partial_merged_tensors.append(jnp.concatenate(partitions[idx:idx + n], axis=i))
                    idx += n
                partitions = partial_merged_tensors
            precond_grad = partitions[0]

        grad_norm = jnp.linalg.norm(grad)
        precond_grad_norm = jnp.linalg.norm(precond_grad)
        shampoo_update = precond_grad * (grad_norm / (precond_grad_norm + 1e-16))

        shampoo_update_with_wd = shampoo_update + hps.weight_decay * param
        grad_with_wd = grad + hps.weight_decay * param

        shampoo_update_with_wd_momentum = state.momentum * hps.beta1 + shampoo_update_with_wd
        grad_with_wd_momentum = state.diagonal_momentum * hps.beta1 + grad_with_wd

        run_shampoo = (step >= hps.start_preconditioning_step).astype(grad_with_wd_momentum.dtype)

        momentum_update = run_shampoo * shampoo_update_with_wd_momentum + (1.0 - run_shampoo) * grad_with_wd_momentum
        wd_update = run_shampoo * shampoo_update_with_wd + (1.0 - run_shampoo) * grad_with_wd

        momentum_update = wd_update + hps.beta1 * momentum_update  # nesterov

        new_param = param - hps.learning_rate * momentum_update
        new_state = _ShampooDefaultParamState(state.diagonal_statistics, state.statistics, state.preconditioners,
                                              grad_with_wd_momentum, shampoo_update_with_wd_momentum)
        return new_param, new_state

    def apply_gradient(self, hyper_params, params, state, grads):
        """Applies a gradient for a set of parameters.

        Args:
          hyper_params: a named tuple of hyper parameters.
          params: the parameters that should be updated.
          state: a named tuple containing the state of the optimizer
          grads: the gradient tensors for the parameters.

        Returns:
          A tuple containing the new parameters and the new optimizer state.
        """
        step = state.step
        params = params
        states = state.param_states
        grads = grads

        new_states_flat = [compute_shampoo_statistics(hyper_params, param, state, grad)
                           for param, state, grad in zip(params, states, grads)]

        new_states_flat = [_ShampooDefaultParamState(state.diagonal_statistics, state.statistics,
                                                     [matrix_inverse_pth_root(stat, state.preconditioners,
                                                                              ridge_epsilon=hyper_params.matrix_eps)
                                                      for stat in state.statistics],
                                                     state.diagonal_momentum, state.momentum)
                           for state in new_states_flat]

        out = [self.apply_per_param_gradient(step, hyper_params, param, state, grad)
               for param, state, grad in zip(params, new_states_flat, grads)]

        new_params, new_states = list(zip(*out))
        new_state = OptimizerState(step + 1, new_states)
        return new_params, new_state

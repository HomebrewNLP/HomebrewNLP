import copy
import typing

import numpy as np
import revlib
import torch
import torch.utils.data
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

from src.dataclass import Context
from src.optimizers.build import build_optimizer

QUAD_TENSOR = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def orthonormal(inp: typing.Union[torch.Tensor, torch.nn.Parameter, typing.List[int]], gain: float):
    original_input = inp
    if isinstance(inp, list):
        inp = torch.zeros(inp)
    if isinstance(inp, torch.nn.Parameter):
        inp = inp.data
    flat_shape = (inp.shape[0], np.prod(inp.shape[1:]))
    a = torch.rand(flat_shape)
    u, _, v = torch.linalg.svd(a, full_matrices=False)
    inp.copy_((u if u.shape == flat_shape else v).reshape(inp.shape).mul(gain).to(device=inp.device, dtype=inp.dtype))
    if isinstance(original_input, list):
        return torch.nn.Parameter(inp)
    return original_input


class TripleNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scale0: torch.Tensor, scale1: torch.Tensor, shift: torch.Tensor, norm_power: int):
        scale0_relu = scale0.relu()
        inp = scale0_relu.pow(3) * scale1 + shift
        inp = inp - inp.mean(1, True)
        rstd = inp.size(1) ** (1 / norm_power) / inp.norm(norm_power, 1, True)
        inp *= rstd
        if scale1.requires_grad:
            ctx.save_for_backward(scale0_relu, scale1, inp, rstd)
        return inp

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None, None, None
        scale0_relu, scale1, out, rstd = ctx.saved_tensors
        dout = dout * rstd
        dout -= (dout * out).mean(1, True) * out
        dout -= dout.mean(1, True)
        d_scale = dout * scale0_relu.square()
        return d_scale * scale1 * 3, d_scale * scale0_relu, dout, None


def conv(inp: torch.Tensor, weight: torch.Tensor, groups: int, use_pad: bool) -> torch.Tensor:
    if use_pad and weight.size()[-1] - 1 > 0:
        inp = F.pad(inp, (weight.size()[-1] - 1, 0))
    return F.conv1d(inp, weight, groups=groups)


def expert_matmul(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bgf,gfo->bgo", inp, weight)


class AuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        ctx.save_for_backward(inp)
        return inp

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        inp, = ctx.saved_tensors
        inp.mean().backward()


def moe(inp: torch.Tensor, expert_weights: torch.nn.ParameterList, training: bool,
        jitter_epsilon: float, groups: int, experts: int) -> torch.Tensor:
    *expert_weights, gate = expert_weights
    batch, features, sequence = inp.size()
    tokens = batch * sequence
    capacity = tokens // experts

    # get gates
    if gate.dtype != torch.float32:
        gate = gate.float()
    inp = inp.transpose(1, 2).reshape(tokens, features)
    input_fp32 = inp.float()
    if training:
        input_fp32 = input_fp32 * (torch.rand_like(input_fp32) * jitter_epsilon + 1)
    logits = input_fp32.mm(gate)
    gates = F.softmax(logits, dim=1)

    # calculate permutation
    with torch.no_grad():
        mask = torch.ones_like(gates[:, 0])
        out = []
        for g in gates.unbind(1):
            _, idx = torch.topk(g * mask, capacity, 0)
            out.append(idx)
            mask[idx] = 0
        expert_permutation = torch.stack(out, 1)
        expert_permutation = expert_permutation.view(-1, 1).long()
        permutation_inverse = torch.argsort(expert_permutation, 0).view(-1, 1)
        expert_index = permutation_inverse // capacity

    # apply loss
    AuxLoss(gates.sum() / tokens)
    inp = inp * gates.gather(1, expert_index)

    # permute
    inp = inp.gather(0, expert_permutation.expand_as(inp))

    inp = inp.view(tokens // experts, experts * groups, features // groups)
    if len(expert_weights) == 1:
        inp = expert_matmul(inp, expert_weights[0])
    else:
        inp = torch.cat([expert_matmul(c, w) for c, w in zip(inp.chunk(len(expert_weights), 1), expert_weights)], -1)
    inp = inp.reshape(tokens, -1)
    inp = inp.gather(0, permutation_inverse.view(-1, 1).expand_as(inp))
    inp = inp.view(batch, sequence, -1).transpose(1, 2)
    return inp


def moe_check(inp: torch.Tensor, w: torch.nn.ParameterList, training: bool,
              jitter_epsilon: float, groups: int, experts: int) -> torch.Tensor:
    if experts > 0:
        return moe(inp, w, training, jitter_epsilon, groups, experts)
    return conv(inp, w[0], groups, False)


def linear_attention(inp: torch.Tensor, w0: torch.nn.ParameterList, groups0: int, experts0: int, w1: torch.Tensor,
                     w2: torch.nn.ParameterList, groups2: int, experts2: int, bottleneck_group: int, training: bool,
                     norm_power: int, jitter_epsilon: float) -> torch.Tensor:
    inp = moe_check(inp, w0, training, jitter_epsilon, groups0, experts0)
    inp = TripleNorm.apply(*inp.chunk(3, 1), norm_power)
    inp = conv(inp, w1, bottleneck_group, True)
    inp = TripleNorm.apply(*inp.chunk(3, 1), norm_power)
    return moe_check(inp, w2, training, jitter_epsilon, groups2, experts2)


def conv_weight(in_features: int, out_features: int, kernel_size: int, groups: int, std: float):
    return orthonormal(torch.nn.Conv1d(in_features, out_features, (kernel_size,), groups=groups).weight, 1 / std)


def get_lr_scheduler_fn(ctx: Context) -> typing.Callable[[int], float]:
    def _fn(step: int) -> float:
        final_lr = 1 - 2 / (ctx.optimizer.final_step - ctx.optimizer.warmup_end)
        lr = step
        lr /= max(step, ctx.optimizer.warmup_end)
        lr *= final_lr ** max(step - ctx.optimizer.warmup_end, 0)
        # lr *= ctx.optimizer.learning_rate  # It's a multiplier for the initial learning rate, not the LR itself
        return lr

    return _fn


class Trainer(torch.nn.Module):
    def __init__(self, ctx: Context, model: torch.nn.Module, data: typing.Optional[torch.Tensor]):
        super(Trainer, self).__init__()
        self.ctx = ctx
        self.model = torch.jit.trace(model, data) if data else model
        self.optimizer = build_optimizer(ctx, self.model.parameters())
        self.scheduler = LambdaLR(self.optimizer, get_lr_scheduler_fn(ctx))

    @torch.no_grad()
    def _to_device_detach(self, inp: torch.Tensor) -> torch.Tensor:
        return inp.to(device=self.ctx.model.device, non_blocking=True).detach()

    def _forward_backward(self, src: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        src = self._to_device_detach(src)
        msk = torch.rand(src.size(), dtype=torch.float32, device=src.device) > self.ctx.dataset.dropout
        out = self.model(src * msk)
        msk = ~msk
        masked = msk.sum()
        loss = (F.cross_entropy(out, src, reduction="none") * msk).sum() / masked
        loss.backward()
        with torch.inference_mode():
            return loss.detach(), (torch.argmax(out, 1) == src).mul(msk).sum().float() / masked

    @torch.no_grad()
    def _clip_gradient(self):
        for p in self.gradients():
            g_norm = p.grad.norm(2, 0, True).clamp(min=self.ctx.optimizer.agc.zero_division_eps)
            p_norm = p.norm(2, 0, True).clamp(min=self.ctx.optimizer.agc.eps)
            grad_scale = (p_norm / g_norm * self.ctx.optimizer.agc.gradient_clipping).clamp(max=1)
            p.grad.data.copy_(p.grad * grad_scale)

    def accumulated_step(self, data: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        loss = 0
        accuracy = 0
        for src in data:
            lss, acc = self._forward_backward(src)
            loss += lss
            accuracy += acc
        self._clip_gradient()
        return loss, accuracy

    @torch.no_grad()
    def zero_grad(self):
        for p in self.model.parameters():
            p.grad = None

    @torch.no_grad()
    def gradients(self) -> torch.nn.Parameter:
        for p in self.model.parameters():
            if p.grad is None:
                continue
            yield p

    def save(self):
        torch.save(self.state_dict(), self.ctx.model.checkpoint_path)

    def load(self):
        wrong_keys = self.load_state_dict(torch.load(self.ctx.model.checkpoint_path), strict=False)
        for key in wrong_keys.missing_keys + wrong_keys.unexpected_keys:
            if not any(k.startswith('_') for k in key.split('.')):
                if key in wrong_keys.missing_keys:
                    raise ValueError(f"{key} is missing in checkpoint but exists in model")
                if key in wrong_keys.unexpected_keys:
                    raise ValueError(f"{key} is missing in model but exists in checkpoint")


class MomentumNetSide(torch.nn.Module):
    def __init__(self, beta: float):
        super(MomentumNetSide, self).__init__()
        self.beta = beta

    def forward(self, inp: torch.Tensor):
        return inp * self.beta


class LinearAttention(torch.nn.Module):
    def __init__(self, ctx: Context):
        super(LinearAttention, self).__init__()
        self.embedding = torch.nn.Embedding(ctx.dataset.classes, ctx.model.features * 2).to(ctx.model.device)
        orthonormal(self.embedding.weight, ctx.model.input_embedding_std * 2 ** -0.5)

        pos_embd = torch.arange(0, ctx.model.sequence_length).unsqueeze(0) + 1
        self.register_buffer("divisor", pos_embd.unsqueeze(0).to(torch.float).to(ctx.model.device))

        ff = FeedForward(self, ctx, 1, 1)

        modules = [mod.__name__ for mod in attention_modules]
        if ctx.model.attention not in modules:
            raise ValueError(f"{ctx.model.attention} is not a known type of attention. You can pick any of the"
                             f" following: {modules}")
        attn = attention_modules[modules.index(ctx.model.attention)](self, ctx, 1)
        self.expand_sequence = (not attn.get_last) | (not ff.get_last)
        self.stem = revlib.ReversibleSequential(*[c
                                                  for i in range(1, 1 + ctx.model.depth * 2, 2)
                                                  for c in [ff.momentum((1 - ctx.model.momentumnet_beta) /
                                                                        ctx.model.momentumnet_beta ** i,
                                                                        not ctx.model.weight_sharing,
                                                                        i + 1),
                                                            MomentumNetSide(ctx.model.momentumnet_beta ** i),
                                                            attn.momentum((1 - ctx.model.momentumnet_beta) /
                                                                          ctx.model.momentumnet_beta ** (i + 1),
                                                                          not ctx.model.weight_sharing,
                                                                          i + 1),
                                                            MomentumNetSide(ctx.model.momentumnet_beta ** (i + 1))]],
                                                target_device=ctx.model.device,
                                                memory_mode=revlib.MemoryModes.autograd_function)
        self.output = torch.nn.Conv1d(ctx.model.features * 2, ctx.dataset.classes, (1,)).to(ctx.model.device)
        torch.nn.init.orthogonal_(self.output.weight.data)

    def forward(self, inp: torch.Tensor):
        out = inp = self.embedding(inp).transpose(1, 2)
        batch, features, sequence = inp.size()
        if self.expand_sequence:
            out = torch.cat([inp, torch.zeros((batch, features, sequence * len(self.stem.stem)), device=inp.device,
                                              dtype=inp.dtype)], 2)

        out = self.stem(out)
        if self.expand_sequence:
            out = out.view(batch, features, -1, sequence).mean(2)
        return self.output(out)

    def reset_cache(self):
        for mod in self.stem.modules():
            if isinstance(mod, FeedForward):
                mod.reset_cache()


class ParameterStore(torch.nn.Module):
    """
    Something (likely deepspeed) changes all parameters in a ParameterList to [1] even though standalone parameters
    work. That's why a torch.nn.ModuleList of ParameterStores needs to be initialized.
    """

    def __init__(self, param: torch.Tensor):
        super(ParameterStore, self).__init__()
        self.param = torch.nn.Parameter(param)

    def __repr__(self):
        return (f'{self.__class__.__name__}(shape={str(list(self.param.size()))}, device={self.param.device}, '
                f'dtype={self.param.dtype})')


def get_moe_param(in_features: int, out_features: int, groups: int, experts: int, expert_chunks: int, std: float
                  ) -> typing.List[torch.nn.Parameter]:
    if experts:
        experts = groups if experts < 0 else experts
        out = orthonormal([in_features // groups, out_features // groups], std).view(1, in_features // groups, -1)
        out = out.repeat(experts // expert_chunks * groups, 1, 1).detach()
        gate = [orthonormal([in_features, experts], 1)]
        return [torch.nn.Parameter(copy.deepcopy(out)) for _ in range(expert_chunks)] + gate
    return [torch.nn.Parameter(conv_weight(in_features, out_features, 1, groups, std))]


class FeedForward(torch.nn.Module):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float, feature_factor: float = 1):
        super(FeedForward, self).__init__()
        self.ctx = ctx
        self.divisor = lambda: base.divisor
        self.init_scale = init_scale
        self.caching = ctx.eval.cache
        self.kernel_size = ctx.model.conv_kernel_size
        self.bottleneck_group = ctx.model.bottleneck_group
        self.norm_power = ctx.model.norm_power
        self.groups0 = ctx.model.input_groups
        self.groups2 = ctx.model.output_groups
        self.experts0 = ctx.model.experts_in_input
        self.experts2 = ctx.model.experts_in_output
        self.jitter_epsilon = ctx.model.moe_jitter_epsilon
        self.expert_chunks = ctx.model.expert_chunks
        intermediate = int(ctx.model.features * ctx.model.feed_forward_intermediate_factor)
        if feature_factor:
            self.w0 = torch.nn.ParameterList(get_moe_param(ctx.model.features * feature_factor, intermediate * 3,
                                                           self.groups0, self.experts0, self.expert_chunks,
                                                           ctx.model.activation_std))
            self.w1 = conv_weight(intermediate, intermediate * 3, ctx.model.conv_kernel_size,
                                  ctx.model.bottleneck_group, ctx.model.activation_std)
            self.w2 = torch.nn.ParameterList(get_moe_param(intermediate, ctx.model.features * feature_factor,
                                                           self.groups2, self.experts2, self.expert_chunks, 1))
        self.idx: int = 0
        self.depth: int = 0
        self.get_last: bool = True

    def __repr__(self):
        extra = '\n  '.join([f'{name}: {param.size()}' for name, param in self.named_parameters()])
        return f"{self._get_name()}(\n  {extra}\n)"

    def _cut_off(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.size(2) == self.ctx.model.sequence_length:
            return inp

        base_len = self.ctx.model.sequence_length * self.depth
        max_len = base_len + self.ctx.model.sequence_length
        if self.get_last:
            return inp[base_len:max_len]
        return inp[:max_len]

    def _pad(self, inp: torch.Tensor, out: torch.Tensor):
        if inp.size(2) == out.size(2):
            return inp

        batch, features, sequence = inp.size()
        if self.get_last:
            return torch.cat([torch.zeros((batch, features, self.ctx.model.sequence_length * self.depth),
                                          device=out.device, dtype=out.dtype), out,
                              torch.zeros((batch, features,
                                           sequence - self.ctx.model.sequence_length * (self.depth + 1)),
                                          device=out.device, dtype=out.dtype)], 2)
        return torch.cat([out, torch.zeros((batch, features, sequence - out.size(2)), device=out.device,
                                           dtype=out.dtype)], 2)

    def _ff(self, inp: torch.Tensor) -> torch.Tensor:
        return linear_attention(inp, self.w0, self.groups0, self.experts0, self.w1, self.w2, self.groups2,
                                self.experts2,
                                self.bottleneck_group, self.training, self.norm_power, self.jitter_epsilon)

    def _inner_forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self._ff(inp)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self._pad(inp, self._inner_forward(self._cut_off(inp))) * self.init_scale

    def momentum(self, init_scale: float, deep: bool, depth: int):
        out = copy.deepcopy(self) if deep else copy.copy(self)
        out.init_scale = init_scale
        out.depth = depth
        return out


class AttentionBase(FeedForward):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float, feature_factor: float):
        super(AttentionBase, self).__init__(base, ctx, init_scale, feature_factor)
        self.get_last = not ctx.model.omnidirectional


class FFTAttention(AttentionBase):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(FFTAttention, self).__init__(base, ctx, init_scale, 2)

    def _inner_forward(self, inp: torch.Tensor) -> torch.Tensor:
        batch, features, sequence = inp.size()
        out = torch.view_as_real(torch.fft.rfft(inp, 2 * sequence, norm="ortho"))
        out = out.transpose(2, 3).reshape(batch, features * 2, sequence + 1)
        out = self._ff(out)
        out = out.view(batch, features, 2, sequence + 1).transpose(2, 3).contiguous()
        out = torch.view_as_complex(out)
        return torch.fft.irfft(out, 2 * sequence, norm="ortho")[:, :, :sequence]


class SumAttention(AttentionBase):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(SumAttention, self).__init__(base, ctx, init_scale, ctx.model.sum_attention_level)
        self.sum_attention_level = ctx.model.sum_attention_level
        self.weight = conv_weight(ctx.model.features, ctx.model.features, 3, 1, 1)

    def _inner_forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = self._ff(inp).chunk(self.sum_attention_level, 1)
        batch, features, seq = out[0].size()
        return sum(conv(torch.relu(out[0] + sum(out[inner + 1][outer // batch ** inner % batch]
                                                for inner in range(self.sum_attention_level)).unsqueeze(0)),
                        self.weight, 1, True)
                   for outer in range(batch ** (self.sum_attention_level - 1)))


class SqueezeExcitation(AttentionBase):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(SqueezeExcitation, self).__init__(base, ctx, init_scale, 0)
        self.weight0 = orthonormal([ctx.model.features * 2, ctx.model.features * 2 * 3], 1)
        self.weight1 = orthonormal([ctx.model.features * 2, ctx.model.features], 1)

    def _inner_forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = torch.cat([inp.mean(2), inp.max(2).values], 1)
        out = out.mm(self.weight0).chunk(3, 1)
        out = TripleNorm.apply(*out, self.ctx.model.norm_power)
        out = out.mm(self.weight1)
        return out.unsqueeze(2) * inp


class SelfAttention(AttentionBase):
    def __init__(self, base: LinearAttention, ctx: Context, init_scale: float):
        super(SelfAttention, self).__init__(base, ctx, init_scale, 0)
        self.mha = torch.nn.MultiheadAttention(ctx.model.features, 4)
        self.get_last = not ctx.model.omnidirectional

    def _inner_forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.permute(2, 0, 1)
        return self.mha(inp, inp, inp)[0].permute(1, 2, 0)


attention_modules = [FeedForward, FFTAttention, SumAttention, SqueezeExcitation, SelfAttention]

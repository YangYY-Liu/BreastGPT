import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Optional
from einops import rearrange
from torch.nn import LayerNorm
import torch.distributed as dist

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

def get_data_parallel_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_data_parallel_group, "_global_group"):
            get_data_parallel_group._global_group = dist.new_group()
        return get_data_parallel_group._global_group
    else:
        return None

def get_rank(group):
    return dist.get_rank(group=group)

def get_world_size(group):
    if torch.distributed.is_initialized():
        return dist.get_world_size(group=group)
    else:
        return 1

def get_data_parallel_rank():
    return get_rank(get_data_parallel_group())

def get_data_parallel_world_size():
    return get_world_size(get_data_parallel_group())


class Allgather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        world_size = get_data_parallel_world_size()
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype,
                            device=torch.cuda.current_device())
        torch.distributed._all_gather_base(output, input_.contiguous(),
                                        group=get_data_parallel_group())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = get_data_parallel_world_size()

        dim_size = list(grad_output.size())
        assert dim_size[0] % world_size == 0, \
            "First dimension of the tensor should be divisible by tensor parallel size"
        
        dim_size[0] = dim_size[0] // world_size
    
        output = torch.empty(dim_size, dtype=grad_output.dtype,
                            device=torch.cuda.current_device())
        
        torch.distributed._reduce_scatter_base(output, grad_output.contiguous(), 
                                            group=get_data_parallel_group())
        
        return output

all_gather_func = Allgather.apply

class EncoderConfig(object):
    def __init__(self, **kwargs):
        self.encoder_embed_dim = kwargs.pop("encoder_embed_dim", 768)
        self.encoder_attention_heads = kwargs.pop("encoder_attention_heads", 12)
        self.encoder_ffn_embed_dim = kwargs.pop("encoder_ffn_embed_dim", 3072)
        self.encoder_layers = kwargs.pop("encoder_layers", 12)
        self.encoder_normalize_before = kwargs.pop("encoder_normalize_before", True)
        self.normalize_output = kwargs.pop("normalize_output", True)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.moe_freq = kwargs.pop("moe_freq", 0)
        self.moe_top1_expert = kwargs.pop("moe_top1_expert", False)
        self.moe_expert_count = kwargs.pop("moe_expert_count", 0)
        self.moe_gating_use_fp32 = kwargs.pop("moe_gating_use_fp32", True)
        self.moe_eval_capacity_token_fraction = kwargs.pop(
            "moe_eval_capacity_token_fraction", 0.25
        )
        self.moe_second_expert_policy = kwargs.pop("moe_second_expert_policy", "random")
        self.moe_normalize_gate_prob_before_dropping = kwargs.pop(
            "moe_normalize_gate_prob_before_dropping", False
        )
        self.use_xmoe = kwargs.pop("use_xmoe", False)
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.bert_init = kwargs.pop("bert_init", False)
        self.multiway = kwargs.pop("multiway", False)
        self.share_encoder_input_output_embed = kwargs.pop(
            "share_encoder_input_output_embed", False
        )
        self.max_source_positions = kwargs.pop("max_source_positions", 1024)
        self.no_output_layer = kwargs.pop("no_output_layer", False)
        self.layernorm_eps = kwargs.pop("layernorm_eps", 1e-5)
        # Text
        self.vocab_size = kwargs.pop("vocab_size", -1)
        # Vision
        self.img_size = kwargs.pop("img_size", 224)
        self.patch_size = kwargs.pop("patch_size", 16)
        self.in_chans = kwargs.pop("in_chans", 3)
        # Fairscale
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        self.xpos_rel_pos = kwargs.pop("xpos_rel_pos", False)
        self.xpos_scale_base = kwargs.pop("xpos_scale_base", 512)
        # Dilated Attention
        self.flash_attention = kwargs.pop("flash_attention", False)
        self.segment_length = kwargs.pop("segment_length", None)
        self.dilated_ratio = kwargs.pop("dilated_ratio", None)
        self.seq_parallel = kwargs.pop("seq_parallel", False)
        self.postprocessing()

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)
        self.postprocessing()

    def postprocessing(self):
        if self.segment_length is not None and self.segment_length != '':
            self.segment_length = eval(self.segment_length)
        if self.dilated_ratio is not None and self.dilated_ratio != '':
            self.dilated_ratio = eval(self.dilated_ratio)

        if self.deepnorm:
            self.encoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.encoder_normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0


def MultiwayWrapper(args, module, dim=1):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    def __init__(
        self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x

if torch.cuda.is_available():
    try:
        if torch.cuda.get_device_capability()[0] > 7:
            from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func

            def flash_attn_func(q, k, v, dropout=0.0, bias=None, softmax_scale=None, is_causal=False):
                assert bias is None
                attn, lse, _ = _flash_attn_func(q, k, v, dropout_p=dropout, softmax_scale=softmax_scale, causal=is_causal, return_attn_probs=True)
                return attn, lse

        else:
            from xformers.ops.fmha import (
                cutlass,
                Inputs,
                Context,
                _memory_efficient_attention_forward_requires_grad,
                _memory_efficient_attention_backward,
                LowerTriangularMask,
            )

            class FlashAttnFunc(torch.autograd.Function):
                @staticmethod
                # type: ignore
                def forward(ctx, q, k, v, dropout=0.0, bias=None, softmax_scale=None, is_causal=False):
                    if is_causal:
                        assert bias is None
                        attn_bias = LowerTriangularMask()
                    else:
                        attn_bias = bias

                    inp = Inputs(
                        query=q,
                        key=k,
                        value=v,
                        attn_bias=attn_bias,
                        p=dropout,
                        scale=softmax_scale,
                    )
                    op_fw = cutlass.FwOp
                    op_bw = cutlass.BwOp

                    out, op_ctx = _memory_efficient_attention_forward_requires_grad(
                        inp=inp, op=op_fw
                    )

                    # Saving attn_bias is a bit complicated, as the
                    # torch part should go in `save_for_backward`
                    if isinstance(inp.attn_bias, torch.Tensor):
                        attn_bias_tensor = inp.attn_bias
                        attn_bias_ctx = None
                    else:
                        attn_bias_tensor = None
                        attn_bias_ctx = inp.attn_bias

                    ctx.save_for_backward(
                        inp.query,
                        inp.key,
                        inp.value,
                        op_ctx.out,
                        op_ctx.lse,
                    )
                    ctx.rng_state = op_ctx.rng_state
                    ctx.attn_bias_tensor = attn_bias_tensor
                    if op_ctx.op_bw is not None:
                        if op_bw is not None and op_bw is not op_ctx.op_bw:
                            raise ValueError(
                                f"Specified op_bw={op_bw.NAME}, but forward op "
                                f"can only run with op_bw={op_ctx.op_bw.NAME}. Please set op_bw=None."
                            )
                        op_bw = op_ctx.op_bw
                    ctx.op_fw = op_fw
                    ctx.op_bw = op_bw
                    ctx.p = inp.p

                    ctx.scale = inp.scale
                    ctx.attn_bias_ctx = attn_bias_ctx
                    return out, op_ctx.lse

                @staticmethod
                def deserialize_bias(
                    attn_bias_ctx, attn_bias_tensor: Optional[torch.Tensor]
                ) -> Any:
                    if attn_bias_tensor is None:
                        return attn_bias_ctx
                    return attn_bias_tensor

                @classmethod
                @torch.autograd.function.once_differentiable
                def backward(cls, ctx, grad, dlse):
                    # Re-create context
                    query, key, value, out, lse = ctx.saved_tensors
                    attn_bias_tensor = ctx.attn_bias_tensor
                    rng_state = ctx.rng_state
                    inp = Inputs(
                        query=query,
                        key=key,
                        value=value,
                        attn_bias=cls.deserialize_bias(ctx.attn_bias_ctx, attn_bias_tensor),
                        p=ctx.p,
                        scale=ctx.scale,
                    )
                    op_ctx = Context(
                        lse=lse,
                        out=out,
                        rng_state=rng_state,
                    )
                    grads = _memory_efficient_attention_backward(
                        ctx=op_ctx, inp=inp, grad=grad, op=ctx.op_bw
                    )
                    return grads.dq, grads.dk, grads.dv, None, grads.db, None, None
            
            flash_attn_func = FlashAttnFunc.apply
    except ModuleNotFoundError:
        flash_attn_func = None
else:
    flash_attn_func = None

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.out_proj = MultiwayWrapper(
            args, nn.Linear(embed_dim, embed_dim, bias=True)
        )
        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, args.xpos_scale_base)
            if args.xpos_rel_pos and self.self_attention
            else None
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def attention_ops(self, q, k, v, key_padding_mask=None, attn_mask=None, rel_pos=None, is_causal=False):
        if not self.args.flash_attention:
            q *= self.scaling
            attn_weights = torch.bmm(q, k.transpose(1, 2))

            if attn_mask is not None:
                attn_weights = torch.nan_to_num(attn_weights)
                attn_mask = attn_mask.unsqueeze(0)
                attn_weights += attn_mask

            if key_padding_mask is not None:
                attn_weights = rearrange(attn_weights, '(b h) t s -> b h t s', h=self.num_heads)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                attn_weights = rearrange(attn_weights, 'b h t s -> (b h) t s')

            if rel_pos is not None:
                rel_pos = rel_pos.view(attn_weights.size())
                attn_weights = attn_weights + rel_pos

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
                attn_weights
            )
            attn_probs = self.dropout_module(attn_weights)

            attn = torch.bmm(attn_probs, v)
            attn = rearrange(attn, '(b h) l d -> b l (h d)', h=self.num_heads)
        else:
            assert flash_attn_func is not None
            assert rel_pos is None
            q = rearrange(q, '(b h) l d -> b l h d', h=self.num_heads)
            k = rearrange(k, '(b h) l d -> b l h d', h=self.num_heads)
            v = rearrange(v, '(b h) l d -> b l h d', h=self.num_heads)
            # flash_attn_func(q, k, v, dropout=0.0, bias=None, softmax_scale=None, is_causal=False)
            attn, lse = flash_attn_func(q, k, v, self.dropout, attn_mask, None, is_causal)
            # attn, lse = flash_attn_func(q, k, v, dropout=self.dropout, is_causal=is_causal)
            attn = rearrange(attn, 'b l h d -> b l (h d)')
            attn_weights = lse[:, :, :attn.size(1)]

        return attn, attn_weights

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
        is_causal=False,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> (b h) l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> (b h) l d', h=self.num_heads)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None and not is_first_step:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn, attn_weights = self.attention_ops(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rel_pos=rel_pos, is_causal=is_causal)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)

        return attn, attn_weights
    

class DilatedAttention(MultiheadAttention):

    def dense_to_sparse(self, x, ratio):
        length = x.size(1)
        padding = padding_to_multiple_of(length, ratio)
        head_padding = padding_to_multiple_of(self.num_heads, ratio)

        if padding > 0 or head_padding > 0:
            x = F.pad(x, (0, 0, 0, head_padding, 0, padding), value = 0.)

        x = rearrange(x, 'b (l r1) (r2 h) d -> b l h d r1 r2', r1=ratio, r2=ratio)
        x = torch.diagonal(x, offset=0, dim1=4, dim2=5)
        x = rearrange(x, 'b l h d r -> b l (r h) d')
        
        if head_padding > 0:
            x = x[:, :, :self.num_heads]

        return x

    def sparse_to_dense(self, out, lse, ratio):
        head_padding = padding_to_multiple_of(self.num_heads, ratio)

        if head_padding > 0:
            out = F.pad(out, (0, 0, 0, head_padding), value = 0.)
            lse = F.pad(lse, (0, 0, 0, head_padding), value = -1e8)

        out = rearrange(out, 'b l (r h) d -> b l h d r', r=ratio)
        out = torch.diag_embed(out, offset=0, dim1=4, dim2=5)
        out = rearrange(out, 'b l h d r1 r2 -> b (r2 h) (l r1) d', r1=ratio, r2=ratio)

        lse = rearrange(lse, 'b (r h) l -> b l h r', r=ratio)
        lse = torch.diag_embed(lse, offset=0, dim1=3, dim2=4)
        lse = lse.masked_fill_(lse==0, -1e8)
        lse = rearrange(lse, 'b l h r1 r2 -> b (r2 h) (l r1) 1', r1=ratio, r2=ratio)

        if head_padding > 0:
            out = out[:, :self.num_heads]
            lse = lse[:, :self.num_heads]

        return out, lse

    def gather_kv(self, x, sl, seq_len, is_causal=True):
        bsz = x.size(0)
        assert sl % seq_len == 0
        num_rank_per_segment = sl // seq_len

        x = all_gather_func(x)
        current_rank = get_data_parallel_rank()
        x = rearrange(x, '(w b) l h d -> w b l h d', b=bsz)
        
        if is_causal:
            if current_rank > 0:
                x = x[:current_rank]
            else:
                x = x[:1] * 0
        
        current_segment = current_rank // num_rank_per_segment * num_rank_per_segment
        x = x[current_segment:current_segment+num_rank_per_segment]

        x = rearrange(x, 'w b l h d -> b (w l) h d')
        return x
    
    def gathering(self, x, dr, sl, is_causal=True, offset=0, is_kv=False, seq_parall=True):

        curr_x = x
        if offset > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, offset % sl, 0), value=0.)
        seq_len = curr_x.size(1)
        should_gather_kv = is_kv and (get_data_parallel_world_size() > 1) and (sl > seq_len) and seq_parall
        _sl = sl
        sl = min(sl, seq_len)
        padding = padding_to_multiple_of(seq_len, sl)

        if padding > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, 0, padding), value = 0.)

        curr_x = rearrange(curr_x, 'b (n g) h d -> (b n) g h d', g=sl)
        curr_x = self.dense_to_sparse(curr_x, dr)

        if should_gather_kv:
            curr_x = self.gather_kv(curr_x, _sl, seq_len, is_causal)

        curr_x = rearrange(curr_x, 'b l h d -> (b h) l d')
        
        return curr_x

    def scattering(self, outs, lses, seq_len, bsz, offset=0):
        assert len(outs) == len(lses)
        assert len(outs) % len(self.args.dilated_ratio) == 0
        all_outs, all_lses = [], []
        drs = self.args.dilated_ratio
        if len(outs) > len(drs):
            drs = drs * (len(outs) // len(drs))

        for dr, o, lse in zip(drs, outs, lses):
            o = rearrange(o, 'b l (h d) -> b l h d', h=self.num_heads)
            o, lse = self.sparse_to_dense(o, lse, dr)
            o = rearrange(o, '(b n) h g d -> (b h) (n g) d', b=bsz)
            lse = rearrange(lse, '(b n) h g 1 -> (b h) (n g) 1', b=bsz)
            o = o[:, offset:offset+seq_len]
            lse = lse[:, offset:offset+seq_len]

            all_outs.append(o)
            all_lses.append(lse)

        with torch.no_grad():
            max_lse = torch.stack(all_lses, dim=0)
            max_lse = max_lse.max(0)[0]
            all_lses = [torch.exp(lse-max_lse) for lse in all_lses]
            lse_sum = torch.stack(all_lses, dim=0).sum(0)
            all_lses = [lse / lse_sum for lse in all_lses]

        out = 0
        for o, lse in zip(all_outs, all_lses):
            out += o * lse.type_as(o)
        out = rearrange(out, '(b h) l d -> b l (h d)', h=self.num_heads)

        return out

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
        is_causal=False,
    ):
        assert self.args.flash_attention
        assert rel_pos is None
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> (b h) l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> (b h) l d', h=self.num_heads)

        if incremental_state is not None and not is_first_step:
            offset = src_len - 1
        else:
            offset = 0

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None and not is_first_step:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)
        
        q = rearrange(q, '(b h) l d -> b l h d', h=self.num_heads)
        k = rearrange(k, '(b h) l d -> b l h d', h=self.num_heads)
        v = rearrange(v, '(b h) l d -> b l h d', h=self.num_heads)

        outs, lses = [], []
        for sl, dr in zip(self.args.segment_length, self.args.dilated_ratio):
            ki = self.gathering(k, dr, sl, is_causal=is_causal, offset=0, is_kv=True, seq_parall=self.args.seq_parallel)
            vi = self.gathering(v, dr, sl, is_causal=is_causal, offset=0, is_kv=True, seq_parall=self.args.seq_parallel)
            qi = self.gathering(q, dr, sl, is_causal=is_causal, offset=offset, is_kv=False, seq_parall=self.args.seq_parallel)

            out, lse = self.attention_ops(qi, ki, vi, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rel_pos=rel_pos, is_causal=is_causal)

            outs.append(out)
            lses.append(lse)

        attn = self.scattering(outs, lses, tgt_len, bsz, offset=offset)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)

        return attn, None
    

class LongNetEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, 
                 dilated_ratio=[1, 2, 4],
                 segment_length=[512, 1024, 2048]):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        args = EncoderConfig(
            encoder_embed_dim=embed_dim,
            encoder_attention_heads=num_heads,
            dilated_ratio=str(dilated_ratio),
            segment_length=str(segment_length),
            flash_attention=True,
            seq_parallel=False,
        )
        self.self_attn = DilatedAttention(args, embed_dim, num_heads, self_attention=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        attn_out, _ = self.self_attn(
            self.ln1(x), self.ln1(x), self.ln1(x),
            is_causal=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x
    
class LongNet(nn.Module):
    def __init__(self, embed_dim, llm_hidden_size, num_heads, num_layers=3, dilated_ratio=[1, 2, 4],
                 segment_length=[512, 1024, 2048], num_deepstack=3):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([LongNetEncoderLayer(embed_dim, num_heads, dilated_ratio, segment_length) for _ in range(num_layers)])

        # 3. 初始化 Token Merger
        # 使用自适应平均池化，将序列长度强制统一为 config.histo_token_num (如 100)
        self.feature_projector = nn.Linear(embed_dim, llm_hidden_size)
        self.intermediate_projectors = nn.ModuleList([
            nn.Linear(embed_dim, llm_hidden_size) for _ in range(num_deepstack)
        ])

    def forward(self, x, return_intermediate=False):
        intermediates = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_intermediate and i < self.num_layers:
                intermediates.append(self.intermediate_projectors[i](x))
        x = self.feature_projector(x)
        if return_intermediate:
            return x, intermediates
        return x
    

if __name__ == "__main__":
    x = torch.randn(1, 10000, 1280, device=0)
    net = LongNet(1280, 128, 8, 4).to(0)
    x, y = net(x, True)
    import pdb; pdb.set_trace()
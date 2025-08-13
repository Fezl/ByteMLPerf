import sys
import pathlib
from functools import partial
import torch

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}


try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    # https://github.com/Dao-AILab/flash-attention
    class FA2Op(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["flash_attn_v2"]

            self.input_tensor_info.update({
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.num_tokens // self.batch_size, self.q_head_num, self.head_dim], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name()
                ), 
                "k_cache": OpTensorInfo(
                    shape=[self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim], 
                    dtype=self.cache_torch_dtype, 
                    device=self.backend.get_torch_device_name(), 
                    creator=torch.empty
                ), 
                "v_cache": OpTensorInfo(
                    shape=[self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim], 
                    dtype=self.cache_torch_dtype, 
                    device=self.backend.get_torch_device_name(), 
                    creator=torch.empty
                ),
            })

            # currently not support prefill_session_cache mode
            # cause not support different q_lens
            if self.mode in ["prefill_session_cache"]:
                raise NotImplementedError("not support prefill_session_cache")

            # currently not support c8
            if self.dtype != self.cache_dtype:
                raise NotImplementedError("not support q_dtype != cache_dtype")


        def flash_attention_run(self, tensor_mapping):
            # get pre-allocated tensors
            q = tensor_mapping["q"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            cache_lens = tensor_mapping["cache_lens"]
            cache_slot_ids = tensor_mapping["cache_slot_ids"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            k_scale = tensor_mapping.get("k_scale", None)
            v_scale = tensor_mapping.get("v_scale", None)

            # ignore k_scale/v_scale
            if self.mode == "prefill" and self.cache_len == 0:
                # q: [1, q_seq_len, q_head_num, head_dim]
                # k: [1, q_seq_len, kv_head_num, head_dim]
                # v: [1, q_seq_len, kv_head_num, head_dim]
                out = flash_attn_func(
                    q, k_cache, v_cache, 
                    softmax_scale=self.softmax_scale, 
                    causal=True
                )
            else:
                # q: [batch_size, q_seq_len, q_head_num, head_dim]
                # k: [batch_size, max_kv_len, kv_head_num, head_dim]
                # v: [batch_size, max_kv_len, kv_head_num, head_dim]
                out = flash_attn_with_kvcache(
                    q=q, 
                    k_cache=k_cache, 
                    v_cache=v_cache, 
                    cache_seqlens=cache_lens, 
                    cache_batch_idx=cache_slot_ids, 
                    softmax_scale=self.softmax_scale, 
                    causal=True
                )
            return out
    
    OP_MAPPING["flash_attn_v2"] = FA2Op
except:
    pass


try:
    from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache

    # https://github.com/Dao-AILab/flash-attention
    class FA3Op(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["flash_attn_v3"]

            self.input_tensor_info.update({
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.num_tokens // self.batch_size, self.q_head_num, self.head_dim], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name()
                ), 
                "k_cache": OpTensorInfo(
                    shape=[self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim], 
                    dtype=self.cache_torch_dtype, 
                    device=self.backend.get_torch_device_name(), 
                    creator=torch.empty
                ), 
                "v_cache": OpTensorInfo(
                    shape=[self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim], 
                    dtype=self.cache_torch_dtype, 
                    device=self.backend.get_torch_device_name(), 
                    creator=torch.empty
                ),
            })

            # currently not support prefill_session_cache mode
            # cause not support different q_lens
            if self.mode in ["prefill_session_cache"]:
                raise NotImplementedError("not support prefill_session_cache")

            # currently not support c8
            if self.dtype != self.cache_dtype:
                raise NotImplementedError("not support q_dtype != cache_dtype")


        def flash_attention_run(self, tensor_mapping):
            # get pre-allocated tensors
            q = tensor_mapping["q"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            cache_lens = tensor_mapping["cache_lens"]
            cache_slot_ids = tensor_mapping["cache_slot_ids"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            k_scale = tensor_mapping.get("k_scale", None)
            v_scale = tensor_mapping.get("v_scale", None)

            # ignore k_scale/v_scale
            if self.mode == "prefill" and self.cache_len == 0:
                # q: [1, q_seq_len, q_head_num, head_dim]
                # k: [1, q_seq_len, kv_head_num, head_dim]
                # v: [1, q_seq_len, kv_head_num, head_dim]
                out = flash_attn_func(
                    q, k_cache, v_cache, 
                    softmax_scale=self.softmax_scale, 
                    causal=True
                )
            else:
                # q: [batch_size, q_seq_len, q_head_num, head_dim]
                # k: [batch_size, max_kv_len, kv_head_num, head_dim]
                # v: [batch_size, max_kv_len, kv_head_num, head_dim]
                out = flash_attn_with_kvcache(
                    q=q, 
                    k_cache=k_cache, 
                    v_cache=v_cache, 
                    cache_seqlens=cache_lens, 
                    cache_batch_idx=cache_slot_ids, 
                    softmax_scale=self.softmax_scale, 
                    causal=True
                )
            return out
    
    OP_MAPPING["flash_attn_v3"] = FA3Op
except:


# import math
# import torch
# import torch_npu  # pip install torch-npu ；并确保模型/张量 .npu()
# # 1) 准备 Q/K/V：先在 host 侧完成 RoPE（如有）与 GQA 映射，再放到 NPU
# # q: (B, Sq, Hq, D), k_cache/v_cache: (B, Sk_max, Hkv, D)
# # 当前 step 的有效长度
# seqlen_q = q.size(1)        # 增量解码典型是 1
# seqlen_kv = cache_lens      # 你的 (B,) 有效长度张量或 Python list

# # 2) 取出本 step 需要参与注意力的 K/V 视图（含历史缓存）
# # 这里假设 k_cache/v_cache 已经就地写入了最新 token（或在调用后写入也可）
# k = k_cache[:, :seqlen_kv_max, ...]  # 若每个batch不同长，传 actual_seq_lengths_kv 去指示
# v = v_cache[:, :seqlen_kv_max, ...]

# # 3) 调整到 BNSD 布局
# q_bnsd = q.permute(0, 2, 1, 3).contiguous()   # (B, Hq, Sq, D)
# k_bnsd = k.permute(0, 2, 1, 3).contiguous()   # (B, Hkv, Sk, D)
# v_bnsd = v.permute(0, 2, 1, 3).contiguous()   # (B, Hkv, Sk, D)

# # 4) 窗口/因果：自回归通常 next_tokens=0；若要滑窗(left,right)，映射到 pre_tokens/next_tokens
# L, R = window_size  # 你的参数；(-1,-1) 表示全局
# if L is None or L < 0: pre_tokens = 2**31 - 1
# else: pre_tokens = L
# if R is None or R < 0: next_tokens = 2**31 - 1
# else: next_tokens = R
# if causal:
#     next_tokens = 0  # 严格因果

# # 5) 有效长度（注意 FIA 需要 host 侧 IntArray；可以传单值或按 batch 列表）
# act_len_q  = [int(seqlen_q)] if isinstance(seqlen_q, int) else [int(seqlen_q.item())]
# act_len_kv = [int(x) for x in (seqlen_kv.tolist() if torch.is_tensor(seqlen_kv) else seqlen_kv)]
# seqlen_kv_max = max(act_len_kv)

# # 6) 比例因子
# scale = 1.0 / math.sqrt(q.size(-1)) if softmax_scale is None else float(softmax_scale)

# # 7) 调用 FIA
# out_bnsd, _lse = torch_npu.npu_fused_infer_attention_score(
#     q_bnsd, k_bnsd, v_bnsd,
#     actual_seq_lengths=act_len_q,
#     actual_seq_lengths_kv=act_len_kv,
#     num_heads=q_bnsd.size(1),
#     num_key_value_heads=k_bnsd.size(1),
#     input_layout="BNSD",
#     scale_value=scale,
#     pre_tokens=pre_tokens,
#     next_tokens=next_tokens,
#     # 可选项：atten_mask=...（bool/uint8），block_table=...（分页KV）
# )

# # 8) 还原到原来的 (B, Sq, Hq, D)
# out = out_bnsd.permute(0, 2, 1, 3).contiguous()

    pass

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
    pass


# import math
# import torch

# def npu_flash_attn_with_kvcache(
#     q,
#     k_cache,
#     v_cache,
#     cache_seqlens,                  # int / list[int] / Tensor[int32] (B,)
#     cache_batch_idx=None,           # Tensor[int32] (B,)，可为 None
#     softmax_scale=None,
#     causal=True,
#     window_size=(-1, -1),           # (-1,-1) 表示全局注意力
#     block_table=None,               # 分页KV可选
# ):
#     """
#     昇腾NPU替代 flash_attn_with_kvcache 的最小实现，基于 torch_npu.npu_fused_infer_attention_score。
#     输入输出与原函数保持相同的主维度/布局：
#         in:  q: (B, Sq, Hq, D)
#              k_cache/v_cache: (B, Sk_max, Hkv, D)
#         out: (B, Sq, Hq, D)
#     """
#     import torch_npu  # 确保已安装并在NPU环境

#     # ---------- 基本检查 ----------
#     assert q.dim() == 4, "q should be (B, Sq, Hq, D)"
#     assert k_cache.dim() == 4 and v_cache.dim() == 4, "k_cache/v_cache should be (B, Sk_max, Hkv, D)"
#     B, Sq, Hq, D = q.shape

#     # 设备与连续性
#     dev = q.device
#     if dev.type != "npu":
#         raise RuntimeError("q/k_cache/v_cache 必须在 NPU 上，请先调用 .to('npu')/.npu()")
#     q = q.contiguous()
#     k_cache = k_cache.contiguous()
#     v_cache = v_cache.contiguous()

#     # ---------- 根据 cache_batch_idx 选择对应的 cache 行 ----------
#     if cache_batch_idx is not None:
#         # 允许 cache_slot_id 与当前 batch 顺序不同的场景
#         cache_batch_idx = cache_batch_idx.to(dtype=torch.int64, device=dev)
#         k_cache = torch.index_select(k_cache, 0, cache_batch_idx).contiguous()
#         v_cache = torch.index_select(v_cache, 0, cache_batch_idx).contiguous()

#     # ---------- 计算每个样本的 KV 有效长度，并得到 seqlen_kv_max ----------
#     if isinstance(cache_seqlens, int):
#         act_len_kv = [cache_seqlens] * B
#     elif isinstance(cache_seqlens, (list, tuple)):
#         assert len(cache_seqlens) == B, "len(cache_seqlens) must equal batch size"
#         act_len_kv = [int(x) for x in cache_seqlens]
#     elif torch.is_tensor(cache_seqlens):
#         if cache_seqlens.dim() == 0:
#             act_len_kv = [int(cache_seqlens.item())] * B
#         else:
#             assert cache_seqlens.numel() == B, "cache_seqlens tensor must have shape (B,)"
#             act_len_kv = cache_seqlens.to("cpu", non_blocking=True).tolist()
#             act_len_kv = [int(x) for x in act_len_kv]
#     else:
#         raise TypeError("cache_seqlens must be int / list[int] / 1D Tensor[int]")

#     # ✅ 先初始化再使用，避免未定义
#     seqlen_kv_max = max(act_len_kv) if len(act_len_kv) > 0 else k_cache.size(1)

#     # 截取到本步实际需要参与注意力的 KV（历史 + 当前步，如果你已写入的话）
#     k = k_cache[:, :seqlen_kv_max, :, :].contiguous()
#     v = v_cache[:, :seqlen_kv_max, :, :].contiguous()

#     # ---------- 布局从 (B, S, H, D) 转为 FIA 需要的 BNSD ----------
#     # q: (B, Sq, Hq, D) -> (B, Hq, Sq, D)
#     # k/v: (B, Sk, Hkv, D) -> (B, Hkv, Sk, D)
#     q_bnsd = q.permute(0, 2, 1, 3).contiguous()
#     k_bnsd = k.permute(0, 2, 1, 3).contiguous()
#     v_bnsd = v.permute(0, 2, 1, 3).contiguous()

#     Hkv = k_bnsd.size(1)

#     # ---------- 窗口与因果设置 ----------
#     def _inf():
#         # FIA 接受超大整数表示“无限窗口”
#         return 2**31 - 1

#     if window_size is None or window_size == (-1, -1):
#         pre_tokens, next_tokens = _inf(), _inf()
#     else:
#         L, R = window_size
#         pre_tokens = _inf() if (L is None or L < 0) else int(L)
#         next_tokens = _inf() if (R is None or R < 0) else int(R)

#     if causal:
#         # 因果解码：禁止看未来
#         next_tokens = 0

#     # ---------- 有效长度（按 batch 传） ----------
#     act_len_q = [Sq] * B  # 增量解码通常是 1，但这里按实际 Sq 设置

#     # ---------- softmax scale ----------
#     scale_value = (1.0 / math.sqrt(D)) if (softmax_scale is None) else float(softmax_scale)

#     # ---------- 组装并调用 NPU 融合注意力 ----------
#     fia_kwargs = dict(
#         actual_seq_lengths=act_len_q,
#         actual_seq_lengths_kv=act_len_kv,
#         num_heads=Hq,
#         num_key_value_heads=Hkv,
#         input_layout="BNSD",
#         scale_value=scale_value,
#         pre_tokens=pre_tokens,
#         next_tokens=next_tokens,
#     )
#     if block_table is not None:
#         fia_kwargs["block_table"] = block_table  # 若你使用分页KV

#     # out_bnsd: (B, Hq, Sq, D)
#     out_bnsd, _lse = torch_npu.npu_fused_infer_attention_score(
#         q_bnsd, k_bnsd, v_bnsd, **fia_kwargs
#     )

#     # ---------- 还原到原始布局 (B, Sq, Hq, D) ----------
#     out = out_bnsd.permute(0, 2, 1, 3).contiguous()
#     return out

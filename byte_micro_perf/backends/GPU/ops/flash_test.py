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
    # Prefer native npu operator if available
    import torch_npu
except Exception:
    torch_npu = None

try:
    # keep original flash_attn imports as fallback
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

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
            block_table = tensor_mapping.get("block_table", None)

            # If native NPU operator available, use it. Otherwise fallback to flash_attn_with_kvcache.
            if torch_npu is None:
                # fallback to existing implementation
                if self.mode == "prefill" and self.cache_len == 0:
                    out = flash_attn_func(
                        q, k_cache, v_cache, 
                        softmax_scale=self.softmax_scale, 
                        causal=True
                    )
                else:
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

            # Map inputs to npu_incre_flash_attention
            # Notes / assumptions made here:
            #  - q, k_cache, v_cache shapes are expected: (batch, seqlen, nheads, headdim)
            #  - choose input_layout for 4D tensors as "BSNDBSNDBSND" (BSND style). Adjust if your
            #    environment expects the other variant (BNSD...)
            #  - causal/local attention, rotary embeddings and ALiBi are NOT auto-handled here.
            #    If you need those features, build an appropriate atten_mask / pse_shift / block_table
            #    and pass them in tensor_mapping.

            try:
                # prefill (no previously cached tokens) : treat k_cache/v_cache as current k/v
                if self.mode == "prefill" and self.cache_len == 0:
                    out = torch_npu.npu_incre_flash_attention(
                        q,                # query
                        k_cache,          # key
                        v_cache,          # value
                        num_heads=self.q_head_num,
                        num_key_value_heads=self.kv_head_num,
                        input_layout="BSNDBSNDBSND",
                        scale_value=self.softmax_scale,
                    )                else:
                    # incremental cached scenario
                    # pass actual_seq_lengths so the operator knows per-batch valid KV length
                    actual_seq_lengths = None
                    if cache_lens is not None:
                        try:
                            actual_seq_lengths = cache_lens.cpu().numpy().tolist()
                        except Exception:
                            actual_seq_lengths = cache_lens

                    # prefer an explicit block_table if provided by the caller
                    bt = block_table if block_table is not None else None

                    # Adapt cache_slot_ids -> block_table for paged KV cache scenarios when
                    # caller provided a slot mapping but not an explicit block_table.
                    # Heuristic/assumptions used here (best-effort):
                    #  - cache_slot_ids represents the starting block index for each batch in the
                    #    global paged KV cache (dtype int or int tensor of shape [batch]).
                    #  - block_size defaults to 256 unless user specifies self.block_size.
                    #  - We construct a rectangular block_table with the minimal number of blocks
                    #    needed for the largest actual_seq_lengths across the batch.
                    if bt is None and cache_slot_ids is not None and actual_seq_lengths is not None:
                        try:
                            bs = getattr(self, "block_size", 256) or 256
                            # ensure Python int
                            block_size = int(bs)

                            # compute blocks needed per the longest KV length
                            max_seq = int(max(actual_seq_lengths)) if isinstance(actual_seq_lengths, (list, tuple)) else int(actual_seq_lengths)
                            max_blocks = (max_seq + block_size - 1) // block_size
                            if max_blocks <= 0:
                                max_blocks = 1

                            # normalize cache_slot_ids to CPU list of ints
                            if isinstance(cache_slot_ids, torch.Tensor):
                                csids = cache_slot_ids.cpu().tolist()
                            else:
                                csids = list(cache_slot_ids)

                            # build block_table: shape (batch, max_blocks), dtype int32
                            bt_tensor = torch.empty((len(csids), max_blocks), dtype=torch.int32, device=q.device)
                            for i, sid in enumerate(csids):
                                start = int(sid)
                                # fill sequential block indices starting from start
                                row = [start + j for j in range(max_blocks)]
                                bt_tensor[i] = torch.tensor(row, dtype=torch.int32, device=q.device)

                            bt = bt_tensor
                        except Exception:
                            # if adapter fails, leave bt as None and let operator attempt without block_table
                            bt = None

                    out = torch_npu.npu_incre_flash_attention(
                        q,
                        k_cache,
                        v_cache,
                        actual_seq_lengths=actual_seq_lengths,
                        block_table=bt,
                        num_heads=self.q_head_num,
                        num_key_value_heads=self.kv_head_num,
                        input_layout="BSNDBSNDBSND",
                        scale_value=self.softmax_scale,
                    )

            except Exception as e:
                # If native call fails, fall back to python implementation to preserve behavior
                # and surfacing the error for debugging.
                print("Warning: npu_incre_flash_attention call failed, falling back to flash_attn_with_kvcache:", e)
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
except Exception:
    # if import fails just skip this provider
    pass

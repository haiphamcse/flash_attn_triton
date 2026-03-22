import os
import torch
import torch.nn.functional as F
import math
import random
import numpy as np

import triton
import triton.language as tl

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        bs, d1, d = Q.shape
        _, d2, _ = K.shape

        device = Q.device
        dtype = Q.dtype
        sqrt_d = math.sqrt(d)
        block_size = 16

        Td1 = math.ceil(d1 / block_size)
        Td2 = math.ceil(d2 / block_size)

        O = torch.zeros((bs, d1, d), device=device)
        L = torch.zeros((bs, d1), device=device)

        for j in range(Td1):
            start_j = j * block_size
            end_j = min((j + 1) * block_size, d1)
            block_q = end_j - start_j  # actual query block size (may be < block_size)

            m_prev = torch.full((bs, block_q), float("-inf"), device=device)
            l_prev = torch.zeros((bs, block_q), device=device)
            o_prev = torch.zeros((bs, block_q, d), device=device)

            for i in range(Td2):
                start_i = i * block_size
                end_i = min((i + 1) * block_size, d2)

                if start_i >= end_i:
                    continue

                # Early exit: all keys in this block are beyond all query positions
                if is_causal and start_i >= end_j:
                    break

                K_block = K[:, start_i:end_i]
                V_block = V[:, start_i:end_i]

                x_curr = torch.einsum("bkd,bnd->bkn", Q[:, start_j:end_j], K_block) / sqrt_d

                # Causal mask for partially overlapping blocks
                if is_causal:
                    q_idx = torch.arange(start_j, end_j, device=device).unsqueeze(-1)
                    k_idx = torch.arange(start_i, end_i, device=device).unsqueeze(0)
                    mask = q_idx >= k_idx  # (block_q, block_k)
                    x_curr = torch.where(mask.unsqueeze(0), x_curr,
                                        torch.tensor(float("-inf"), device=device, dtype=dtype))

                block_max = x_curr.max(dim=-1).values
                m_curr = torch.maximum(m_prev, block_max)

                exp_m_prev = torch.exp(m_prev - m_curr)
                exp_x_m = torch.exp(x_curr - m_curr[:, :, None])

                l_curr = l_prev * exp_m_prev + exp_x_m.sum(dim=-1)

                o_prev = o_prev * exp_m_prev[:, :, None] + torch.einsum("bkn,bnd->bkd", exp_x_m, V_block)
                m_prev = m_curr
                l_prev = l_curr

            O[:, start_j:end_j] = o_prev / l_prev[:, :, None]
            L[:, start_j:end_j] = m_prev + torch.log(l_prev + 1e-20)

        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        ctx.sqrt_d = sqrt_d

        return O

    @staticmethod
    def backward(ctx, dO):
        """
        Attention backward pass.

        Args:
            ctx: Context object with saved tensors
            dO: Gradient of loss w.r.t. output

        Returns:
            dQ: Gradient w.r.t. Q
            dK: Gradient w.r.t. K
            dV: Gradient w.r.t. V
            None: No gradient for is_causal
        """
        Q, K, V, L, O = ctx.saved_tensors
        dQ, dK, dV, _ = attention_backward_impl(
            Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal
        )
        return dQ, dK, dV, None


def attention_backward_impl(Q, K, V, L, O, dO, sqrt_d, is_causal):
    """
    Backward pass implementation for Flash Attention.

    Uses standard attention gradient formulas with recomputation of P
    from saved log-sum-exp values L.

    Args:
        Q: Query tensor of shape (batch, seq_q, d)
        K: Key tensor of shape (batch, seq_k, d)
        V: Value tensor of shape (batch, seq_k, d)
        L: Log-sum-exp values from forward pass, shape (batch, seq_q)
        O: Output from forward pass, shape (batch, seq_q, d)
        dO: Gradient of loss w.r.t. output, shape (batch, seq_q, d)
        sqrt_d: Square root of head dimension (for scaling)
        is_causal: Whether to apply causal masking

    Returns:
        dQ: Gradient w.r.t. Q
        dK: Gradient w.r.t. K
        dV: Gradient w.r.t. V
        None: Placeholder for compatibility
    """
    # Your code here
    #
    D = torch.sum(O * dO, dim=-1) # [b, seq_q] ?
    S = torch.einsum("bnd,bkd->bnk",Q,K) / sqrt_d
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(Q.shape[1], K.shape[1], device=Q.device, dtype=torch.bool), diagonal=1
        )
        S = S.masked_fill(causal_mask[None], float('-inf'))
    P = torch.exp(S - L[..., None]) # [b, seq_q, seq_k]
    dV = torch.einsum("bnk,bnd->bkd", P, dO) # [b, seq_k, d]
    dP = torch.einsum("bnd,bkd->bnk", dO, V) # [b, seq_q, seq_k]
    dS = P * (dP - D[..., None]) # [b, seq_q, seq_k]
    dQ = torch.einsum("bnk,bkd->bnd", dS, K)/ sqrt_d # [b, seq_q, d]
    dK = torch.einsum("bnk,bnd->bkd", dS, Q) / sqrt_d
    return dQ, dK, dV, None

    # Steps:
    # 1. Compute D = rowsum(O ⊙ dO)
    # 2. Recompute S = Q @ K^T / sqrt(d)
    # 3. Apply causal mask if is_causal
    # 4. Recompute P = exp(S - L)
    # 5. Compute dV = P^T @ dO
    # 6. Compute dP = dO @ V^T
    # 7. Compute dS = P ⊙ (dP - D)
    # 8. Compute dQ = dS @ K / sqrt(d)
    # 9. Compute dK = dS^T @ Q / sqrt(d)


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    is_causal: tl.constexpr,
):
    """Flash Attention forward kernel using online softmax algorithm."""
    # Get program IDs
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Create block pointers for this batch and query tile
    Q_ptr = Q_ptr + pid_b * stride_qb
    # Create block pointer for queries (fixed position)
    Q_block = tl.make_block_ptr(
        Q_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )

    V_ptr = V_ptr + pid_b * stride_vb
    V_block = tl.make_block_ptr(
        V_ptr,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D),
        order=(1,0)
    )
    K_ptr = K_ptr + pid_b * stride_kb
    K_block = tl.make_block_ptr(
        K_ptr,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D),
        order=(1,0)
    )

    output_block = tl.make_block_ptr(
        O_ptr + pid_b * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),

    )

    L_block = tl.make_block_ptr(
        L_ptr + pid_b * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(pid_q * BLOCK_Q,),
        block_shape=(BLOCK_Q,),
        order=(0,),
   
    )

    # sqrt_d = tl.sqrt(D)
    num_blocks = tl.cdiv(N_KEYS, BLOCK_K) # number of chunks to process BLOCK_K keys simultaneously

    # M = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    # L = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O = tl.zeros((BLOCK_Q, D), dtype=tl.float32) # [Bq, d]
    q = tl.load(Q_block, boundary_check=(0,1), padding_option='zero') # [Bq, d]
    q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_idx < N_QUERIES

    m_prev = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    for _ in range(num_blocks):
      ###
      k = tl.load(K_block, boundary_check=(0,1), padding_option='zero') # [Bk, d]
      v = tl.load(V_block, boundary_check=(0,1), padding_option='zero') # [Bk, d]

      x = tl.dot(q, tl.trans(k)) / scale # [Bq, Bk]
      
      k_start = _ * BLOCK_K
      k_idx = k_start + tl.arange(0, BLOCK_K)
      k_mask = k_idx < N_KEYS
      x = tl.where(k_mask[None, :], x, float("-inf"))
      x = tl.where(q_mask[:, None], x, float("-inf"))

      if is_causal:
          mask = q_idx[:, None] >= k_idx[None, :]
          x = tl.where(mask, x, float('-inf'))
          
      block_max = tl.max(x, axis=-1) # [Bq,]
      m_curr = tl.maximum(m_prev, block_max) # [Bq,]

      m_exp = tl.exp(m_prev - m_curr) # [Bq,]
      x_exp = tl.exp(x - m_curr[:, None]) # [Bq, Bk]

      l_curr = l_prev * m_exp #[Bq,]
      l_curr += tl.sum(x_exp, axis=-1) #[Bq,]

      o_right = m_exp * l_prev / l_curr  #[Bq,] - adding * l_prev / l_curr is wrong?
      o_right = o_right[:, None] * O # [Bq, d]

      o_left = tl.dot(x_exp / l_curr[:, None], v)
      O = (o_right + o_left) #[Bq, d]

      m_prev = m_curr
      l_prev = l_curr



      K_block = K_block.advance((BLOCK_K, 0))
      V_block = V_block.advance((BLOCK_K, 0))

    tl.store(output_block, O, boundary_check=(0,1))
    tl.store(L_block, m_prev + tl.log(l_prev + 1e-20), boundary_check=(0,))


@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    is_causal: tl.constexpr,
):
    """Flash Attention backward kernel.

    One program handles one query block and one batch element.
    dK/dV are accumulated atomically because many query blocks contribute
    to the same key/value rows.
    """
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Batch offsets
    Q_ptr = Q_ptr + pid_b * stride_qb
    K_ptr = K_ptr + pid_b * stride_kb
    V_ptr = V_ptr + pid_b * stride_vb
    O_ptr = O_ptr + pid_b * stride_ob
    L_ptr = L_ptr + pid_b * stride_lb
    dO_ptr = dO_ptr + pid_b * stride_dqb
    dQ_ptr = dQ_ptr + pid_b * stride_dqb
    dK_ptr = dK_ptr + pid_b * stride_dkb
    dV_ptr = dV_ptr + pid_b * stride_dvb

    Q_block = tl.make_block_ptr(
        Q_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    O_block = tl.make_block_ptr(
        O_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    dO_block = tl.make_block_ptr(
        dO_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    dQ_block = tl.make_block_ptr(
        dQ_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    L_block = tl.make_block_ptr(
        L_ptr,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(pid_q * BLOCK_Q,),
        block_shape=(BLOCK_Q,),
        order=(0,),
    )

    Q = tl.load(Q_block, boundary_check=(0, 1), padding_option="zero")
    O = tl.load(O_block, boundary_check=(0, 1), padding_option="zero")
    dO = tl.load(dO_block, boundary_check=(0, 1), padding_option="zero")
    L = tl.load(L_block, boundary_check=(0,), padding_option="zero")

    q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_idx < N_QUERIES
    d_offsets = tl.arange(0, D)

    # D = rowsum(O * dO)
    D_local = tl.sum(O * dO, axis=1)

    # Accumulator for dQ
    dQ_local = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

    num_key_blocks = tl.cdiv(N_KEYS, BLOCK_K)
    K_block = tl.make_block_ptr(
        K_ptr,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D),
        order=(1, 0),
    )
    V_block = tl.make_block_ptr(
        V_ptr,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D),
        order=(1, 0),
    )

    for block_idx in range(num_key_blocks):
        K = tl.load(K_block, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block, boundary_check=(0, 1), padding_option="zero")

        # Recompute scores and probabilities
        S = tl.dot(Q, tl.trans(K)) / scale
        k_idx = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_idx < N_KEYS
        S = tl.where(k_mask[None, :], S, float("-inf"))
        S = tl.where(q_mask[:, None], S, float("-inf"))

        if is_causal:
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            S = tl.where(causal_mask, S, float("-inf"))

        P = tl.exp(S - L[:, None])
        P = tl.where(q_mask[:, None] & k_mask[None, :], P, 0.0)

        # dV contribution: P^T @ dO
        dV_block_local = tl.dot(tl.trans(P).to(dO.dtype), dO)

        # dP = dO @ V^T
        dP = tl.dot(dO, tl.trans(V))

        # dS = P * (dP - D)
        dS = P * (dP - D_local[:, None])

        # dQ += dS @ K / scale
        dQ_local = dQ_local + tl.dot(dS.to(K.dtype), K) / scale

        # dK contribution: dS^T @ Q / scale
        dK_block_local = tl.dot(tl.trans(dS).to(Q.dtype), Q) / scale

        # Atomic add dK / dV into global gradients
        ptr_dK = dK_ptr + k_idx[:, None] * stride_dkk + d_offsets[None, :] * stride_dkd
        ptr_dV = dV_ptr + k_idx[:, None] * stride_dvk + d_offsets[None, :] * stride_dvd
        kv_mask = k_mask[:, None] & (d_offsets[None, :] < D)
        tl.atomic_add(ptr_dK, dK_block_local, mask=kv_mask)
        tl.atomic_add(ptr_dV, dV_block_local, mask=kv_mask)

        K_block = K_block.advance((BLOCK_K, 0))
        V_block = V_block.advance((BLOCK_K, 0))

    tl.store(dQ_block, dQ_local, boundary_check=(0, 1))


class FlashAttentionTriton(torch.autograd.Function):
    """Flash Attention using Triton kernel for forward pass."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass using Triton kernel.

        Args:
            Q: Query tensor of shape (batch, seq_q, d)
            K: Key tensor of shape (batch, seq_k, d)
            V: Value tensor of shape (batch, seq_k, d)
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape (batch, seq_q, d)
        """
        # Allocate output tensors O and L
        # Choose block sizes (e.g., BLOCK_Q = BLOCK_K = 64)
        # Configure grid: (num_query_blocks, batch_size)
        # Launch flash_fwd_kernel
        # Save tensors for backward
        # Return O
        device = Q.device
        dtype = Q.dtype
        batch_size, seq_q, d = Q.shape
        seq_k = K.shape[1]

        O = torch.empty((batch_size, seq_q, d), device=device, dtype=dtype)
        L = torch.empty((batch_size, seq_q), device=device, dtype=dtype)

        # Calculate grid dimensions
        BLOCK_Q = 16
        BLOCK_K = 16
        grid = (triton.cdiv(seq_q, BLOCK_Q), batch_size)

        # Launch Triton kernel for both causal and non-causal cases. Masking is
        # handled inside the kernel via the `is_causal` constexpr parameter.
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_q, seq_k, math.sqrt(d),
            D=d, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, is_causal=is_causal
        )

        # Save for backward
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        ctx.sqrt_d = math.sqrt(d)
        return O

    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass — reuse your PyTorch implementation from Part 2.
        """
        Q, K, V, L, O = ctx.saved_tensors
        # Try Triton backward kernel first; fall back to PyTorch formula if
        # Triton is unavailable or fails on the current GPU/runtime.
        if Q.is_cuda and K.is_cuda and V.is_cuda and dO.is_cuda:
            batch_size, seq_q, d = Q.shape
            seq_k = K.shape[1]

            dQ = torch.empty_like(Q)
            dK = torch.zeros_like(K)
            dV = torch.zeros_like(V)

            BLOCK_Q = 16
            BLOCK_K = 16
            grid = (triton.cdiv(seq_q, BLOCK_Q), batch_size)

            try:
                flash_bwd_kernel[grid](
                    Q, K, V, O, L, dO, dQ, dK, dV,
                    Q.stride(0), Q.stride(1), Q.stride(2),
                    K.stride(0), K.stride(1), K.stride(2),
                    V.stride(0), V.stride(1), V.stride(2),
                    O.stride(0), O.stride(1), O.stride(2),
                    L.stride(0), L.stride(1),
                    dQ.stride(0), dQ.stride(1), dQ.stride(2),
                    dK.stride(0), dK.stride(1), dK.stride(2),
                    dV.stride(0), dV.stride(1), dV.stride(2),
                    seq_q, seq_k, ctx.sqrt_d,
                    D=d, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, is_causal=ctx.is_causal,
                )
                return dQ, dK, dV, None
            except Exception:
                pass

        dQ, dK, dV, _ = attention_backward_impl(Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal)
        return dQ, dK, dV, None

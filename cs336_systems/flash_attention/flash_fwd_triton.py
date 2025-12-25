
import torch
import triton
import triton.language as tl
from jaxtyping import Float

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # very important, as we relay on tl.advance to move K_block_ptr
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Q_i = tl.load(Q_block_ptr)
    O_ij_prev = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_ij_prev = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_ij_prev = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    T_kv = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(T_kv):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale

        # if IS_CAUSAL:
        #     tl.fill(S_ij, float("-inf"), tl.arange(0, Q_TILE_SIZE) >= tl.arange(0, K_TILE_SIZE))
            
        m_ij = tl.maximum(m_ij_prev, tl.max(S_ij, axis=1))

        P_ij = tl.exp(S_ij - m_ij[:, None]) # expand dim of m_ij to the same shape as S_ij
        alpha = tl.exp(m_ij_prev - m_ij)
        l_ij = alpha * l_ij_prev + tl.sum(P_ij, axis=1)
        O_ij = alpha[:, None] * O_ij_prev + tl.dot(P_ij.to(V_j.dtype), V_j)

        O_ij_prev = O_ij
        l_ij_prev = l_ij
        m_ij_prev = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i = O_ij_prev / l_ij_prev[:, None] # divide each row by l_ij_prev
    L_i = m_ij_prev + tl.log(l_ij_prev)
    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
         Q: Float[torch.Tensor, '... seq d'],
         K: Float[torch.Tensor, '... seq d'],
         V: Float[torch.Tensor, '... seq d'],
         is_causal: bool = False):
        
        batch_size = Q.shape[0]
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        T_q = (Q.shape[-2] + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        T_kv = (K.shape[-2] + K_TILE_SIZE - 1) // K_TILE_SIZE
        O = torch.empty_like(Q)
        L = torch.empty(Q.shape[:-1], dtype=torch.float32, device=Q.device)
        flash_fwd_kernel[(T_q, batch_size)](
            Q_ptr=Q,
            K_ptr=K,
            V_ptr=V,
            O_ptr=O,
            L_ptr=L,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=O.stride(0),
            stride_oq=O.stride(1),
            stride_od=O.stride(2),
            stride_lb=L.stride(0),
            stride_lq=L.stride(1),
            N_QUERIES=Q.shape[-2],
            N_KEYS=K.shape[-2],
            scale=1.0 / Q.shape[-1]**0.5,
            D=Q.shape[-1],
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=is_causal,
        )
        ctx.save_for_backward(O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

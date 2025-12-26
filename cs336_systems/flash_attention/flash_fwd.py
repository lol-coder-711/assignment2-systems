
import torch
from jaxtyping import Float, Int
from einops import einsum

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
         Q: Float[torch.Tensor, '... seq d'],
         K: Float[torch.Tensor, '... seq d'],
         V: Float[torch.Tensor, '... seq d'],
         is_causal: bool = False):
        """
        In this function, we implement the forward pass of FlashAttention-2.
        """
        tile_size = (16, 16)
        B_q, B_k = tile_size
        T_q = (Q.shape[-2] + B_q - 1) // B_q
        T_kv = (K.shape[-2] + B_k - 1) // B_k
        d = Q.shape[-1]
        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1], device=Q.device, dtype=torch.float32)
        # print("\n!!!!!!!!!!!!!!!!!!!!! shape of Q, K, V", Q.shape, K.shape, V.shape)
        for i in range(T_q):
            Q_i = Q[:, i * B_q:(i + 1) * B_q]
            O_ij_prev = torch.zeros_like(Q_i)
            l_ij_prev = torch.zeros(Q_i.shape[:-1], dtype=torch.int32)
            m_ij_prev = torch.full(Q_i.shape[:-1], -torch.inf, device=Q.device, dtype=torch.float32)
            for j in range(T_kv):
                K_j = K[:, j * B_k:(j + 1) * B_k]
                V_j = V[:, j * B_k:(j + 1) * B_k]
                S_ij = einsum(Q_i, K_j, '... q d, ... k d -> ... q k') / d**0.5
                m_ij = torch.max(m_ij_prev, torch.max(S_ij, dim=-1).values)
                P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))
                l_ij = einsum(torch.exp(m_ij_prev - m_ij), l_ij_prev, '... q, ... q -> ... q') + torch.sum(P_ij, dim=-1)
                O_ij = torch.exp(m_ij_prev - m_ij).unsqueeze(-1) * O_ij_prev + einsum(P_ij, V_j, '... Bq Bk, ... Bk d -> ... Bq d')

                O_ij_prev = O_ij
                l_ij_prev = l_ij
                m_ij_prev = m_ij
            O_i = O_ij_prev / l_ij_prev.unsqueeze(-1) # use unsqueeze to broadcast
            L_i = m_ij_prev + torch.log(l_ij_prev)
            O[:, i * B_q:(i + 1) * B_q] = O_i
            L[:, i * B_q:(i + 1) * B_q] = L_i
        ctx.save_for_backward(O, L, Q, K, V)
        ctx.is_causal = is_causal
        return O
                


    @staticmethod
    def backward(ctx, dO):
        """
        In this function, we implement the backward pass of FlashAttention-2.
        """
        O, L, Q, K, V = ctx.saved_tensors
        D = torch.sum(O * dO, dim=-1)
        d = Q.shape[-1]
        S = einsum(Q, K, '... q d, ... k d -> ... q k') / d**0.5
        # Need to also apply mask to S during backward
        if ctx.is_causal:
                mask = torch.arange(Q.shape[-2], device=Q.device)[None, :, None] >= torch.arange(K.shape[-2], device=K.device)[None, None, :]
                S = torch.where(mask, S, float("-inf"))

        P = torch.exp(S - L.unsqueeze(-1))
        dV = einsum(P, dO, '... q k, ... q d -> ... k d')
        dP = einsum(dO, V, '... q d, ... k d -> ... q k')
        dS = P * (dP - D.unsqueeze(-1))
        dQ = einsum(dS, K, '... q k, ... k d -> ... q d') / d**0.5
        dK = einsum(dS, Q, '... q k, ... q d -> ... k d') / d**0.5
        return dQ, dK, dV, None



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .basic_layers import PreNorm, PreNorm_qkv, Attention, FeedForward


class VariationalCodebook(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        codebook_size: int = 512,
        num_codebooks: int = 3,
        decay: float = 0.99,
        eps: float = 1e-5,
        temperature: float = 0.07,
        commitment_weight: float = 0.25,
        threshold_ema_dead_code: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.decay = decay
        self.eps = eps
        self.temperature = temperature
        self.commitment_weight = commitment_weight
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        self.register_buffer('codebook', torch.randn(num_codebooks, codebook_size, dim))
        self.register_buffer('codebook_usage', torch.zeros(num_codebooks, codebook_size))
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('ema_embed_sum', torch.zeros(num_codebooks, codebook_size, dim))
        
        self.register_buffer('initialized', torch.zeros(num_codebooks, dtype=torch.bool))
        
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
    def _initialize_codebook(self, embeddings: torch.Tensor, modality_idx: int):
        B, T, D = embeddings.shape
        flat = embeddings.reshape(-1, D)

        n_samples = min(flat.size(0), self.codebook_size)
        indices = torch.randperm(flat.size(0), device=flat.device)[:n_samples]

        sampled = flat[indices].detach()
        init_cluster_size = 1.0

        self.codebook[modality_idx, :n_samples] = sampled
        self.ema_embed_sum[modality_idx, :n_samples] = sampled * init_cluster_size
        self.ema_cluster_size[modality_idx, :n_samples] = init_cluster_size
        
    @torch.no_grad()
    def update_codebook_ema(self, embeddings: torch.Tensor, modality_idx: int):
        if not self.training:
            return
            
        B, T, D = embeddings.shape
        flat = embeddings.reshape(-1, D)
        
        if not self.initialized[modality_idx]:
            self._initialize_codebook(embeddings, modality_idx)
            self.initialized[modality_idx] = True
        
        codebook_m = self.codebook[modality_idx]
        
        flat_norm = (flat ** 2).sum(dim=-1, keepdim=True)
        codebook_norm = (codebook_m ** 2).sum(dim=-1)
        dist = flat_norm + codebook_norm - 2 * flat @ codebook_m.t()
        
        min_indices = dist.argmin(dim=-1)
        
        one_hot = F.one_hot(min_indices, self.codebook_size).float()
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ flat
        
        self.ema_cluster_size[modality_idx] = (
            self.decay * self.ema_cluster_size[modality_idx] + 
            (1 - self.decay) * cluster_size
        )
        self.ema_embed_sum[modality_idx] = (
            self.decay * self.ema_embed_sum[modality_idx] + 
            (1 - self.decay) * embed_sum
        )
        
        n = self.ema_cluster_size[modality_idx].sum()
        cluster_size_normalized = (
            (self.ema_cluster_size[modality_idx] + self.eps) / 
            (n + self.codebook_size * self.eps) * n
        )
        
        self.codebook[modality_idx] = (
            self.ema_embed_sum[modality_idx] /
            cluster_size_normalized.unsqueeze(-1).clamp(min=self.eps)
        )

        self.codebook_usage[modality_idx] = (
            0.99 * self.codebook_usage[modality_idx] + 0.01 * cluster_size
        )

        if self.threshold_ema_dead_code > 0:
            dead_codes = self.ema_cluster_size[modality_idx] < self.threshold_ema_dead_code
            num_dead = dead_codes.sum().item()

            if num_dead > 0:
                n_samples = flat.size(0)
                if n_samples >= num_dead:
                    replace_indices = torch.randperm(n_samples, device=flat.device)[:int(num_dead)]
                else:
                    replace_indices = torch.randint(0, n_samples, (int(num_dead),), device=flat.device)

                sampled = flat[replace_indices].detach()

                self.codebook[modality_idx, dead_codes] = sampled
                self.ema_cluster_size[modality_idx, dead_codes] = self.threshold_ema_dead_code
                self.ema_embed_sum[modality_idx, dead_codes] = sampled * self.threshold_ema_dead_code

    def probabilistic_retrieve(
        self,
        query: torch.Tensor,
        modality_idx: int,
        completeness: torch.Tensor = None,
        top_k: int = 8,
    ) -> tuple:
        B, T, D = query.shape
        codebook_m = self.codebook[modality_idx].detach()

        query_flat = query.reshape(B * T, D)

        query_sq = (query_flat ** 2).sum(dim=-1, keepdim=True)
        codebook_sq = (codebook_m ** 2).sum(dim=-1)
        dist_sq = query_sq + codebook_sq - 2 * query_flat @ codebook_m.t()

        scale = D ** 0.5

        sim = -dist_sq / scale

        temperature = self.log_temperature.exp()
        if completeness is not None:
            comp_expanded = completeness.unsqueeze(1).expand(B, T).reshape(B * T, 1)
            adaptive_temp = temperature * (2.0 - comp_expanded)
        else:
            adaptive_temp = temperature

        if top_k < self.codebook_size:
            topk_sim, topk_indices = sim.topk(top_k, dim=-1)

            if self.training:
                gumbel_noise = -torch.log(-torch.log(
                    torch.rand_like(topk_sim).clamp(min=1e-8, max=1-1e-8)
                ))
                topk_weights = F.softmax((topk_sim + gumbel_noise) / adaptive_temp, dim=-1)
            else:
                topk_weights = F.softmax(topk_sim / adaptive_temp, dim=-1)

            topk_codebook = codebook_m[topk_indices]
            retrieved = (topk_weights.unsqueeze(-1) * topk_codebook).sum(dim=-2)

            retrieval_weights = torch.zeros_like(sim).scatter_(-1, topk_indices, topk_weights)
        else:
            if self.training:
                gumbel_noise = -torch.log(-torch.log(
                    torch.rand_like(sim).clamp(min=1e-8, max=1-1e-8)
                ))
                retrieval_weights = F.softmax((sim + gumbel_noise) / adaptive_temp, dim=-1)
            else:
                retrieval_weights = F.softmax(sim / adaptive_temp, dim=-1)

            retrieved = retrieval_weights @ codebook_m

        retrieved = retrieved.reshape(B, T, D)
        retrieval_weights = retrieval_weights.reshape(B, T, -1)

        return retrieved, retrieval_weights

    def compute_commitment_loss(self, embeddings: torch.Tensor, modality_idx: int) -> torch.Tensor:
        B, T, D = embeddings.shape
        flat = embeddings.reshape(-1, D)
        codebook_m = self.codebook[modality_idx].detach()

        dist = torch.cdist(flat, codebook_m)
        min_indices = dist.argmin(dim=-1)

        quantized = codebook_m[min_indices]

        commitment_loss = F.mse_loss(flat, quantized)

        return commitment_loss * self.commitment_weight

    def get_codebook_utilization(self) -> dict:
        usage = self.codebook_usage.sum(dim=-1)
        active_codes = (self.codebook_usage > 0.01).sum(dim=-1).float()
        utilization = active_codes / self.codebook_size

        return {
            'total_usage': usage,
            'active_codes': active_codes,
            'utilization_ratio': utilization,
        }


class MemoryAugmentedPerceiverLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.self_attn = PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        completeness: torch.Tensor = None
    ) -> torch.Tensor:
        B, T, D = inputs.shape

        if completeness is not None:
            comp = completeness.view(B, 1, 1)
            gate_weight = comp * 0.5 + 0.25
        else:
            gate_weight = 0.5

        concat_feat = torch.cat([inputs, memory], dim=-1)
        gate = self.gate(concat_feat)

        fused_kv = gate * inputs + (1 - gate) * memory

        latents = self.cross_attn(latents, fused_kv, fused_kv) + latents

        latents = self.self_attn(latents, latents, latents) + latents

        latents = self.ff(latents) + latents

        return latents


class PRM(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        codebook_size: int = 512,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 32,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        num_latents: int = 8,
        decay: float = 0.99,
        temperature: float = 0.07,
        commitment_weight: float = 0.25,
        retrieval_top_k: int = 32,
        share_perceiver: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.retrieval_top_k = retrieval_top_k
        self.share_perceiver = share_perceiver

        self.codebook = VariationalCodebook(
            dim=dim,
            codebook_size=codebook_size,
            num_codebooks=3,
            decay=decay,
            temperature=temperature,
            commitment_weight=commitment_weight,
        )

        if share_perceiver:
            self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
            self.layers = nn.ModuleList([
                MemoryAugmentedPerceiverLayer(
                    dim=dim, heads=heads, dim_head=dim_head,
                    mlp_dim=mlp_dim, dropout=dropout
                ) for _ in range(depth)
            ])
            self.output_proj = nn.Linear(dim, dim)
        else:
            self.latents_per_modality = nn.ParameterList([
                nn.Parameter(torch.randn(1, num_latents, dim))
                for _ in range(3)
            ])
            self.layers_per_modality = nn.ModuleList([
                nn.ModuleList([
                    MemoryAugmentedPerceiverLayer(
                        dim=dim, heads=heads, dim_head=dim_head,
                        mlp_dim=mlp_dim, dropout=dropout
                    ) for _ in range(depth)
                ]) for _ in range(3)
            ])
            self.output_proj_per_modality = nn.ModuleList([
                nn.Linear(dim, dim) for _ in range(3)
            ])

    def _enhance_single_modality(
        self,
        h: torch.Tensor,
        modality_idx: int,
        completeness: torch.Tensor = None,
    ) -> tuple:
        B, T, D = h.shape

        memory, retrieval_weights = self.codebook.probabilistic_retrieve(
            query=h,
            modality_idx=modality_idx,
            completeness=completeness,
            top_k=self.retrieval_top_k,
        )

        if self.share_perceiver:
            latents = self.latents.expand(B, -1, -1)
            layers = self.layers
            output_proj = self.output_proj
        else:
            latents = self.latents_per_modality[modality_idx].expand(B, -1, -1)
            layers = self.layers_per_modality[modality_idx]
            output_proj = self.output_proj_per_modality[modality_idx]

        for layer in layers:
            latents = layer(latents, h, memory, completeness)

        h_enhanced = output_proj(latents)

        if h_enhanced.size(1) >= T:
            h_enhanced = h_enhanced[:, :T, :]
        else:
            repeat_times = (T + h_enhanced.size(1) - 1) // h_enhanced.size(1)
            h_enhanced = h_enhanced.repeat(1, repeat_times, 1)
            h_enhanced = h_enhanced[:, :T, :]

        return h_enhanced, retrieval_weights

    def forward(
        self,
        h_l: torch.Tensor,
        h_a: torch.Tensor,
        h_v: torch.Tensor,
        w_l: torch.Tensor = None,
        w_a: torch.Tensor = None,
        w_v: torch.Tensor = None,
    ) -> tuple:
        h_l_enhanced, weights_l = self._enhance_single_modality(h_l, modality_idx=0, completeness=w_l)
        h_a_enhanced, weights_a = self._enhance_single_modality(h_a, modality_idx=1, completeness=w_a)
        h_v_enhanced, weights_v = self._enhance_single_modality(h_v, modality_idx=2, completeness=w_v)

        self._last_retrieval_weights = {
            'text': weights_l,
            'audio': weights_a,
            'video': weights_v,
        }

        return h_l_enhanced, h_a_enhanced, h_v_enhanced

    @torch.no_grad()
    def update_codebook(
        self,
        h_l_complete: torch.Tensor,
        h_a_complete: torch.Tensor,
        h_v_complete: torch.Tensor,
    ):
        if not self.training:
            return

        self.codebook.update_codebook_ema(h_l_complete, modality_idx=0)
        self.codebook.update_codebook_ema(h_a_complete, modality_idx=1)
        self.codebook.update_codebook_ema(h_v_complete, modality_idx=2)

    def compute_rectification_loss(
        self,
        h_l: torch.Tensor,
        h_a: torch.Tensor,
        h_v: torch.Tensor,
    ) -> torch.Tensor:
        commitment_l = self.codebook.compute_commitment_loss(h_l, modality_idx=0)
        commitment_a = self.codebook.compute_commitment_loss(h_a, modality_idx=1)
        commitment_v = self.codebook.compute_commitment_loss(h_v, modality_idx=2)

        commitment_loss = (commitment_l + commitment_a + commitment_v) / 3.0

        utilization = self.codebook.get_codebook_utilization()
        diversity_loss = (1.0 - utilization['utilization_ratio'].mean()) * 0.1

        return commitment_loss + diversity_loss

    def get_codebook_stats(self) -> dict:
        stats = self.codebook.get_codebook_utilization()
        stats['temperature'] = self.codebook.log_temperature.exp().item()
        return stats

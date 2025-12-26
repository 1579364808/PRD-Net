import torch
from torch import nn
import torch.nn.functional as F


class UnimodalVIB(nn.Module):
    def __init__(self, dim=128, name='modality', out_dim=1):
        super().__init__()
        self.name = name

        self.encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(dim, dim)
        self.fc_logvar = nn.Linear(dim, dim)

        self.predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, out_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, h):
        h_pooled = h.mean(dim=1)

        h_encoded = self.encoder(h_pooled)
        mu = self.fc_mu(h_encoded)
        logvar = self.fc_logvar(h_encoded)

        z = self.reparameterize(mu, logvar)

        pred = self.predictor(z)

        return z, mu, logvar, pred


class HID(nn.Module):
    def __init__(self, dim=128, use_poe_fusion=False, out_dim=1):
        super().__init__()

        self.use_poe_fusion = use_poe_fusion

        self.unimodal_vib_l = UnimodalVIB(dim, name='language', out_dim=out_dim)
        self.unimodal_vib_a = UnimodalVIB(dim, name='audio', out_dim=out_dim)
        self.unimodal_vib_v = UnimodalVIB(dim, name='video', out_dim=out_dim)

        if use_poe_fusion:
            self.multimodal_encoder = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim * 2)
            )
        else:
            self.multimodal_encoder = nn.Sequential(
                nn.Linear(dim * 3, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim * 2)
            )

        self.fc_mu = nn.Linear(dim * 2, dim)
        self.fc_logvar = nn.Linear(dim * 2, dim)

        self.multimodal_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, out_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def product_of_experts_fusion(self, mu_l, logvar_l, mu_a, logvar_a, mu_v, logvar_v):
        var_l = torch.exp(logvar_l)
        var_a = torch.exp(logvar_a)
        var_v = torch.exp(logvar_v)

        precision_l = 1.0 / (var_l + 1e-8)
        precision_a = 1.0 / (var_a + 1e-8)
        precision_v = 1.0 / (var_v + 1e-8)

        precision_fused = precision_l + precision_a + precision_v

        var_fused = 1.0 / (precision_fused + 1e-8)
        logvar_fused = torch.log(var_fused + 1e-8)

        mu_fused = var_fused * (precision_l * mu_l + precision_a * mu_a + precision_v * mu_v)

        return mu_fused, logvar_fused

    def forward(self, h_l, h_a, h_v):
        b = h_l.size(0)

        z_l, mu_l, logvar_l, pred_l = self.unimodal_vib_l(h_l)
        z_a, mu_a, logvar_a, pred_a = self.unimodal_vib_a(h_a)
        z_v, mu_v, logvar_v, pred_v = self.unimodal_vib_v(h_v)

        if self.use_poe_fusion:
            mu_fused, logvar_fused = self.product_of_experts_fusion(
                mu_l, logvar_l, mu_a, logvar_a, mu_v, logvar_v
            )

            var_l = torch.exp(logvar_l)
            var_a = torch.exp(logvar_a)
            var_v = torch.exp(logvar_v)
            precision_l = 1.0 / (var_l + 1e-8)
            precision_a = 1.0 / (var_a + 1e-8)
            precision_v = 1.0 / (var_v + 1e-8)
            precision_sum = precision_l + precision_a + precision_v
            w_l = precision_l / (precision_sum + 1e-8)
            w_a = precision_a / (precision_sum + 1e-8)
            w_v = precision_v / (precision_sum + 1e-8)

            z_fused = self.reparameterize(mu_fused, logvar_fused)
            fusion_input = z_fused
        else:
            z_concat = torch.cat([z_l, z_a, z_v], dim=-1)
            fusion_input = z_concat
            w_l = w_a = w_v = None

        h_encoded = self.multimodal_encoder(fusion_input)
        mu_multi = self.fc_mu(h_encoded)
        logvar_multi = self.fc_logvar(h_encoded)

        z_multi = self.reparameterize(mu_multi, logvar_multi)

        h_proxy = z_multi.unsqueeze(1).expand(-1, 8, -1)

        pred_multi = self.multimodal_predictor(z_multi)

        outputs = {
            'mu_l': mu_l, 'logvar_l': logvar_l, 'pred_l': pred_l,
            'mu_a': mu_a, 'logvar_a': logvar_a, 'pred_a': pred_a,
            'mu_v': mu_v, 'logvar_v': logvar_v, 'pred_v': pred_v,
            'mu_multi': mu_multi, 'logvar_multi': logvar_multi,
            'pred_multi': pred_multi
        }

        if self.use_poe_fusion:
            outputs.update({
                'w_l': w_l, 'w_a': w_a, 'w_v': w_v,
                'mu_fused': mu_fused, 'logvar_fused': logvar_fused
            })

        return h_proxy, outputs

import torch
from torch import nn
from .basic_layers import Transformer, UGLR
from .bert import BertTextEncoder
from .hid import HID
from .prm import PRM
from einops import repeat


class PRD_Net(nn.Module):
    def __init__(self, args):
        super(PRD_Net, self).__init__()

        self.use_unimodal_vib = args['base'].get('use_unimodal_vib', True)
        self.use_multimodal_vib = args['base'].get('use_multimodal_vib', True)
        self.use_poe_fusion = args['base'].get('use_poe_fusion', True)
        self.use_reconstructor = args['base'].get('use_reconstructor', True)
        self.use_uglr = args['base'].get('use_uglr', True)
        self.use_prm = args['base'].get('use_prm', True)

        if not self.use_multimodal_vib:
            self.h_latent_param = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0],
                                                   args['model']['feature_extractor']['hidden_dims'][0]))

        self.bertmodel = BertTextEncoder(
            use_finetune=True,
            transformers='bert',
            pretrained=args['model']['feature_extractor']['bert_pretrained']
        )

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0],
                     args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0],
                       save_hidden=False,
                       token_len=args['model']['feature_extractor']['token_length'][0],
                       dim=args['model']['feature_extractor']['hidden_dims'][0],
                       depth=args['model']['feature_extractor']['depth'],
                       heads=args['model']['feature_extractor']['heads'],
                       mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2],
                     args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2],
                       save_hidden=False,
                       token_len=args['model']['feature_extractor']['token_length'][2],
                       dim=args['model']['feature_extractor']['hidden_dims'][2],
                       depth=args['model']['feature_extractor']['depth'],
                       heads=args['model']['feature_extractor']['heads'],
                       mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1],
                     args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1],
                       save_hidden=False,
                       token_len=args['model']['feature_extractor']['token_length'][1],
                       dim=args['model']['feature_extractor']['hidden_dims'][1],
                       depth=args['model']['feature_extractor']['depth'],
                       heads=args['model']['feature_extractor']['heads'],
                       mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )

        if self.use_prm:
            prm_config = args['model'].get('prm', {})
            hidden_dim = prm_config.get('dim', 128)

            self.prm = PRM(
                dim=prm_config.get('dim', 128),
                codebook_size=prm_config.get('codebook_size', 512),
                depth=prm_config.get('depth', 2),
                heads=prm_config.get('heads', 4),
                dim_head=prm_config.get('dim_head', 32),
                mlp_dim=prm_config.get('mlp_dim', 256),
                dropout=prm_config.get('dropout', 0.1),
                num_latents=prm_config.get('num_latents', 8),
                decay=prm_config.get('decay', 0.99),
                temperature=prm_config.get('temperature', 0.07),
                commitment_weight=prm_config.get('commitment_weight', 0.25),
                retrieval_top_k=prm_config.get('retrieval_top_k', 32),
                share_perceiver=prm_config.get('share_perceiver', True),
            )

            self.completeness_estimator_l = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            self.completeness_estimator_a = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            self.completeness_estimator_v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        if self.use_reconstructor:
            self.reconstructor = nn.ModuleList([
                Transformer(num_frames=args['model']['reconstructor']['input_length'],
                           save_hidden=False,
                           token_len=None,
                           dim=args['model']['reconstructor']['input_dim'],
                           depth=args['model']['reconstructor']['depth'],
                           heads=args['model']['reconstructor']['heads'],
                           mlp_dim=args['model']['reconstructor']['hidden_dim']) for _ in range(3)
            ])

        if self.use_multimodal_vib:
            self.hid = HID(dim=128, use_poe_fusion=self.use_poe_fusion)

        regression_config = args['model'].get('regression', {})
        self.regression = nn.Linear(
            regression_config.get('input_dim', 128),
            regression_config.get('out_dim', 1)
        )

        if self.use_uglr:
            uglr_config = args['model'].get('uglr', {})
            self.uglr = UGLR(
                latent_dim=uglr_config.get('latent_dim', 128),
                text_dim=uglr_config.get('text_dim', args['model']['feature_extractor']['hidden_dims'][0]),
                audio_dim=uglr_config.get('audio_dim', args['model']['feature_extractor']['hidden_dims'][2]),
                video_dim=uglr_config.get('video_dim', args['model']['feature_extractor']['hidden_dims'][1]),
                embed_dim=uglr_config.get('embed_dim', 128),
                num_layers=uglr_config.get('num_layers', 3),
                attn_dropout=uglr_config.get('attn_dropout', 0.1)
            )
        else:
            concat_dim = args['model']['feature_extractor']['hidden_dims'][0] * 3
            self.concat_proj = nn.Linear(concat_dim, args['model']['feature_extractor']['hidden_dims'][0])


    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        b = vision_m.size(0)

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        h_1_l_enhanced, h_1_a_enhanced, h_1_v_enhanced = h_1_l, h_1_a, h_1_v
        w_l, w_a, w_v = None, None, None

        l_rect = None
        if self.use_prm:
            w_l = self.completeness_estimator_l(h_1_l.mean(dim=1))
            w_a = self.completeness_estimator_a(h_1_a.mean(dim=1))
            w_v = self.completeness_estimator_v(h_1_v.mean(dim=1))

            h_1_l_enhanced, h_1_a_enhanced, h_1_v_enhanced = self.prm(
                h_1_l, h_1_a, h_1_v,
                w_l=w_l.squeeze(-1),
                w_a=w_a.squeeze(-1),
                w_v=w_v.squeeze(-1)
            )

            l_rect = self.prm.compute_rectification_loss(h_1_l, h_1_a, h_1_v)

        rec_feats, complete_feats = None, None
        complete_language_feat, complete_vision_feat, complete_audio_feat = None, None, None

        if self.use_reconstructor and (vision is not None) and (audio is not None) and (language is not None):
            rec_feat_a = self.reconstructor[0](h_1_a)
            rec_feat_v = self.reconstructor[1](h_1_v)
            rec_feat_l = self.reconstructor[2](h_1_l)
            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1)

            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]
            complete_feats = torch.cat([complete_audio_feat, complete_vision_feat, complete_language_feat], dim=1)

        vib_outputs = None
        if self.use_multimodal_vib:
            h_latent, vib_outputs = self.hid(h_1_l_enhanced, h_1_a_enhanced, h_1_v_enhanced)
        else:
            h_latent = repeat(self.h_latent_param, '1 n d -> b n d', b=b)

        if self.use_uglr:
            weight_t_v_a = None
            if vib_outputs is not None and 'weights' in vib_outputs:
                weights = vib_outputs['weights']
                weight_t_v_a = torch.stack([
                    weights[:, 0:1].unsqueeze(-1).unsqueeze(-1).expand_as(h_1_l_enhanced),
                    weights[:, 2:3].unsqueeze(-1).unsqueeze(-1).expand_as(h_1_v_enhanced),
                    weights[:, 1:2].unsqueeze(-1).unsqueeze(-1).expand_as(h_1_a_enhanced)
                ], dim=0)

            h_refined = self.uglr(h_latent, h_1_l_enhanced, h_1_a_enhanced, h_1_v_enhanced, weight_t_v_a)
        else:
            h_refined = torch.cat([h_1_l_enhanced, h_1_a_enhanced, h_1_v_enhanced], dim=-1)
            h_refined = self.concat_proj(h_refined)

        h_refined_pooled = torch.mean(h_refined[:, :], dim=1)
        output = self.regression(h_refined_pooled)

        return {
            'sentiment_preds': output,
            'vib_outputs': vib_outputs,
            'rec_feats': rec_feats,
            'complete_feats': complete_feats,
            'h_1_l': h_1_l_enhanced,
            'h_1_a': h_1_a_enhanced,
            'h_1_v': h_1_v_enhanced,
            'w_l': w_l,
            'w_a': w_a,
            'w_v': w_v,
            'l_rect': l_rect,
            'complete_l': complete_language_feat,
            'complete_a': complete_audio_feat,
            'complete_v': complete_vision_feat,
            'h_refined': h_refined,
            'h_refined_pooled': h_refined_pooled,
            'tsne_viz': {
                'Raw_L': torch.mean(h_1_l, dim=1),
                'Raw_A': torch.mean(h_1_a, dim=1),
                'Raw_V': torch.mean(h_1_v, dim=1),
                'Rec_L': torch.mean(h_1_l_enhanced, dim=1),
                'Rec_A': torch.mean(h_1_a_enhanced, dim=1),
                'Rec_V': torch.mean(h_1_v_enhanced, dim=1),
            },
        }


def build_model(args):
    return PRD_Net(args)

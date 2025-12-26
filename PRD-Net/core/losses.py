import torch
from torch import nn
from torch.nn import functional as F


class PRD_Loss(nn.Module):
    """
    PRD-Net Loss Function
    包含：主任务损失 + 单模态VIB损失 + 多模态VIB损失 + PRM修正损失
    """
    def __init__(self, args):
        super().__init__()
        self.sigma = args['base']['sigma']  # 主任务权重

        # 单模态VIB参数
        self.beta_uni = args['base'].get('beta_uni', 0.0001)  # 单模态KL权重
        self.alpha_uni = args['base'].get('alpha_uni', 0.1)  # 单模态VIB总权重

        # 多模态VIB参数
        self.beta_multi = args['base'].get('beta_multi', 0.001)  # 多模态KL权重
        self.alpha_multi = args['base'].get('alpha_multi', 0.1)  # 多模态VIB总权重

        # β warmup 参数（MIB论文建议）
        self.beta_warmup_epochs = args['base'].get('beta_warmup_epochs', 0)  # warmup轮数，0表示不使用
        self.current_epoch = 0  # 当前epoch，需要外部更新

        # 重构器参数
        self.gamma = args['base'].get('gamma', 0.1)  # 重构损失权重

        # 完整性监督参数
        self.eta = args['base'].get('eta', 0.1)  # 完整性监督损失权重

        # PRM参数
        self.lambda_rect = args['base'].get('lambda_rect', 0.1)  # 原型校正损失权重

        # 配置开关
        self.use_unimodal_vib = args['base'].get('use_unimodal_vib', True)
        self.use_multimodal_vib = args['base'].get('use_multimodal_vib', True)
        self.use_reconstructor = args['base'].get('use_reconstructor', True)
        self.use_prm = args['base'].get('use_prm', True)

        self.MSE_Fn = nn.MSELoss()

    def set_epoch(self, epoch):
        """更新当前epoch用于β warmup"""
        self.current_epoch = epoch

    def get_beta_schedule(self, beta_target):
        """
        计算当前epoch的β值（带warmup）

        Args:
            beta_target: 目标β值
        Returns:
            当前β值
        """
        if self.beta_warmup_epochs <= 0:
            return beta_target

        # 线性warmup: β从0逐渐增加到目标值
        warmup_factor = min(1.0, self.current_epoch / self.beta_warmup_epochs)
        return warmup_factor * beta_target

    def kl_divergence(self, mu, logvar):
        """
        Args:
            mu: [B, D]
            logvar: [B, D]
        """
        logvar = torch.clamp(logvar, min=-10, max=10) 
        
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(kl)

    def forward(self, out, label):
        """
        Args:
            out: 模型输出字典
            label: 标签字典
        Returns:
            损失字典
        """
        # 1. 主任务损失（情感预测）
        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels'])

        # 2. 重构器损失（在VIB之前）
        l_rec = 0
        if self.use_reconstructor and out['rec_feats'] is not None and out['complete_feats'] is not None:
            l_rec = self.MSE_Fn(out['rec_feats'], out['complete_feats'])

        # 如果没有使用VIB，只返回主任务损失+重构损失
        if out['vib_outputs'] is None or not self.use_multimodal_vib:
            loss = self.sigma * l_sp + self.gamma * l_rec
            return {
                'loss': loss,
                'l_sp': l_sp,
                'l_rec': l_rec,
                'l_kl_uni': 0,
                'l_kl_multi': 0,
                'l_task_uni': 0,
                'l_task_multi': 0
            }

        vib = out['vib_outputs']

        # 3. 单模态VIB损失（带β warmup）
        l_kl_uni = 0
        l_task_uni = 0

        if self.use_unimodal_vib:
            # 单模态KL散度
            l_kl_l = self.kl_divergence(vib['mu_l'], vib['logvar_l'])
            l_kl_a = self.kl_divergence(vib['mu_a'], vib['logvar_a'])
            l_kl_v = self.kl_divergence(vib['mu_v'], vib['logvar_v'])
            l_kl_uni = (l_kl_l + l_kl_a + l_kl_v) / 3

            # 单模态辅助任务损失
            l_task_l = self.MSE_Fn(vib['pred_l'], label['sentiment_labels'])
            l_task_a = self.MSE_Fn(vib['pred_a'], label['sentiment_labels'])
            l_task_v = self.MSE_Fn(vib['pred_v'], label['sentiment_labels'])
            l_task_uni = (l_task_l + l_task_a + l_task_v) / 3

        # 4. 多模态VIB损失（带β warmup）
        l_kl_multi = 0
        l_task_multi = 0

        if self.use_multimodal_vib:
            # 多模态KL散度
            l_kl_multi = self.kl_divergence(vib['mu_multi'], vib['logvar_multi'])

            # 多模态辅助任务损失
            l_task_multi = self.MSE_Fn(vib['pred_multi'], label['sentiment_labels'])

        # 5. 完整性监督损失
        l_cc = 0
        if self.use_prm and 'completeness_labels_l' in label:
            if out['w_l'] is not None:
                l_cc += self.MSE_Fn(out['w_l'].squeeze(-1), label['completeness_labels_l'].squeeze(-1))
            if out['w_a'] is not None:
                l_cc += self.MSE_Fn(out['w_a'].squeeze(-1), label['completeness_labels_a'].squeeze(-1))
            if out['w_v'] is not None:
                l_cc += self.MSE_Fn(out['w_v'].squeeze(-1), label['completeness_labels_v'].squeeze(-1))
            l_cc = l_cc / 3  # 三模态平均

        # 6. PRM损失（原型校正：commitment + diversity）
        l_rect = 0
        if self.use_prm and out.get('l_rect') is not None:
            l_rect = out['l_rect']

        # 7. 总损失（应用β warmup）
        # 计算当前epoch的β值
        beta_uni_current = self.get_beta_schedule(self.beta_uni)
        beta_multi_current = self.get_beta_schedule(self.beta_multi)

        # VIB损失 = β(带warmup) × KL散度 + 辅助任务损失
        l_vib_uni = beta_uni_current * l_kl_uni + l_task_uni
        l_vib_multi = beta_multi_current * l_kl_multi + l_task_multi

        loss = (self.sigma * l_sp +
                self.gamma * l_rec +
                self.alpha_uni * l_vib_uni +
                self.alpha_multi * l_vib_multi +
                self.eta * l_cc +
                self.lambda_rect * l_rect)

        return {
            'loss': loss,
            'l_sp': l_sp,
            'l_rec': l_rec,
            'l_kl_uni': l_kl_uni,
            'l_kl_multi': l_kl_multi,
            'l_task_uni': l_task_uni,
            'l_task_multi': l_task_multi,
            'l_cc': l_cc,
            'l_rect': l_rect,
            'beta_uni_current': beta_uni_current,
            'beta_multi_current': beta_multi_current,
        }





import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== Semantic Alignment Loss ======================
class SemanticAlignmentLoss(nn.Module):
    def __init__(self, num_samples, num_clusters, js_weight=0.1):
        super().__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.js_weight = js_weight

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(False)
        for i in range(N // 2):
            mask[i, N // 2 + i] = False
            mask[N // 2 + i, i] = False
        return mask

    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward_prob(self, q_i, q_j):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        p_i = q_i.sum(0).view(-1)
        p_i = p_i / (p_i.sum() + 1e-9)
        ne_i = (p_i * torch.log(p_i + 1e-9)).sum()

        p_j = q_j.sum(0).view(-1)
        p_j = p_j / (p_j.sum() + 1e-9)
        ne_j = (p_j * torch.log(p_j + 1e-9)).sum()

        kl_ij = torch.sum(p_i * torch.log((p_i + 1e-9) / (p_j + 1e-9)))
        kl_ji = torch.sum(p_j * torch.log((p_j + 1e-9) / (p_i + 1e-9)))
        js_divergence = 0.5 * (kl_ij + kl_ji)

        entropy = ne_i + ne_j + self.js_weight * js_divergence
        return entropy

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):
        """
        q_i, q_j: [B, K]
        先转 target distribution，再在 cluster dimension 上做对比。
        """
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        q_i = q_i.t()  # [K, B]
        q_j = q_j.t()  # [K, B]

        N = 2 * self.num_clusters
        q = torch.cat((q_i, q_j), dim=0)  # [2K, B]

        if normalized:
            sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l
        else:
            sim = torch.matmul(q, q.T) / temperature_l

        # 正样本：前 K 个与后 K 个一一对应
        sim_i_j = torch.diag(sim, self.num_clusters)      # [K]
        sim_j_i = torch.diag(sim, -self.num_clusters)     # [K]
        positives = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)  # [2K, 1]

        # 负样本：去掉对角线和正配对
        mask = self.mask_correlated_samples(N).to(q.device)
        negatives = sim[mask].view(N, -1)  # [2K, 2K-2]

        logits = torch.cat((positives, negatives), dim=1)  # [2K, 2K-1]
        labels = torch.zeros(N, dtype=torch.long, device=q.device)    # 正样本恒在第 0 列

        loss = self.criterion(logits, labels) / N
        return loss


# ====================== Multi-path Consensus Loss ======================
def _info_nce(z1, z2, tau=0.2, valid: torch.Tensor = None):
    """
    z1,z2: [B,D], 内部L2归一。
    valid: [B] bool，仅对 True 的样本计算（用于单视图可见性过滤）
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    if valid is not None:
        idx = valid.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() <= 1:
            return torch.tensor(0.0, device=z1.device)
        z1 = z1[idx]
        z2 = z2[idx]

    B = z1.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=z1.device)

    sim = torch.mm(z1, z2.t()) / tau
    labels = torch.arange(B, device=z1.device)
    loss = F.cross_entropy(sim, labels)
    return loss


def _sym_kl(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=1).mean() + \
           (q * (q.log() - p.log())).sum(dim=1).mean()


class MultiPathConsensusLoss(nn.Module):
    """
    三部分：
      1) 融合对比: INCE(Z_fuse_1, Z_fuse_2)
      2) 单视图 ↔ 融合:
         Σ_v 0.5 * [INCE(Z_uni_1[v], Z_fuse_2) + INCE(Z_uni_2[v], Z_fuse_1)]
         （仅在可见样本上）
      3) 随机掩码融合 ↔ 未掩码融合:
         E_k 0.5 * [INCE(Z_mask_1[k], Z_fuse_1) + INCE(Z_mask_2[k], Z_fuse_2)]
    外加：增强一致性 KL（小权重）
    """

    def __init__(self, temperature=0.2,
                 lambda_fuse=1.0, lambda_uni=1.0, lambda_mask=1.0, lambda_cons=0.05):
        super().__init__()
        self.tau = temperature
        self.l_fuse = lambda_fuse
        self.l_uni = lambda_uni
        self.l_mask = lambda_mask
        self.l_cons = lambda_cons

    def forward(self, aug: dict):
        Z_fuse_1 = aug["Z_fuse_1"]
        Z_fuse_2 = aug["Z_fuse_2"]
        Z_uni_1 = aug["Z_uni_1"]      # list[V]
        Z_uni_2 = aug["Z_uni_2"]      # list[V]
        Z_mask_1 = aug["Z_mask_1"]    # list[K]
        Z_mask_2 = aug["Z_mask_2"]    # list[K]

        Q_fuse_1 = aug["Q_fuse_1"]
        Q_fuse_2 = aug["Q_fuse_2"]
        Q_uni_1 = aug["Q_uni_1"]      # list[V]
        Q_uni_2 = aug["Q_uni_2"]      # list[V]
        Q_mask_1 = aug["Q_mask_1"]
        Q_mask_2 = aug["Q_mask_2"]

        # 兼容两种键名：visible_masks / vis_masks
        visible_masks = aug.get("visible_masks", aug.get("vis_masks", None))

        device = Z_fuse_1.device

        # 1) 融合对比
        L_fuse = _info_nce(Z_fuse_1, Z_fuse_2, tau=self.tau)

        # 2) 单视图 ↔ 融合
        L_uni = torch.tensor(0.0, device=device)
        if visible_masks is None:
            # 如果没有提供 visible_masks，则默认所有视图当前 batch 都可见
            visible_masks = [
                torch.ones(Z_uni_1[v].size(0), dtype=torch.bool, device=device)
                for v in range(len(Z_uni_1))
            ]

        for v in range(len(Z_uni_1)):
            vm = visible_masks[v]  # [B] bool
            if vm.sum() <= 1:
                continue
            L_uni = L_uni + 0.5 * _info_nce(Z_uni_1[v], Z_fuse_2, tau=self.tau, valid=vm)
            L_uni = L_uni + 0.5 * _info_nce(Z_uni_2[v], Z_fuse_1, tau=self.tau, valid=vm)

        # 3) 随机掩码融合 ↔ 未掩码融合
        L_mask = torch.tensor(0.0, device=device)
        if len(Z_mask_1) > 0:
            for k in range(len(Z_mask_1)):
                L_mask = L_mask + 0.5 * _info_nce(Z_mask_1[k], Z_fuse_1, tau=self.tau)
                L_mask = L_mask + 0.5 * _info_nce(Z_mask_2[k], Z_fuse_2, tau=self.tau)
            L_mask = L_mask / float(len(Z_mask_1))

        # 4) 两次增强一致性（软标签 KL）
        L_cons = _sym_kl(Q_fuse_1, Q_fuse_2)

        if Q_mask_1 is not None and Q_mask_2 is not None:
            L_cons = L_cons + _sym_kl(Q_mask_1, Q_mask_2)

        for v in range(len(Q_uni_1)):
            L_cons = L_cons + _sym_kl(Q_uni_1[v], Q_uni_2[v])

        total = self.l_fuse * L_fuse + self.l_uni * L_uni + self.l_mask * L_mask + self.l_cons * L_cons

        return {
            "loss_total": total,
            "L_fuse": L_fuse.detach(),
            "L_uni": L_uni.detach(),
            "L_mask": L_mask.detach(),
            "L_cons": L_cons.detach(),
        }


# Backward-compatible aliases for older scripts.
DeepMVCLoss = SemanticAlignmentLoss
MVInfMaskingLoss = MultiPathConsensusLoss

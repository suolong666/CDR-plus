"""CDR+: geometry-aware consistency-discrepancy recommendation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDRPlus(nn.Module):
    """CDR with type-specific projection, orthogonality and reconstruction.

    Nodes are ordered as tuples followed by objects, exactly as produced by
    ``data_processing.data_param_prepare``.  The projected representations are
    used by both CD loss and inference, as specified by Algorithm 1 of CDR+.
    """

    def __init__(self, args, alpha, beta, temperature):
        super().__init__()
        self.args = args
        self.temperature = temperature
        self.embeds = nn.Embedding(args.n_node, args.embedding_dim)
        self.type_proj = nn.ModuleList(
            [nn.Linear(args.embedding_dim, args.embedding_dim) for _ in range(2)]
        )
        self.type_recon = nn.ModuleList(
            [nn.Linear(args.embedding_dim, args.embedding_dim) for _ in range(2)]
        )
        self.set_contrastive_data(alpha, beta, temperature)
        self.reset_parameters()
        self.latest_losses = {}

    def reset_parameters(self):
        nn.init.normal_(self.embeds.weight, std=1e-4)
        for module in [*self.type_proj, *self.type_recon]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def set_contrastive_data(self, alpha, beta, temperature):
        """Switch CD matrices/temperature between pre-training and fine-tuning."""
        self.temperature = temperature
        self.alpha = torch.as_tensor(alpha.toarray(), dtype=torch.float32, device=self.args.device)
        self.beta = torch.as_tensor(beta.toarray(), dtype=torch.float32, device=self.args.device)

    def projected_embeddings(self):
        """Return Z=[Z_tuple; Z_object], the representation used by CDR+."""
        tuple_embeddings = self.embeds.weight[: self.args.n_head]
        object_embeddings = self.embeds.weight[self.args.n_head :]
        return torch.cat(
            (self.type_proj[0](tuple_embeddings), self.type_proj[1](object_embeddings)), dim=0
        )

    @staticmethod
    def cd_loss(embeddings, nodes, positives, positive_weight, negative_weight, temperature):
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        anchor_embeddings = embeddings[nodes]
        positive_embeddings = embeddings[positives]
        positive_score = (anchor_embeddings * positive_embeddings).sum(dim=-1)
        all_scores = anchor_embeddings @ embeddings.T

        numerator = positive_weight * torch.exp(positive_score / temperature)
        denominator = (negative_weight * torch.exp(all_scores / temperature)).sum(dim=-1)
        return -torch.log(numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)).sum()

    def orthogonality_loss(self, projected):
        tuple_center = projected[: self.args.n_head].mean(dim=0, keepdim=True)
        object_center = projected[self.args.n_head :].mean(dim=0, keepdim=True)
        return F.cosine_similarity(tuple_center, object_center).abs().sum()

    def reconstruction_loss(self, projected):
        tuple_reconstruction = self.type_recon[0](projected[: self.args.n_head])
        object_reconstruction = self.type_recon[1](projected[self.args.n_head :])
        tuple_loss = F.mse_loss(tuple_reconstruction, self.embeds.weight[: self.args.n_head])
        object_loss = F.mse_loss(object_reconstruction, self.embeds.weight[self.args.n_head :])
        return tuple_loss + object_loss

    def l2_loss(self):
        return sum(parameter.pow(2).sum() for parameter in self.parameters()) / 2

    def forward(self, nodes, positives):
        projected = self.projected_embeddings()
        positive_weight = self.alpha[nodes, positives]
        negative_weight = self.beta[nodes]
        cd = self.cd_loss(projected, nodes, positives, positive_weight, negative_weight, self.temperature)
        orthogonality = self.orthogonality_loss(projected)
        reconstruction = self.reconstruction_loss(projected)
        total = (
            cd
            + self.args.lambda_ortho * orthogonality
            + self.args.lambda_recon * reconstruction
            + self.args.gamma * self.l2_loss()
        )
        self.latest_losses = {
            "cd": cd.detach(), "ortho": orthogonality.detach(), "recon": reconstruction.detach()
        }
        return total

    @torch.no_grad()
    def test_forward(self, users):
        projected = self.projected_embeddings()
        users = torch.as_tensor(users, dtype=torch.long, device=projected.device)
        objects = projected[self.args.n_head :]
        return projected[users] @ objects.T

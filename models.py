# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class EdgeImportanceGNN(nn.Module):
    """
    改进点：
      1) 支持 multi-scale message passing：local edge_index + far edge_index_far
      2) 输入维度可变（mesh_dataset.py 已把 PE 拼进 x）
      3) edge_mlp 使用：h_src, h_dst, |h_src-h_dst| + 几何边特征（dihedral/len）
      4) ✅ 加入 LayerNorm + Dropout：降低 importance 噪声，提升训练稳定性
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 64,
        num_layers: int = 3,
        edge_feat_dim: int = 2,
        use_multiscale: bool = True,
        far_scale: float = 0.5,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.edge_feat_dim = int(edge_feat_dim)
        self.use_multiscale = bool(use_multiscale)
        self.far_scale = float(far_scale)
        self.dropout = float(dropout)

        # local conv
        self.convs_local = nn.ModuleList()
        self.convs_local.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs_local.append(GCNConv(hidden_channels, hidden_channels))

        # far conv（可选）
        if self.use_multiscale:
            self.convs_far = nn.ModuleList()
            self.convs_far.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs_far.append(GCNConv(hidden_channels, hidden_channels))
        else:
            self.convs_far = None

        # norms（每层一个）
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

        edge_input_dim = hidden_channels * 3 + self.edge_feat_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, 1),
        )

    @staticmethod
    def _pad_or_trim_edge_attr(edge_attr: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        保证 edge_attr 最后一维 == target_dim：
          - 少了右侧补 0
          - 多了截断
        """
        if edge_attr is None:
            return None
        if target_dim <= 0:
            return None

        cur = int(edge_attr.size(-1))
        if cur == target_dim:
            return edge_attr
        if cur > target_dim:
            return edge_attr[..., :target_dim]

        pad = target_dim - cur
        zeros = edge_attr.new_zeros(edge_attr.size(0), pad)
        return torch.cat([edge_attr, zeros], dim=-1)

    def encode_vertices(self, x: torch.Tensor, edge_index: torch.Tensor, edge_index_far: torch.Tensor = None) -> torch.Tensor:
        """
        多尺度编码：每层都做
          h_local = GCN(h, edge_index)
          h_far   = GCN(h, edge_index_far)  (若存在)
          h = norm(relu(h_local + far_scale*h_far))
        """
        h = x
        for i, conv_local in enumerate(self.convs_local):
            h_local = conv_local(h, edge_index)

            if self.use_multiscale and (edge_index_far is not None) and (edge_index_far.numel() > 0):
                h_far = self.convs_far[i](h, edge_index_far)
                h = h_local + self.far_scale * h_far
            else:
                h = h_local

            h = self.norms[i](h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def _edge_logits_from_embeddings(self, h, src, dst, edge_attr=None):
        h_src = h[src]
        h_dst = h[dst]
        h_diff = torch.abs(h_src - h_dst)

        edge_attr = self._pad_or_trim_edge_attr(edge_attr, self.edge_feat_dim)

        if edge_attr is not None:
            edge_feat = torch.cat([h_src, h_dst, h_diff, edge_attr], dim=-1)
        else:
            if self.edge_feat_dim > 0:
                zeros = h.new_zeros(h_src.size(0), self.edge_feat_dim)
                edge_feat = torch.cat([h_src, h_dst, h_diff, zeros], dim=-1)
            else:
                edge_feat = torch.cat([h_src, h_dst, h_diff], dim=-1)

        logits = self.edge_mlp(edge_feat).squeeze(-1)
        return logits

    def forward(
        self,
        data,
        return_vertex_embeddings: bool = False,
        aggregate_undirected: bool = False,
    ):
        x, edge_index = data.x, data.edge_index
        edge_index_far = getattr(data, "edge_index_far", None)

        h = self.encode_vertices(x, edge_index, edge_index_far=edge_index_far)

        # === 有向边几何特征（映射自无向边） ===
        edge_attr_dir = None
        if self.edge_feat_dim > 0 and hasattr(data, "edge_feature_weight") and hasattr(data, "dir2undir"):
            dir2undir = data.dir2undir.to(h.device)
            geom_list = []

            feat_w = data.edge_feature_weight.to(h.device)
            geom_list.append(feat_w[dir2undir].view(-1, 1))

            if hasattr(data, "edge_length"):
                feat_len = data.edge_length.to(h.device)
                geom_list.append(feat_len[dir2undir].view(-1, 1))

            if len(geom_list) > 0:
                edge_attr_dir = torch.cat(geom_list, dim=-1)

        src, dst = edge_index
        logits_dir = self._edge_logits_from_embeddings(h, src, dst, edge_attr=edge_attr_dir)
        importance_dir = torch.sigmoid(logits_dir)

        out = {"edge_importance_dir": importance_dir}

        # === 无向边重要性（可选） ===
        if aggregate_undirected and hasattr(data, "undirected_edges"):
            undirected_edges = data.undirected_edges.to(h.device)
            u_src = undirected_edges[:, 0]
            u_dst = undirected_edges[:, 1]

            edge_attr_undir = None
            if self.edge_feat_dim > 0 and hasattr(data, "edge_feature_weight"):
                geom_list_u = []
                feat_w_u = data.edge_feature_weight.to(h.device).view(-1, 1)
                geom_list_u.append(feat_w_u)

                if hasattr(data, "edge_length"):
                    feat_len_u = data.edge_length.to(h.device).view(-1, 1)
                    geom_list_u.append(feat_len_u)

                edge_attr_undir = torch.cat(geom_list_u, dim=-1) if len(geom_list_u) > 0 else None

            logits_undir = self._edge_logits_from_embeddings(h, u_src, u_dst, edge_attr=edge_attr_undir)
            importance_undir = torch.sigmoid(logits_undir)
            out["edge_importance_undir"] = importance_undir

        if return_vertex_embeddings:
            out["vertex_embeddings"] = h

        return out

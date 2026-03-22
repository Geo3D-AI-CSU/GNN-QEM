# train_edge_importance.py
import os
import json
import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from mesh_dataset import MeshDataset, MeshGraphData
from models import EdgeImportanceGNN

def edge_contrastive_loss(h, edge_index, num_neg: int = 1) -> torch.Tensor:
    """
    简单对比损失（无监督）：
      - 正样本：真实边 (i, j)
      - 负样本：随机点对 (i, k)，尽量让两者 embedding 不相似

    h:          (N, H) 顶点 embedding
    edge_index: (2, E) 图边
    """
    src, dst = edge_index
    pos_h_src = h[src]
    pos_h_dst = h[dst]

    pos_sim = torch.cosine_similarity(pos_h_src, pos_h_dst, dim=-1)
    pos_loss = (1.0 - pos_sim).mean()

    num_edges = edge_index.size(1)
    num_neg_samples = min(num_edges, 2000)
    if num_neg_samples <= 0:
        return pos_loss

    idx = torch.randint(0, num_edges, (num_neg_samples,), device=h.device)
    neg_i = src[idx]
    num_nodes = h.size(0)
    neg_j = torch.randint(0, num_nodes, (num_neg_samples,), device=h.device)

    neg_h_i = h[neg_i]
    neg_h_j = h[neg_j]
    neg_sim = torch.cosine_similarity(neg_h_i, neg_h_j, dim=-1)
    neg_loss = torch.clamp(neg_sim, min=0).mean()

    return pos_loss + neg_loss


def edge_importance_regularization(
    importance: torch.Tensor,
    edge_index: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """
    让 high-importance 边对应的 embedding 差异更大（弱正则）。
    diff.detach() 防止训练朝“瞎拉开 embedding”崩掉。
    """
    src, dst = edge_index
    diff = (h[src] - h[dst]).pow(2).sum(dim=-1).sqrt()
    return - (importance * diff.detach()).mean()


def sharpness_ranking_loss(
    importance: torch.Tensor,
    target_dir: torch.Tensor,
    margin: float = 0.10,
    high_thresh: float = 0.75,
    low_thresh: float = 0.25,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """
    基于几何锐度（edge_feature_weight）构造 ranking loss（弱先验）：
      high: target_dir >= high_thresh
      low : target_dir <= low_thresh
      L_rank = mean( relu(margin - (imp_high - imp_low)) )
    """
    with torch.no_grad():
        pos_idx = (target_dir >= high_thresh).nonzero(as_tuple=True)[0]
        neg_idx = (target_dir <= low_thresh).nonzero(as_tuple=True)[0]

    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return importance.new_tensor(0.0)

    k = min(max_pairs, pos_idx.numel(), neg_idx.numel())
    if k <= 0:
        return importance.new_tensor(0.0)

    perm_pos = pos_idx[torch.randperm(pos_idx.numel(), device=importance.device)[:k]]
    perm_neg = neg_idx[torch.randperm(neg_idx.numel(), device=importance.device)[:k]]

    diff = importance[perm_pos] - importance[perm_neg]
    return F.relu(margin - diff).mean()


def feature_hinge_loss(
    importance: torch.Tensor,
    target_dir: torch.Tensor,
    edge_is_boundary_dir: torch.Tensor,
    min_imp: float = 0.60,
    dihedral_feat_thresh: float = 0.70,
) -> torch.Tensor:
    """
    只在“边界边 or 高二面角边”上约束 importance 不要太低：
      L_hinge = mean( relu(min_imp - imp[e]) ) for e in feature_edges
    """
    with torch.no_grad():
        feat_mask = (edge_is_boundary_dir > 0.5) | (target_dir >= dihedral_feat_thresh)

    if int(feat_mask.sum().item()) == 0:
        return importance.new_tensor(0.0)

    return F.relu(min_imp - importance[feat_mask]).mean()


def smoothness_loss_incident(
    importance: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    use_abs: bool = True,
) -> torch.Tensor:
    """
    ✅ O(E) 的平滑正则（训练稳，避免 importance 乱跳导致折叠乱三角）：
    同一顶点的 incident edges importance 应该相对一致

      - 先按 src 聚合每个顶点的 mean importance
      - 再让每条边的 importance 接近 (mean[src] + mean[dst]) / 2
    """
    src, dst = edge_index
    ones = torch.ones_like(importance)

    sum_src = torch.zeros((num_nodes,), device=importance.device)
    cnt_src = torch.zeros((num_nodes,), device=importance.device)
    sum_src.scatter_add_(0, src, importance)
    cnt_src.scatter_add_(0, src, ones)
    mean_src = sum_src / (cnt_src + 1e-6)

    sum_dst = torch.zeros((num_nodes,), device=importance.device)
    cnt_dst = torch.zeros((num_nodes,), device=importance.device)
    sum_dst.scatter_add_(0, dst, importance)
    cnt_dst.scatter_add_(0, dst, ones)
    mean_dst = sum_dst / (cnt_dst + 1e-6)

    mean_edge = 0.5 * (mean_src[src] + mean_dst[dst])
    diff = importance - mean_edge
    return diff.abs().mean() if use_abs else (diff * diff).mean()


def importance_entropy_regularization(importance: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    ✅ 替代旧的 (mean-0.35)^2：形状无关、更稳健的分布正则
    - 目标：避免 importance 塌缩为全 0/全 1
    - 做法：最大化 Bernoulli entropy（训练里以 loss 形式加入：loss += w * (-entropy)）

    返回：loss_entropy = -H(p) （越小越好）
    """
    p = importance.clamp(eps, 1.0 - eps)
    ent = - (p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()
    return -ent


def _safe_np_colorize(values_01, cmap_name: str = "jet"):
    """
    把 [0,1] 映射到 RGB(0-255)。优先用 matplotlib；没有则用简易渐变。
    """
    v = values_01.astype("float32")
    v = v.clip(0.0, 1.0)

    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(cmap_name)
        c = cmap(v)[:, :3]
        return (c * 255.0).astype("uint8")
    except Exception:
        # fallback: blue->red
        r = v
        g = 1.0 - (v - 0.5).clip(0, 0.5) * 2.0
        b = 1.0 - v
        c = torch.stack([
            torch.from_numpy(r),
            torch.from_numpy(g.astype("float32")),
            torch.from_numpy(b),
        ], dim=1).numpy()
        c = c.clip(0.0, 1.0)
        return (c * 255.0).astype("uint8")


def save_importance_map_from_graph(
    data: MeshGraphData,
    importance_undir: torch.Tensor,
    out_path: str,
    cmap: str = "jet",
) -> None:
    """
    将无向边 importance 映射到顶点颜色，并导出带 vertex_colors 的 PLY（或 OBJ，但 PLY 更稳定）。
    data.x 必须前 3 维是 xyz；data.faces 为三角面；data.undirected_edges 对齐 importance_undir。
    """
    import numpy as np
    import trimesh

    v = data.x[:, :3].detach().cpu().numpy().astype(np.float64)
    f = data.faces.detach().cpu().numpy().astype(np.int64)
    e = data.undirected_edges.detach().cpu().numpy().astype(np.int64)
    imp = importance_undir.detach().cpu().numpy().reshape(-1).astype(np.float32)

    n = v.shape[0]
    vsum = np.zeros((n,), dtype=np.float32)
    vcnt = np.zeros((n,), dtype=np.float32)
    if e.shape[0] > 0 and imp.shape[0] == e.shape[0]:
        a = e[:, 0]
        b = e[:, 1]
        np.add.at(vsum, a, imp)
        np.add.at(vsum, b, imp)
        np.add.at(vcnt, a, 1.0)
        np.add.at(vcnt, b, 1.0)

    vimp = np.divide(vsum, vcnt, out=np.zeros_like(vsum), where=vcnt > 0)
    colors = _safe_np_colorize(vimp, cmap_name=cmap)

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.visual.vertex_colors = colors

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mesh.export(out_path)


@dataclass
class LossWeights:
    contrast: float = 1.0
    hinge: float = 0.7
    smooth: float = 0.2
    rank: float = 0.25
    imp_entropy: float = 0.03   # ✅ 新：分布正则（越大越避免塌缩，但也更“均匀”）
    feat_reg: float = 0.02


def train(
    root: str = ".",
    mesh_dir: str = "data_tosc/toscahires_obj",
    epochs: int = 50,
    batch_size: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
    ckpt_path: str = "checkpoints/edge_gnn_tosc.pt",
    bidirectional_graph: bool = True,

    # dataset / graph settings（对齐 mesh_dataset.py）
    pe_dim: int = 16,
    use_far_graph: bool = True,
    max_far_neighbors: int = 12,

    # model multiscale settings（对齐 models.py）
    use_multiscale: bool = True,
    far_scale: float = 0.5,
    dropout: float = 0.15,

    # ablation
    no_dihedral_feat: bool = False,
    shuffle_dihedral_feat: bool = False,

    # weak-prior hyper
    dihedral_feat_thresh: float = 0.70,
    hinge_min_imp: float = 0.60,

    # perf
    num_workers: Optional[int] = None,

    # monitoring
    log_root: str = "logs/train",
    run_name: Optional[str] = None,
    tb_flush_secs: int = 10,
    vis_every: int = 5,
    vis_index: int = 0,
    vis_dir: str = "logs/vis",
    loss_w: LossWeights = LossWeights(),
):
    """
    训练 EdgeImportanceGNN（弱先验 + 结构无监督 + ranking）并加入可视化监控：
      1) TensorBoard：Loss 曲线、importance 分布、grad norm、学习率等
      2) Importance 热力图导出：每 N 个 epoch 导出一次带颜色的 PLY 便于 MeshLab 观察
    """
    dataset = MeshDataset(
        root=root,
        mesh_dir=mesh_dir,
        bidirectional=bidirectional_graph,
        pe_dim=pe_dim,
        use_far_graph=use_far_graph,
        max_far_neighbors=max_far_neighbors,
    )

    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4

    pin_memory = str(device).startswith("cuda")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=(int(num_workers) > 0),
    )

    in_channels = dataset.num_node_features
    model = EdgeImportanceGNN(
        in_channels=in_channels,
        hidden_channels=64,
        num_layers=3,
        edge_feat_dim=2,
        use_multiscale=use_multiscale,
        far_scale=far_scale,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # --- TensorBoard writer ---
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_name is None:
        run_name = f"{ts}_pe{pe_dim}_far{int(use_far_graph)}_k{max_far_neighbors}_bi{int(bidirectional_graph)}"

    log_dir = os.path.join(log_root, run_name)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=int(tb_flush_secs))

    # dump config
    cfg = dict(
        version=TRAINER_VERSION,
        root=root,
        mesh_dir=mesh_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        ckpt_path=ckpt_path,
        bidirectional_graph=bidirectional_graph,
        pe_dim=pe_dim,
        use_far_graph=use_far_graph,
        max_far_neighbors=max_far_neighbors,
        use_multiscale=use_multiscale,
        far_scale=far_scale,
        dropout=dropout,
        no_dihedral_feat=no_dihedral_feat,
        shuffle_dihedral_feat=shuffle_dihedral_feat,
        dihedral_feat_thresh=dihedral_feat_thresh,
        hinge_min_imp=hinge_min_imp,
        num_workers=int(num_workers),
        loss_weights=asdict(loss_w),
        log_dir=log_dir,
    )
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[Trainer] Version: {TRAINER_VERSION}")
    print(f"[Trainer] Device: {device} | pin_memory={pin_memory} | num_workers={num_workers}")
    print(f"[Trainer] InChannels={in_channels}")
    print(f"[Trainer] DatasetCache: {dataset.processed_paths[0]}")
    print(f"[Trainer] TensorBoard log_dir: {log_dir}")
    print(f"[Trainer] LossWeights: {loss_w}")

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        last_items = {}

        for data in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", ncols=100):
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            # --- Ablation：处理 edge_feature_weight（影响模型 edge_attr）---
            if hasattr(data, "edge_feature_weight"):
                if no_dihedral_feat:
                    data.edge_feature_weight = torch.zeros_like(data.edge_feature_weight)
                elif shuffle_dihedral_feat:
                    w = data.edge_feature_weight
                    perm = torch.randperm(w.numel(), device=w.device)
                    data.edge_feature_weight = w[perm]

            out = model(data, return_vertex_embeddings=True, aggregate_undirected=False)
            importance = out["edge_importance_dir"]
            h = out["vertex_embeddings"]

            contrast_loss = edge_contrastive_loss(h, data.edge_index)

            # ✅ 新：entropy reg（替代 mean-0.35 先验）
            ent_reg = importance_entropy_regularization(importance)

            # weak prior：hinge + rank
            if hasattr(data, "edge_feature_weight") and hasattr(data, "dir2undir"):
                feat_undir = data.edge_feature_weight.to(device)
                dir2undir = data.dir2undir.to(device)
                target_dir = feat_undir[dir2undir]

                if hasattr(data, "edge_is_boundary"):
                    b_undir = data.edge_is_boundary.to(device)
                    edge_is_boundary_dir = b_undir[dir2undir]
                else:
                    edge_is_boundary_dir = torch.zeros_like(target_dir)

                hinge_loss = feature_hinge_loss(
                    importance=importance,
                    target_dir=target_dir.detach(),
                    edge_is_boundary_dir=edge_is_boundary_dir.detach(),
                    min_imp=hinge_min_imp,
                    dihedral_feat_thresh=dihedral_feat_thresh,
                )

                rank_loss = sharpness_ranking_loss(
                    importance=importance,
                    target_dir=target_dir.detach(),
                    margin=0.10,
                    high_thresh=max(0.75, dihedral_feat_thresh),
                    low_thresh=0.25,
                    max_pairs=4096,
                )
            else:
                hinge_loss = torch.tensor(0.0, device=device)
                rank_loss = torch.tensor(0.0, device=device)

            smooth_loss = smoothness_loss_incident(
                importance=importance,
                edge_index=data.edge_index,
                num_nodes=int(data.num_nodes),
                use_abs=True,
            )

            feat_reg = edge_importance_regularization(importance, data.edge_index, h)

            loss = (
                loss_w.contrast * contrast_loss
                + loss_w.hinge * hinge_loss
                + loss_w.smooth * smooth_loss
                + loss_w.rank * rank_loss
                + loss_w.imp_entropy * ent_reg
                + loss_w.feat_reg * feat_reg
            )

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).item())
            optimizer.step()

            total_loss += float(loss.item())
            global_step += 1

            # --- per-step logging（可选：这里控制频率）---
            if global_step % 20 == 0:
                with torch.no_grad():
                    imp_mean = float(importance.mean().item())
                    imp_std = float(importance.std().item())
                writer.add_scalar("Step/Loss", float(loss.item()), global_step)
                writer.add_scalar("Step/GradNorm", grad_norm, global_step)
                writer.add_scalar("Step/ImportanceMean", imp_mean, global_step)
                writer.add_scalar("Step/ImportanceStd", imp_std, global_step)

            last_items = dict(
                loss=float(loss.item()),
                contrast=float(contrast_loss.item()),
                hinge=float(hinge_loss.item()),
                smooth=float(smooth_loss.item()),
                rank=float(rank_loss.item()),
                ent_reg=float(ent_reg.item()),
                grad_norm=grad_norm,
            )

        avg_loss = total_loss / max(1, len(loader))

        # --- epoch logging ---
        writer.add_scalar("Loss/Total", avg_loss, epoch)
        writer.add_scalar("Loss/Contrastive", last_items.get("contrast", 0.0), epoch)
        writer.add_scalar("Loss/Hinge", last_items.get("hinge", 0.0), epoch)
        writer.add_scalar("Loss/Smooth", last_items.get("smooth", 0.0), epoch)
        writer.add_scalar("Loss/Rank", last_items.get("rank", 0.0), epoch)
        writer.add_scalar("Loss/EntropyReg", last_items.get("ent_reg", 0.0), epoch)
        writer.add_scalar("Train/GradNorm", last_items.get("grad_norm", 0.0), epoch)
        writer.add_scalar("Train/LR", float(optimizer.param_groups[0]["lr"]), epoch)

        # importance distribution histogram (pick last batch stats if possible)
        try:
            with torch.no_grad():
                # 用 dataset[vis_index] 重新推一遍，避免 batch 合并的影响
                sample = dataset[int(max(0, min(vis_index, len(dataset)-1)))]
                sample = sample.to(device)
                out_u = model(sample, return_vertex_embeddings=False, aggregate_undirected=True)
                imp_undir = out_u["edge_importance_undir"]
                writer.add_histogram("Importance/Undirected", imp_undir.detach().cpu(), epoch)
                writer.add_scalar("Importance/UndirectedMean", float(imp_undir.mean().item()), epoch)
                writer.add_scalar("Importance/UndirectedStd", float(imp_undir.std().item()), epoch)
        except Exception:
            pass

        print(
            f"[Epoch {epoch}] avg_loss={avg_loss:.4f} | "
            f"contrast={last_items.get('contrast', 0):.4f} hinge={last_items.get('hinge', 0):.4f} "
            f"smooth={last_items.get('smooth', 0):.4f} rank={last_items.get('rank', 0):.4f} "
            f"ent_reg={last_items.get('ent_reg', 0):.4f}"
        )

        # --- checkpoint ---
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

        # --- visualization: export PLY ---
        if vis_every > 0 and (epoch % int(vis_every) == 0):
            try:
                os.makedirs(vis_dir, exist_ok=True)
                sample = dataset[int(max(0, min(vis_index, len(dataset)-1)))]
                sample = sample.to(device)
                model.eval()
                with torch.no_grad():
                    out_u = model(sample, return_vertex_embeddings=False, aggregate_undirected=True)
                    imp_undir = out_u["edge_importance_undir"]
                # 保存路径：包含 epoch
                out_ply = os.path.join(vis_dir, f"importance_epoch{epoch:03d}.ply")
                save_importance_map_from_graph(sample, imp_undir, out_ply, cmap="jet")
                print(f"[Vis] importance heatmap exported: {out_ply}")
                writer.add_text("Vis/Latest", out_ply, epoch)
            except Exception as e:
                print(f"[Vis] export failed: {e}")

    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--mesh_dir", type=str, default="data_tosc/toscahires_obj")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ckpt", type=str, default="checkpoints/edge_gnn_tosc.pt")

    # graph
    parser.add_argument("--bidirectional_graph", type=int, default=1)
    parser.add_argument("--pe_dim", type=int, default=16)
    parser.add_argument("--use_far_graph", type=int, default=1)
    parser.add_argument("--max_far_neighbors", type=int, default=12)

    # model
    parser.add_argument("--use_multiscale", type=int, default=1)
    parser.add_argument("--far_scale", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.15)

    # ablation
    parser.add_argument("--no_dihedral_feat", type=int, default=0)
    parser.add_argument("--shuffle_dihedral_feat", type=int, default=0)

    # weak-prior hyper
    parser.add_argument("--dihedral_feat_thresh", type=float, default=0.70)
    parser.add_argument("--hinge_min_imp", type=float, default=0.60)

    # perf
    parser.add_argument("--num_workers", type=int, default=-1)

    # monitoring
    parser.add_argument("--log_root", type=str, default="logs/train")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--vis_every", type=int, default=5)
    parser.add_argument("--vis_index", type=int, default=0)
    parser.add_argument("--vis_dir", type=str, default="logs/vis")

    args = parser.parse_args()
    nw = None if args.num_workers < 0 else int(args.num_workers)

    train(
        root=args.root,
        mesh_dir=args.mesh_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        ckpt_path=args.ckpt,
        bidirectional_graph=bool(args.bidirectional_graph),

        pe_dim=args.pe_dim,
        use_far_graph=bool(args.use_far_graph),
        max_far_neighbors=args.max_far_neighbors,

        use_multiscale=bool(args.use_multiscale),
        far_scale=args.far_scale,
        dropout=args.dropout,

        no_dihedral_feat=bool(args.no_dihedral_feat),
        shuffle_dihedral_feat=bool(args.shuffle_dihedral_feat),

        dihedral_feat_thresh=args.dihedral_feat_thresh,
        hinge_min_imp=args.hinge_min_imp,

        num_workers=nw,

        log_root=args.log_root,
        run_name=args.run_name,
        vis_every=args.vis_every,
        vis_index=args.vis_index,
        vis_dir=args.vis_dir,
    )

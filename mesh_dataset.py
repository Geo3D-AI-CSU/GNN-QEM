# mesh_dataset.py
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
import trimesh


class MeshGraphData(Data):


    def __inc__(self, key, value, *args, **kwargs):
        if key == "dir2undir":
            if hasattr(self, "undirected_edges"):
                return int(self.undirected_edges.size(0))
            return 0
        if key in ("undirected_edges", "faces", "edge_index_far"):
            return int(self.num_nodes)
        return super().__inc__(key, value, *args, **kwargs)


def _compute_vertex_boundary_flag(
    num_vertices: int,
    undirected_edges: np.ndarray,
    edge_is_boundary: np.ndarray,
) -> np.ndarray:
    vb = np.zeros((num_vertices,), dtype=np.float32)
    if undirected_edges.shape[0] == 0:
        return vb
    b_edges = undirected_edges[edge_is_boundary.astype(bool)]
    if b_edges.shape[0] == 0:
        return vb
    vb[b_edges[:, 0]] = 1.0
    vb[b_edges[:, 1]] = 1.0
    return vb


def _build_2hop_far_edges(
    num_vertices: int,
    undirected_edges: np.ndarray,
    vertices: np.ndarray,
    max_far_neighbors: int = 12,
) -> np.ndarray:
    """
    构建 2-hop 的“远邻无向边”（用于 multi-scale message passing）。

    - 2-hop：i 的邻居的邻居（排除自身、排除 1-hop 邻居）
    - 为避免爆炸，对每个点最多保留 max_far_neighbors 条（按欧氏距离由近到远取前 max_far_neighbors）
    返回：(E_far_undir, 2) int64，且 a<b 的无向边集合（去重）
    """
    if num_vertices <= 0 or undirected_edges.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    neigh = [set() for _ in range(num_vertices)]
    for u, v in undirected_edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        neigh[u].add(v)
        neigh[v].add(u)

    far_edges_set = set()

    for i in range(num_vertices):
        n1 = neigh[i]
        if not n1:
            continue
        n2 = set()
        for j in n1:
            n2.update(neigh[j])

        # 去掉自身和 1-hop
        n2.discard(i)
        n2.difference_update(n1)
        if not n2:
            continue

        cand = list(n2)

        # 限制数量：按距离取最近的 max_far_neighbors
        if max_far_neighbors is not None and max_far_neighbors > 0 and len(cand) > max_far_neighbors:
            pi = vertices[i]
            dist = np.linalg.norm(vertices[cand] - pi[None, :], axis=1)
            idx = np.argsort(dist)[:max_far_neighbors]
            cand = [cand[k] for k in idx]

        for j in cand:
            a, b = (i, int(j))
            if a == b:
                continue
            if a > b:
                a, b = b, a
            far_edges_set.add((a, b))

    if not far_edges_set:
        return np.zeros((0, 2), dtype=np.int64)

    return np.array(sorted(list(far_edges_set)), dtype=np.int64)


def _canonicalize_eigenvector_sign(vecs: np.ndarray) -> np.ndarray:
    """
    Laplacian 特征向量存在 ±v 等价性；不做处理会导致不同样本/不同次运行出现符号翻转。
    这里做一个常用的 sign canonicalization：
      对每一列，找到绝对值最大的元素，其符号强制为正。
    """
    if vecs.size == 0:
        return vecs
    vecs = vecs.copy()
    for k in range(vecs.shape[1]):
        col = vecs[:, k]
        idx = int(np.argmax(np.abs(col)))
        if col[idx] < 0:
            vecs[:, k] = -col
    return vecs


def _compute_laplacian_pe(
    num_vertices: int,
    undirected_edges: np.ndarray,
    pe_dim: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """
    计算 Laplacian Eigenvectors 作为 positional encoding (PE)。
    返回 shape: (N, pe_dim) float32

    说明：
      - 优先使用 scipy.sparse.linalg.eigsh（更快更省内存）
      - 若 scipy 不可用或求解失败：小图(N<=4000)走 dense eigh，否则返回 0
      - ✅ 增加 sign canonicalization，避免 ±v 翻转导致训练不稳定
      - ✅ 提供确定性的 v0（seed），减少 ARPACK 的随机性
    """
    if pe_dim <= 0 or num_vertices <= 0:
        return np.zeros((num_vertices, 0), dtype=np.float32)

    if undirected_edges.shape[0] == 0:
        return np.zeros((num_vertices, pe_dim), dtype=np.float32)

    u = undirected_edges[:, 0].astype(np.int64)
    v = undirected_edges[:, 1].astype(np.int64)

    # --- scipy sparse path ---
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        data = np.ones((u.shape[0] * 2,), dtype=np.float64)
        row = np.concatenate([u, v], axis=0)
        col = np.concatenate([v, u], axis=0)

        A = sp.coo_matrix((data, (row, col)), shape=(num_vertices, num_vertices)).tocsr()
        deg = np.asarray(A.sum(axis=1)).reshape(-1)
        deg = np.maximum(deg, 1e-12)
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        D_inv_sqrt = sp.diags(d_inv_sqrt)

        I = sp.eye(num_vertices, format="csr")
        L = I - (D_inv_sqrt @ A @ D_inv_sqrt)

        k = min(pe_dim + 1, num_vertices - 1)  # +1 丢掉常数特征向量
        if k <= 0:
            return np.zeros((num_vertices, pe_dim), dtype=np.float32)

        rng = np.random.default_rng(int(seed))
        v0 = rng.normal(size=(num_vertices,)).astype(np.float64)

        vals, vecs = spla.eigsh(L, k=k, which="SM", v0=v0)

        vecs = vecs[:, 1:pe_dim + 1]
        if vecs.shape[1] < pe_dim:
            pad = pe_dim - vecs.shape[1]
            vecs = np.concatenate([vecs, np.zeros((num_vertices, pad), dtype=np.float64)], axis=1)

        vecs = _canonicalize_eigenvector_sign(vecs)

        pe = vecs.astype(np.float32)

        # normalize per-dimension: zero-mean unit-std
        pe = pe - pe.mean(axis=0, keepdims=True)
        pe_std = pe.std(axis=0, keepdims=True) + 1e-6
        pe = pe / pe_std
        return pe

    except Exception:
        # --- dense fallback ---
        if num_vertices > 4000:
            return np.zeros((num_vertices, pe_dim), dtype=np.float32)

        A = np.zeros((num_vertices, num_vertices), dtype=np.float64)
        A[u, v] = 1.0
        A[v, u] = 1.0
        deg = A.sum(axis=1)
        deg = np.maximum(deg, 1e-12)
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(num_vertices, dtype=np.float64) - D_inv_sqrt @ A @ D_inv_sqrt

        vals, vecs = np.linalg.eigh(L)
        vecs = vecs[:, 1:pe_dim + 1]
        if vecs.shape[1] < pe_dim:
            pad = pe_dim - vecs.shape[1]
            vecs = np.concatenate([vecs, np.zeros((num_vertices, pad), dtype=np.float64)], axis=1)

        vecs = _canonicalize_eigenvector_sign(vecs)

        pe = vecs.astype(np.float32)
        pe = pe - pe.mean(axis=0, keepdims=True)
        pe_std = pe.std(axis=0, keepdims=True) + 1e-6
        pe = pe / pe_std
        return pe


def mesh_to_graph(
    mesh: trimesh.Trimesh,
    bidirectional: bool = True,
    pe_dim: int = 16,
    use_far_graph: bool = True,
    max_far_neighbors: int = 12,
) -> Data:
    """
    将 Trimesh 转成 PyG 的 Data

    输出：
      - x: (N, F) 顶点特征：
            [x,y,z, nx,ny,nz, valence_norm, vertex_is_boundary, PE_k...]
      - edge_index: (2, E_dir) 有向边（仅 mesh 1-hop 边）
      - edge_index_far: (2, E_far_dir) 远邻有向边（2-hop），可用于 multi-scale message passing
      - faces: (M, 3)
      - undirected_edges: (E_undir, 2)
      - edge_dihedral: (E_undir,)
      - edge_is_boundary: (E_undir,) ∈ {0,1}
      - edge_feature_weight: (E_undir,) ∈ [0,1]  (dihedral/pi；边界置为1)
      - edge_length: (E_undir,) 归一化边长（除以平均边长）
      - dir2undir: (E_dir,) 有向边 -> 无向边索引
      - pe: (N, pe_dim) （单独保存，方便 ablation）
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32)  # (N, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64)          # (M, 3)

    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]

    # === 1. 面法线 ===
    if num_faces > 0:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn_norm = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        face_normals = fn / fn_norm
    else:
        face_normals = np.zeros((0, 3), dtype=np.float32)

    # === 1.1 顶点法线（面法线加权平均） ===
    vertex_normals = np.zeros((num_vertices, 3), dtype=np.float32)
    if num_faces > 0:
        for fi, f in enumerate(faces):
            i, j, k = int(f[0]), int(f[1]), int(f[2])
            n = face_normals[fi]
            vertex_normals[i] += n
            vertex_normals[j] += n
            vertex_normals[k] += n

        vn_norm = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        nonzero = vn_norm[:, 0] > 1e-12
        vertex_normals[nonzero] /= vn_norm[nonzero]

    # === 2. 无向边 + 邻接面 ===
    edge2faces: Dict[Tuple[int, int], List[int]] = {}
    for fi, f in enumerate(faces):
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        for a, b in ((i, j), (j, k), (k, i)):
            if a > b:
                a, b = b, a
            edge2faces.setdefault((a, b), []).append(fi)

    undirected_edges = np.array(list(edge2faces.keys()), dtype=np.int64) if len(edge2faces) > 0 else np.zeros((0, 2), dtype=np.int64)
    num_undir = undirected_edges.shape[0]

    # === 2.1 valence ===
    valence = np.zeros((num_vertices,), dtype=np.int32)
    for uu, vv in undirected_edges:
        valence[int(uu)] += 1
        valence[int(vv)] += 1
    valence = valence.astype(np.float32)
    valence_norm = valence / (valence.max() + 1e-6) if (num_vertices > 0 and valence.max() > 0) else valence

    # === 3. 二面角 / 边界 / 特征权重 / 边长 ===
    edge_dihedral = np.zeros((num_undir,), dtype=np.float32)
    edge_length = np.zeros((num_undir,), dtype=np.float32)
    edge_is_boundary = np.zeros((num_undir,), dtype=np.float32)

    for idx, (uu, vv) in enumerate(undirected_edges):
        u_i, v_i = int(uu), int(vv)
        faces_idx = edge2faces.get((u_i, v_i), [])

        if len(faces_idx) == 2:
            n1 = face_normals[faces_idx[0]]
            n2 = face_normals[faces_idx[1]]
            cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle = float(np.arccos(cos_theta))
            edge_is_boundary[idx] = 0.0
        else:
            angle = float(np.pi)  # 边界/非流形 -> 强特征
            edge_is_boundary[idx] = 1.0

        edge_dihedral[idx] = angle
        p_u = vertices[u_i]
        p_v = vertices[v_i]
        edge_length[idx] = float(np.linalg.norm(p_u - p_v))

    if num_undir > 0:
        edge_feature_weight = edge_dihedral / np.pi
        edge_feature_weight = np.maximum(edge_feature_weight, edge_is_boundary)  # 边界更强
        mean_len = float(edge_length.mean()) if float(edge_length.mean()) > 1e-12 else 1.0
        edge_length_norm = edge_length / mean_len
    else:
        edge_feature_weight = edge_dihedral
        edge_length_norm = edge_length

    # === 3.1 顶点是否边界 ===
    vertex_is_boundary = _compute_vertex_boundary_flag(num_vertices, undirected_edges, edge_is_boundary)

    # === 3.2 Laplacian PE ===
    pe = _compute_laplacian_pe(num_vertices, undirected_edges, pe_dim=pe_dim)

    # === 4. edge_index（有向）+ dir2undir ===
    if bidirectional and num_undir > 0:
        src = undirected_edges[:, 0]
        dst = undirected_edges[:, 1]
        src_dir = np.concatenate([src, dst], axis=0)
        dst_dir = np.concatenate([dst, src], axis=0)
        edge_index_np = np.stack([src_dir, dst_dir], axis=0)
        edge_index = torch.from_numpy(edge_index_np).long()
        dir2undir = torch.arange(num_undir, dtype=torch.long).repeat(2)
    else:
        edge_index_np = undirected_edges.T if num_undir > 0 else np.zeros((2, 0), dtype=np.int64)
        edge_index = torch.from_numpy(edge_index_np).long()
        dir2undir = torch.arange(num_undir, dtype=torch.long)

    # === 4.1 远邻图（2-hop） ===
    if use_far_graph and num_vertices > 0 and num_undir > 0:
        far_undir = _build_2hop_far_edges(
            num_vertices=num_vertices,
            undirected_edges=undirected_edges,
            vertices=vertices,
            max_far_neighbors=max_far_neighbors,
        )
        if far_undir.shape[0] > 0:
            if bidirectional:
                s = far_undir[:, 0]
                d = far_undir[:, 1]
                s_dir = np.concatenate([s, d], axis=0)
                d_dir = np.concatenate([d, s], axis=0)
                edge_index_far = torch.from_numpy(np.stack([s_dir, d_dir], axis=0)).long()
            else:
                edge_index_far = torch.from_numpy(far_undir.T).long()
        else:
            edge_index_far = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index_far = torch.zeros((2, 0), dtype=torch.long)

    # === 5. 顶点特征 x ===
    if num_vertices > 0:
        base = np.concatenate(
            [vertices, vertex_normals, valence_norm[:, None], vertex_is_boundary[:, None]],
            axis=1,
        ).astype(np.float32)
        x_np = np.concatenate([base, pe.astype(np.float32)], axis=1) if pe_dim > 0 else base
    else:
        x_np = np.zeros((0, 8 + max(pe_dim, 0)), dtype=np.float32)

    data = MeshGraphData(x=torch.from_numpy(x_np), edge_index=edge_index)
    data.faces = torch.from_numpy(faces)
    data.undirected_edges = torch.from_numpy(undirected_edges)
    data.edge_dihedral = torch.from_numpy(edge_dihedral.astype(np.float32))
    data.edge_is_boundary = torch.from_numpy(edge_is_boundary.astype(np.float32))
    data.edge_feature_weight = torch.from_numpy(edge_feature_weight.astype(np.float32))
    data.edge_length = torch.from_numpy(edge_length_norm.astype(np.float32))
    data.dir2undir = dir2undir
    data.vertex_is_boundary = torch.from_numpy(vertex_is_boundary.astype(np.float32))
    data.pe = torch.from_numpy(pe.astype(np.float32)) if pe_dim > 0 else torch.zeros((num_vertices, 0), dtype=torch.float32)
    data.edge_index_far = edge_index_far

    return data


class MeshDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        mesh_dir: str = "data/meshes",
        transform=None,
        pre_transform=None,
        bidirectional: bool = True,
        pe_dim: int = 16,
        use_far_graph: bool = True,
        max_far_neighbors: int = 12,
    ):
        self.mesh_dir = os.path.join(root, mesh_dir)
        self.bidirectional = bool(bidirectional)
        self.pe_dim = int(pe_dim)
        self.use_far_graph = bool(use_far_graph)
        self.max_far_neighbors = int(max_far_neighbors)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # ✅ 必修：把关键参数写进文件名，防止加载旧缓存导致“你以为改了，其实没生效”
        name = (
            f"mesh_graph_pe{self.pe_dim}"
            f"_far{int(self.use_far_graph)}"
            f"_k{self.max_far_neighbors}"
            f"_bi{int(self.bidirectional)}"
            f".pt"
        )
        return [name]

    def download(self):
        pass

    def process(self):
        data_list = []
        if not os.path.isdir(self.mesh_dir):
            raise FileNotFoundError(f"Mesh dir not found: {self.mesh_dir}")

        try:
            from tqdm import tqdm
            it = tqdm(os.listdir(self.mesh_dir), desc="[MeshDataset] Processing", ncols=100)
        except Exception:
            it = os.listdir(self.mesh_dir)

        for fname in it:
            if not fname.lower().endswith(".obj"):
                continue
            path = os.path.join(self.mesh_dir, fname)

            # ✅ 对齐推理：process=False（避免 Trimesh 自动“修复”破坏索引一致性）
            mesh = trimesh.load(path, force="mesh", process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump(concatenate=True)
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)

            data = mesh_to_graph(
                mesh,
                bidirectional=self.bidirectional,
                pe_dim=self.pe_dim,
                use_far_graph=self.use_far_graph,
                max_far_neighbors=self.max_far_neighbors,
            )
            data.name = fname
            data_list.append(data)

        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

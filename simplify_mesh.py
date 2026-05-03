# simplify_mesh.py
# -*- coding: utf-8 -*-
"""
GNN-Guided QEM Simplification: An Energy-Constrained Framework
(Research Version - Paper Submission Ready)

Methodology (Variational Framework):
1. Energy-Constrained Simplification:
   - We formulate simplification as a Multi-Objective Energy Minimization problem.
   - Total Energy: E = E_geom + E_semantic
   - E_geom (QEM cost) is highly heavy-tailed; we robustly normalize it using the unbiased
     Median of the INITIAL state (Two-Pass Normalization) to provide an absolute, scale-invariant
     geometric reference throughout the entire simplification lifecycle.
2. Smooth Additive Barrier (E_semantic):
   - We introduce a continuous, gate-aware additive barrier rather than a hard cutoff.
   - E_semantic = λ * [ max(0, (imp - gate) / (1 - gate)) ]^2
   - Because E_geom is normalized, λ acts as a dataset-agnostic relative scaling factor.
   - The additive penalty ensures strict topological preservation of features even in
     geometrically flat regions (where E_geom ≈ 0).
3. Orthogonal Dynamic Scheduling:
   - Global Progress (t) drives the non-linear relaxation of the continuous gate.
   - Adaptive Hybrid Strategy: The interpolation weight (alpha) is linearly tied to global
     progress, cleanly decoupling the strategy mixing from the feature protection annealing.

Robustness & Extreme Performance:
- Unbiased Priority Queuing: Tie-breaking strictly relies on unique Edge IDs.
- Allocation-Minimized Topology Checks (O(1) Auxiliary Memory): Uses optimized Pythonic
  scalar loops for 1-ring manifold checks, minimizing heap object allocations.
- 50% reduction in np.cross operations via algebraically equivalent tri_quality_with_area.
"""

from __future__ import annotations

import os
import heapq
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

from mesh_dataset import mesh_to_graph
from models import EdgeImportanceGNN

# === 全局缓存与预计算常量 ===
_EYE3 = np.eye(3, dtype=np.float64) * 1e-12
_SQRT3_4 = 4.0 * math.sqrt(3.0)

# Penalty Strength for the Additive Barrier
BARRIER_LAMBDA = 100.0


def _to_1d(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        a = a.reshape(-1)
    return a


def save_importance_map(
        mesh: "trimesh.Trimesh",
        undirected_edges: np.ndarray,
        importance_undir: np.ndarray,
        output_path: str,
        cmap: str = "jet",
):
    v = np.asarray(mesh.vertices)
    n = v.shape[0]
    e = np.asarray(undirected_edges, dtype=np.int64)
    imp = np.asarray(importance_undir, dtype=np.float32).reshape(-1)

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
    vimp = np.clip(vimp, 0.0, 1.0)

    try:
        import matplotlib.pyplot as plt
        cm = plt.get_cmap(cmap)
        colors = (cm(vimp)[:, :3] * 255.0).astype(np.uint8)
    except Exception:
        colors = np.stack([vimp, 1.0 - vimp, 1.0 - vimp], axis=1)
        colors = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)

    mesh_vis = mesh.copy()
    mesh_vis.visual.vertex_colors = colors

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mesh_vis.export(output_path)
    print(f"[Vis] Heatmap saved to {output_path}")


# ========= Geometric Helpers =========

def compute_face_normals_and_planes(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-18)
    normals = n / norm
    d = -np.sum(normals * v0, axis=1, keepdims=True)
    planes = np.concatenate([normals, d], axis=1)
    return normals, planes


def quadric_from_plane(plane: np.ndarray) -> np.ndarray:
    p = plane.reshape(4, 1)
    return p @ p.T


def solve_optimal_position(Q: np.ndarray, v_u: np.ndarray, v_v: np.ndarray) -> np.ndarray:
    A_reg = Q[:3, :3] + _EYE3
    b = -Q[:3, 3]
    try:
        x = np.linalg.solve(A_reg, b)
        if np.all(np.isfinite(x)): return x.astype(np.float64)
    except np.linalg.LinAlgError:
        pass
    return ((v_u + v_v) * 0.5).astype(np.float64)


def compute_qem_cost(Q: np.ndarray, v: np.ndarray) -> float:
    vh = np.array([v[0], v[1], v[2], 1.0], dtype=np.float64)
    return float(vh.T @ Q @ vh)


def tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def tri_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-20: return np.zeros(3, dtype=np.float64)
    return n / norm


def tri_quality_with_area(a: np.ndarray, b: np.ndarray, c: np.ndarray, area: float) -> float:
    ab = b - a;
    bc = c - b;
    ca = a - c
    s = float(np.dot(ab, ab) + np.dot(bc, bc) + np.dot(ca, ca))
    if s < 1e-30: return 0.0
    return float(_SQRT3_4 * area / s)


def export_obj_with_vertex_normals(mesh: trimesh.Trimesh, out_path: str) -> None:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if mesh.vertex_normals is None: mesh.fix_normals()
    vn = np.asarray(mesh.vertex_normals, dtype=np.float64)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("# OBJ exported by simplify_mesh.py\n")
        for p in v: fp.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        if vn.shape[0] == v.shape[0]:
            for n in vn: fp.write(f"vn {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n")
        fp.write("s 1\n")
        has_vn = (vn.shape[0] == v.shape[0])
        for tri in f:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            if has_vn:
                fp.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                fp.write(f"f {a} {b} {c}\n")


# ========= Local Validity Checks =========

def is_edge_collapse_valid_local(
        faces: np.ndarray, vertices: np.ndarray, face_alive: np.ndarray,
        vertex_faces: List[set], u: int, v: int, new_pos: np.ndarray,
        normal_cos_thresh: float, area_eps: float, min_tri_quality: float,
) -> bool:
    for fidx in vertex_faces[u]:
        if not face_alive[fidx]: continue
        a, b, c = int(faces[fidx, 0]), int(faces[fidx, 1]), int(faces[fidx, 2])
        if (a == u or b == u or c == u) and (a == v or b == v or c == v): continue

        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        new_a = new_pos if (a == u or a == v) else pa
        new_b = new_pos if (b == u or b == v) else pb
        new_c = new_pos if (c == u or c == v) else pc

        new_area = tri_area(new_a, new_b, new_c)
        if new_area < area_eps: return False
        if float(tri_quality_with_area(new_a, new_b, new_c, new_area)) < min_tri_quality: return False

        old_n = tri_normal(pa, pb, pc)
        new_n = tri_normal(new_a, new_b, new_c)

        if abs(old_n[0]) <= 1e-8 and abs(old_n[1]) <= 1e-8 and abs(old_n[2]) <= 1e-8: continue
        if abs(new_n[0]) <= 1e-8 and abs(new_n[1]) <= 1e-8 and abs(new_n[2]) <= 1e-8: continue
        if float(np.dot(old_n, new_n)) < normal_cos_thresh: return False

    for fidx in vertex_faces[v]:
        if fidx in vertex_faces[u]: continue
        if not face_alive[fidx]: continue
        a, b, c = int(faces[fidx, 0]), int(faces[fidx, 1]), int(faces[fidx, 2])
        if (a == u or b == u or c == u) and (a == v or b == v or c == v): continue

        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        new_a = new_pos if (a == u or a == v) else pa
        new_b = new_pos if (b == u or b == v) else pb
        new_c = new_pos if (c == u or c == v) else pc

        new_area = tri_area(new_a, new_b, new_c)
        if new_area < area_eps: return False
        if float(tri_quality_with_area(new_a, new_b, new_c, new_area)) < min_tri_quality: return False

        old_n = tri_normal(pa, pb, pc)
        new_n = tri_normal(new_a, new_b, new_c)

        if abs(old_n[0]) <= 1e-8 and abs(old_n[1]) <= 1e-8 and abs(old_n[2]) <= 1e-8: continue
        if abs(new_n[0]) <= 1e-8 and abs(new_n[1]) <= 1e-8 and abs(new_n[2]) <= 1e-8: continue
        if float(np.dot(old_n, new_n)) < normal_cos_thresh: return False

    return True


# ========= Main Simplification Function =========

@dataclass
class DebugStats:
    pop_total: int = 0
    ok_collapse: int = 0
    skip_dead_edge: int = 0
    skip_dead_vertex: int = 0
    skip_boundary: int = 0
    skip_nonmanifold: int = 0
    skip_link: int = 0
    skip_lock: int = 0
    skip_local: int = 0
    skip_vlimit: int = 0
    dup_edge: int = 0
    dup_face: int = 0


def edge_collapse_simplify(
        mesh: trimesh.Trimesh,
        importance_undir: np.ndarray,
        undirected_edges: np.ndarray,
        edge_feature_weight: Optional[np.ndarray],
        target_face_ratio: float = 0.5,

        global_start_faces: int = None,
        global_target_faces: int = None,
        strategy: str = "staged_local",
        hybrid_alpha: float = 0.5,

        gate_threshold: float = 0.20,
        relaxation_power: float = 2.0,

        min_tri_quality: float = 0.02,
        normal_cos_thresh: float = 0.5,
        imp_hard_thresh: float = 0.95,
        sharp_hard_thresh: float = 0.85,
        log_every: int = 200,

        max_valence: int = 0,
        max_valence_increase: int = 0,
        edge_len_penalty_w: float = 0.0,
        tri_quality_penalty: bool = False,
        tri_penalty_w: float = 0.35,
        tri_q_target: float = 0.05,
) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be trimesh.Trimesh")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    nV = int(vertices.shape[0])
    nF = int(faces.shape[0])
    if nV == 0 or nF == 0: return mesh.copy()

    undirected_edges = np.asarray(undirected_edges, dtype=np.int64)
    E = int(undirected_edges.shape[0])

    importance_undir = np.asarray(importance_undir, dtype=np.float32).reshape(-1)

    if edge_feature_weight is None:
        edge_feature_weight = np.zeros_like(importance_undir)
    else:
        edge_feature_weight = np.asarray(edge_feature_weight, dtype=np.float32).reshape(-1)

    face_alive = np.ones((nF,), dtype=bool)
    vert_alive = np.ones((nV,), dtype=bool)
    vertex_faces: List[set] = [set() for _ in range(nV)]
    for fi in range(nF):
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        if a == b or b == c or a == c: face_alive[fi] = False; continue
        vertex_faces[a].add(fi);
        vertex_faces[b].add(fi);
        vertex_faces[c].add(fi)

    start_faces = int(face_alive.sum())
    alive_faces = start_faces
    target_faces_abs = max(4, int(alive_faces * float(target_face_ratio)))

    edge_alive = np.ones((E,), dtype=bool)
    edge_rev = np.zeros((E,), dtype=np.int64)
    vertex_edges: List[set] = [set() for _ in range(nV)]
    edge_map = {}

    for eid in range(E):
        u, v = int(undirected_edges[eid, 0]), int(undirected_edges[eid, 1])
        if u > v: u, v = v, u; undirected_edges[eid, 0], undirected_edges[eid, 1] = u, v
        key = (u, v)
        if key in edge_map: edge_alive[eid] = False; continue
        edge_map[key] = eid
        vertex_edges[u].add(eid);
        vertex_edges[v].add(eid)

    Qv = np.zeros((nV, 4, 4), dtype=np.float64)

    def _plane_quadric(fi: int) -> np.ndarray:
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        n = np.cross(pb - pa, pc - pa)
        nn = float(np.linalg.norm(n))
        if nn < 1e-12: return np.zeros((4, 4), dtype=np.float64)
        n = n / nn
        d = -float(np.dot(n, pa))
        plane = np.array([n[0], n[1], n[2], d], dtype=np.float64)
        return quadric_from_plane(plane) * (0.5 * nn)

    for fi in range(nF):
        if face_alive[fi]:
            K = _plane_quadric(fi)
            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            Qv[a] += K;
            Qv[b] += K;
            Qv[c] += K

    def _one_ring(v: int) -> set:
        nbrs = set()
        for fi in vertex_faces[v]:
            if not face_alive[fi]: continue
            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            if a == v:
                nbrs.add(b);
                nbrs.add(c)
            elif b == v:
                nbrs.add(a);
                nbrs.add(c)
            elif c == v:
                nbrs.add(a);
                nbrs.add(b)
        return {x for x in nbrs if 0 <= x < nV and vert_alive[x]}

    def _common_faces(u: int, v: int) -> List[int]:
        return [fi for fi in vertex_faces[u].intersection(vertex_faces[v]) if face_alive[fi]]

    def _link_ok(u: int, v: int, common: List[int]) -> bool:
        # if len(common) != 2: return False
        # 修改后
        if len(common) not in (1, 2): return False
        Nu = _one_ring(u);
        Nv = _one_ring(v)
        Nu.discard(v);
        Nv.discard(u)
        opp = set()
        for fi in common:
            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            if a != u and a != v: opp.add(a)
            if b != u and b != v: opp.add(b)
            if c != u and c != v: opp.add(c)
        return Nu.intersection(Nv) == opp

    def _get_valence(v: int) -> int:
        return sum(1 for eid in vertex_edges[v] if edge_alive[eid])

    # Absolute Scale-Invariant Baseline calculated during init phase
    stage_median_c = 1e-12

    def _soft_penalty(u, v, pos, common):
        if (not tri_quality_penalty) or tri_penalty_w <= 0: return 0.0
        min_q = 1.0

        for fidx in vertex_faces[u]:
            if not face_alive[fidx] or fidx in common: continue
            a, b, c = int(faces[fidx, 0]), int(faces[fidx, 1]), int(faces[fidx, 2])
            pa = vertices[a] if (a != u and a != v) else pos
            pb = vertices[b] if (b != u and b != v) else pos
            pc = vertices[c] if (c != u and c != v) else pos

            area = tri_area(pa, pb, pc)
            if area < 1e-12: return 1000.0
            q = tri_quality_with_area(pa, pb, pc, area)
            if q < min_q: min_q = q

        for fidx in vertex_faces[v]:
            if fidx in vertex_faces[u]: continue
            if not face_alive[fidx] or fidx in common: continue
            a, b, c = int(faces[fidx, 0]), int(faces[fidx, 1]), int(faces[fidx, 2])
            pa = vertices[a] if (a != u and a != v) else pos
            pb = vertices[b] if (b != u and b != v) else pos
            pc = vertices[c] if (c != u and c != v) else pos

            area = tri_area(pa, pb, pc)
            if area < 1e-12: return 1000.0
            q = tri_quality_with_area(pa, pb, pc, area)
            if q < min_q: min_q = q

        if min_q >= tri_q_target: return 0.0
        return ((tri_q_target - min_q) / tri_q_target) * tri_penalty_w

    def get_current_gate() -> float:
        # ---------------------------------------------------------
        # 定义 1: 阶段局部重置 (Staged Local Reset) 的进度 t_local
        # 仅关注当前 stage 的存活面数比例，每次进入新 stage 从 0 重新起步
        # ---------------------------------------------------------
        rem_loc = max(alive_faces - target_faces_abs, 0)
        tot_loc = max(start_faces - target_faces_abs, 1)
        t_local = max(0.0, min(1.0, 1.0 - (rem_loc / tot_loc)))

        # ---------------------------------------------------------
        # 定义 2: 全局单调归一化 (Global Monotonic) 的进度 t_global
        # 贯穿整个简化生命周期，反映面数从最原始到最终目标的绝对衰减进度
        # ---------------------------------------------------------
        if global_start_faces is None or global_target_faces is None:
            t_global = t_local  # 容错降级
        else:
            rem_glob = max(alive_faces - global_target_faces, 0)
            tot_glob = max(global_start_faces - global_target_faces, 1)
            t_global = max(0.0, min(1.0, 1.0 - (rem_glob / tot_glob)))

        # ---------------------------------------------------------
        # 根据选定策略，计算门限 (Gate)
        # ---------------------------------------------------------
        if strategy == "staged_local":
            # 策略 A: 仅使用局部阶段进度进行指数松弛
            curve = t_local ** relaxation_power
            gate = gate_threshold + (1.0 - gate_threshold) * curve

        elif strategy == "global_monotonic":
            # 策略 B: 严格服从全局单调进度进行松弛
            curve = t_global ** relaxation_power
            gate = gate_threshold + (1.0 - gate_threshold) * curve

        elif strategy in ["hybrid", "adaptive_hybrid"]:
            # 策略 C: 混合模式。分别计算出全局期望门限和局部期望门限
            g_global = gate_threshold + (1.0 - gate_threshold) * (t_global ** relaxation_power)
            g_local = gate_threshold + (1.0 - gate_threshold) * (t_local ** relaxation_power)

            # 权重 alpha 可以是静态参数，或者是与全局进度绑定的自适应项
            alpha = t_global ** relaxation_power if strategy == "adaptive_hybrid" else hybrid_alpha
            gate = alpha * g_global + (1.0 - alpha) * g_local

        else:
            # 默认 fallback 到 local
            curve = t_local ** relaxation_power
            gate = gate_threshold + (1.0 - gate_threshold) * curve

        return float(max(gate_threshold, min(1.0, gate)))

    _init_buffer = []

    def push_edge(eid: int, is_init: bool = False) -> None:
        if not edge_alive[eid]: return
        u, v = int(undirected_edges[eid, 0]), int(undirected_edges[eid, 1])
        if not vert_alive[u] or not vert_alive[v]: return

        common = _common_faces(u, v)
        k = len(common)
        # if k != 2: return
        # 修改后
        if k not in (1, 2): return
        if not _link_ok(u, v, common): return

        if max_valence > 0:
            nv = _get_valence(u) + _get_valence(v) - 4
            if nv > max_valence: return
            if max_valence_increase > 0:
                if (nv - _get_valence(u) > max_valence_increase) or (
                        nv - _get_valence(v) > max_valence_increase): return


        imp = float(importance_undir[eid])
        sharp = float(edge_feature_weight[eid])

        # [删除硬截断]：绝不直接 return，让所有边都有资格进堆！
        # if imp > imp_hard_thresh: return
        # if sharp > sharp_hard_thresh: return

        pu, pv = vertices[u], vertices[v]
        dx, dy, dz = pu[0] - pv[0], pu[1] - pv[1], pu[2] - pv[2]
        elen = math.sqrt(dx * dx + dy * dy + dz * dz)

        Q = Qv[u] + Qv[v]
        pos = solve_optimal_position(Q, vertices[u], vertices[v])
        c = compute_qem_cost(Q, pos)

        c += _soft_penalty(u, v, pos, common)
        if edge_len_penalty_w > 0: c += edge_len_penalty_w * (elen ** 2)

        if is_init:
            # 必须把 sharp 加进元组，否则后面无法判断软保险丝！
            _init_buffer.append((eid, u, v, elen, imp, sharp, c, pos))
            # _init_buffer.append((eid, u, v, elen, imp, c, pos))
            return

        # Energy Formulation: E_total = E_geom(norm) + λ * E_semantic(barrier)
        c_norm = c / max(stage_median_c, 1e-12)
        curr_gate = get_current_gate()

        if imp > curr_gate:
            violation = (imp - curr_gate) / (1.0 - curr_gate + 1e-6)
            barrier = BARRIER_LAMBDA * (violation ** 2)
            c_norm += barrier

        # [新增：动态更新时的 Soft Clamp]
        if imp > imp_hard_thresh or sharp > sharp_hard_thresh:
            c_norm += 10.0 * BARRIER_LAMBDA

        edge_rev[eid] += 1
        # Eid acts as the strict and only tie-breaker, completely avoiding edge-length bias.
        heapq.heappush(heap, (float(c_norm), eid, int(edge_rev[eid]), u, v, pos[0], pos[1], pos[2]))

    heap: List[Tuple[float, int, int, int, int, float, float, float]] = []
    stats = DebugStats()

    # === Two-Pass Unbiased Initialization ===
    for eid in range(E):
        if edge_alive[eid]: push_edge(eid, is_init=True)

    if len(_init_buffer) > 0:
        costs = [item[5] for item in _init_buffer]
        stage_median_c = float(np.median(costs))
        if stage_median_c < 1e-12:
            stage_median_c = float(np.mean(costs))
            if stage_median_c < 1e-12:
                stage_median_c = 1e-12
    else:
        stage_median_c = 1e-12

    init_gate = get_current_gate()
    for item in _init_buffer:
        # 注意这里多解包了一个 sharp
        eid, u, v, elen, imp, sharp, c, pos = item
        # eid, u, v, elen, imp, c, pos = item
        c_norm = c / max(stage_median_c, 1e-12)
        if imp > init_gate:
            violation = (imp - init_gate) / (1.0 - init_gate + 1e-6)
            barrier = BARRIER_LAMBDA * (violation ** 2)
            c_norm += barrier

        # [新增：初始化阶段的 Soft Clamp]
        if imp > imp_hard_thresh or sharp > sharp_hard_thresh:
            c_norm += 10.0 * BARRIER_LAMBDA

        edge_rev[eid] += 1
        heapq.heappush(heap, (float(c_norm), eid, int(edge_rev[eid]), u, v, pos[0], pos[1], pos[2]))

    _init_buffer.clear()

    # ---- Collapse Loop ----
    while alive_faces > target_faces_abs and heap:
        s, eid, rev, u, v, px, py, pz = heapq.heappop(heap)
        stats.pop_total += 1

        if not edge_alive[eid] or rev != edge_rev[eid]: stats.skip_dead_edge += 1; continue
        if not vert_alive[u] or not vert_alive[v]: stats.skip_dead_vertex += 1; continue

        common = _common_faces(u, v)
        # if len(common) != 2: stats.skip_nonmanifold += 1; continue
        # 修改后
        if len(common) not in (1, 2): stats.skip_nonmanifold += 1; continue
        if not _link_ok(u, v, common): stats.skip_link += 1; continue

        if max_valence > 0:
            if (_get_valence(u) + _get_valence(v) - 4) > max_valence:
                stats.skip_vlimit += 1;
                continue

        new_pos = np.array([px, py, pz], dtype=np.float64)
        if not is_edge_collapse_valid_local(faces, vertices, face_alive, vertex_faces, u, v, new_pos, normal_cos_thresh,
                                            1e-12, min_tri_quality):
            stats.skip_local += 1;
            continue

        # # --- [V5 终极拓扑修补：严格等价 QEM 的边邻接守恒检测] ---
        # is_topo_safe = True
        # new_face_sigs = set()
        # future_edge_counts = {}
        #
        # # 1. 获取受本次折叠影响的局部面片集合
        # affected_fidx = set(vertex_faces[u]) | set(vertex_faces[v])
        #
        # for fi in affected_fidx:
        #     if not face_alive[fi] or fi in common:
        #         continue
        #
        #     a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        #
        #     # 模拟顶点坍缩 v -> u
        #     na, nb, nc = a, b, c
        #     if na == v: na = u
        #     if nb == v: nb = u
        #     if nc == v: nc = u
        #
        #     # 退化面判断 (实际上 common 面已被拦截，这里是双重保险)
        #     if na == nb or nb == nc or nc == na:
        #         continue
        #
        #     # 检测重复面 (防折叠撕裂)
        #     if na > nb: na, nb = nb, na
        #     if nb > nc: nb, nc = nc, nb
        #     if na > nb: na, nb = nb, na
        #     f_sig = (na, nb, nc)
        #
        #     if f_sig in new_face_sigs:
        #         is_topo_safe = False
        #         stats.dup_face += 1
        #         break
        #     new_face_sigs.add(f_sig)
        #
        #     # 统计模拟折叠后，各边在局部区域产生的新面数
        #     e1 = (na, nb) if na < nb else (nb, na)
        #     e2 = (nb, nc) if nb < nc else (nc, nb)
        #     e3 = (na, nc) if na < nc else (nc, na)
        #
        #     future_edge_counts[e1] = future_edge_counts.get(e1, 0) + 1
        #     future_edge_counts[e2] = future_edge_counts.get(e2, 0) + 1
        #     future_edge_counts[e3] = future_edge_counts.get(e3, 0) + 1
        #
        # if is_topo_safe:
        #     # 2. 校验全局边邻接守恒 (全局精准映射)
        #     for e, f_count in future_edge_counts.items():
        #         # 该边在折叠前的全局真实面数
        #         orig_global_faces = [f for f in vertex_faces[e[0]].intersection(vertex_faces[e[1]]) if
        #                              face_alive[f]]
        #         orig_global_count = len(orig_global_faces)
        #
        #         # 该边在被影响区域 (affected_fidx) 中的原面数
        #         orig_in_affected = sum(1 for f in orig_global_faces if f in affected_fidx)
        #
        #         # 数学核心：折叠后的全局面数 = 原全局数 - 原局部数 + 新模拟局部数
        #         global_future_count = orig_global_count - orig_in_affected + f_count
        #
        #         # 💡 核心修复：只拦截 >2（非流形边）。允许 == 1（合法的边界收缩）。
        #         if global_future_count > 2:
        #             is_topo_safe = False
        #             stats.skip_nonmanifold += 1
        #             break
        #
        #         # if global_future_count > 2:
        #         #     # 拦截：产生非流形边 (Non-manifold edge)
        #         #     is_topo_safe = False
        #         #     stats.skip_nonmanifold += 1
        #         #     break
        #         # elif global_future_count == 1 and orig_global_count > 1:
        #         #     # 拦截：内部边被撕裂成了边界边 (Torn edge)
        #         #     is_topo_safe = False
        #         #     stats.skip_boundary += 1
        #         #     break
        #
        # if not is_topo_safe:
        #     continue
        # # ----------------------------------------

        # === Do Collapse ===
        keep, dead = u, v

        Qv[keep] += Qv[dead]
        vertices[keep] = new_pos

        for fi in common:
            face_alive[fi] = False
            alive_faces -= 1
            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            vertex_faces[a].discard(fi);
            vertex_faces[b].discard(fi);
            vertex_faces[c].discard(fi)

        updated_faces = set()
        for fi in list(vertex_faces[dead]):
            if not face_alive[fi]: continue

            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            vertex_faces[a].discard(fi);
            vertex_faces[b].discard(fi);
            vertex_faces[c].discard(fi)

            if a == dead: a = keep
            if b == dead: b = keep
            if c == dead: c = keep

            # 必须删掉 tri_area < 1e-12！这里有 alive_faces -= 1 是正确的！
            if a == b or b == c or c == a:
            # if a == b or b == c or c == a or tri_area(vertices[a], vertices[b], vertices[c]) < 1e-12:
                face_alive[fi] = False
                alive_faces -= 1
                continue

            faces[fi, 0], faces[fi, 1], faces[fi, 2] = a, b, c
            vertex_faces[a].add(fi);
            vertex_faces[b].add(fi);
            vertex_faces[c].add(fi)
            updated_faces.add(fi)

        vertex_faces[dead].clear()
        vert_alive[dead] = False

        dead_edges = list(vertex_edges[dead])
        for eid2 in dead_edges:
            if not edge_alive[eid2]: continue
            oa = int(undirected_edges[eid2, 0]);
            ob = int(undirected_edges[eid2, 1])
            key = (oa, ob) if oa < ob else (ob, oa)
            if edge_map.get(key) == eid2: del edge_map[key]

        for eid2 in dead_edges:
            if not edge_alive[eid2]: continue
            oa = int(undirected_edges[eid2, 0]);
            ob = int(undirected_edges[eid2, 1])
            vertex_edges[oa].discard(eid2);
            vertex_edges[ob].discard(eid2)

            if eid2 == eid: edge_alive[eid2] = False; continue

            other = ob if oa == dead else (oa if ob == dead else None)
            if other is None or not vert_alive[other] or other == keep:
                edge_alive[eid2] = False
                continue

            a2, b2 = keep, int(other)
            if a2 > b2: a2, b2 = b2, a2
            key = (a2, b2)
            if key in edge_map and edge_alive[edge_map[key]]:
                edge_alive[eid2] = False
                stats.dup_edge += 1
                continue

            undirected_edges[eid2, 0] = a2
            undirected_edges[eid2, 1] = b2
            edge_map[key] = eid2
            edge_alive[eid2] = True
            vertex_edges[a2].add(eid2);
            vertex_edges[b2].add(eid2)

        vertex_edges[dead].clear()

        refresh_v = _one_ring(keep)
        refresh_v.add(keep)
        affected = set(vertex_faces[keep]) | updated_faces
        for nb in refresh_v: affected |= vertex_faces[nb]

        seen = {}
        for fi in list(affected):
            if not face_alive[fi]: continue

            a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            if a > b: a, b = b, a
            if b > c: b, c = c, b
            if a > b: a, b = b, a
            key = (a, b, c)

            if key in seen:
                face_alive[fi] = False
                alive_faces -= 1
                vertex_faces[a].discard(fi);
                vertex_faces[b].discard(fi);
                vertex_faces[c].discard(fi)
                stats.dup_face += 1
            else:
                seen[key] = fi

        for vv in refresh_v:
            if not vert_alive[vv]: continue
            for e in vertex_edges[vv]:
                if edge_alive[e]: push_edge(e, is_init=False)

        stats.ok_collapse += 1
        if log_every > 0 and (stats.ok_collapse % int(log_every) == 0):
            curr_g = get_current_gate()
            print(f"[C={stats.ok_collapse}] Faces={alive_faces}/{nF} Gate={curr_g:.3f} Heap={len(heap)}")

    new_idx = -np.ones((nV,), dtype=np.int64)
    alive = np.where(vert_alive)[0]
    new_idx[alive] = np.arange(len(alive), dtype=np.int64)
    out_v = vertices[alive]
    out_f = []
    for fi in range(nF):
        if not face_alive[fi]: continue
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        if not vert_alive[a] or not vert_alive[b] or not vert_alive[c]: continue
        out_f.append([new_idx[a], new_idx[b], new_idx[c]])

    if not out_f: return trimesh.Trimesh(vertices=out_v, faces=[], process=False)
    out = trimesh.Trimesh(vertices=out_v, faces=np.array(out_f, dtype=np.int64), process=False)
    out.remove_unreferenced_vertices()
    return out


def simplify_with_model(
        obj_path: str,
        ckpt_path: str,
        device: str = "cpu",
        target_face_ratio: float = 0.5,
        out_path: Optional[str] = None,

        strategy: str = "staged_local",
        hybrid_alpha: float = 0.5,

        gate_c0: float = 0.20,
        relaxation_power: float = 2.0,

        stages: int = 1,
        bidirectional_graph: bool = True,
        refresh_gate_each_stage: bool = True,

        min_tri_quality: float = 0.02,
        normal_cos_thresh: float = 0.5,
        imp_hard_thresh: float = 0.95,
        sharp_hard_thresh: float = 0.98,

        tri_quality_penalty: bool = False,
        tri_penalty_w: float = 0.35,
        tri_q_target: float = 0.08,
        max_valence: int = 0,
        max_valence_increase: int = 0,
        edge_len_penalty_w: float = 0.0,

        log_every: int = 200,
        export_importance_map: bool = False,
        importance_map_path: Optional[str] = None,
        importance_map_cmap: str = "jet",
        write_vertex_normals: bool = True,
):
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)

    model = None

    def _predict_importance(cur_mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        nonlocal model
        data: Data = mesh_to_graph(cur_mesh, bidirectional=bidirectional_graph)
        data = data.to(device)

        if model is None:
            in_channels = data.x.size(1)
            model = EdgeImportanceGNN(in_channels=in_channels, hidden_channels=64, num_layers=3, edge_feat_dim=2).to(
                device)
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.eval()

        with torch.no_grad():
            out = model(data, return_vertex_embeddings=False, aggregate_undirected=True)
            imp_undir = out["edge_importance_undir"].detach().cpu().numpy().reshape(-1)

        sharp = None
        if hasattr(data, "edge_feature_weight") and data.edge_feature_weight is not None:
            sharp = data.edge_feature_weight.detach().cpu().numpy().reshape(-1)

        undir_edges = data.undirected_edges.detach().cpu().numpy()
        return imp_undir, undir_edges, sharp

    stages = max(1, int(stages))
    orig_faces = int(len(mesh.faces))
    orig_verts = int(len(mesh.vertices))
    cur_mesh = mesh

    total_gnn_time = 0.0
    total_cpu_time = 0.0
    peak_gpu_mem_mb = 0.0

    if torch.cuda.is_available() and "cuda" in str(device).lower():
        torch.cuda.reset_peak_memory_stats(device)

    if stages == 1:
        stage_ratios = [target_face_ratio]
    else:
        step = target_face_ratio ** (1.0 / stages)
        stage_ratios = [step ** (i + 1) for i in range(stages)]
        stage_ratios[-1] = target_face_ratio

    global_start_faces = orig_faces
    global_target_faces = max(4, int(orig_faces * stage_ratios[-1]))

    for si, r in enumerate(stage_ratios, start=1):
        target_faces_abs = max(4, int(orig_faces * r))
        cur_faces = int(len(cur_mesh.faces))
        stage_ratio = target_faces_abs / max(cur_faces, 1)

        print(f"\n[Stage {si}/{len(stage_ratios)}] Faces: {cur_faces} -> {target_faces_abs} (r={stage_ratio:.4f})")

        t0 = time.time()
        imp_undir, undir_edges, sharp = _predict_importance(cur_mesh)

        if torch.cuda.is_available() and "cuda" in str(device).lower():
            mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            peak_gpu_mem_mb = max(peak_gpu_mem_mb, mem)

        total_gnn_time += (time.time() - t0)

        if export_importance_map:
            if importance_map_path is None:
                stem0 = os.path.splitext(os.path.basename(out_path or obj_path))[0]
                importance_map_path0 = os.path.join(os.path.dirname(out_path or obj_path), f"{stem0}_imp_stage{si}.ply")
            else:
                base0, ext0 = os.path.splitext(importance_map_path)
                ext0 = ext0 or ".ply"
                importance_map_path0 = f"{base0}_stage{si}{ext0}"
            save_importance_map(cur_mesh, undir_edges, imp_undir, importance_map_path0, cmap=importance_map_cmap)

        t1 = time.time()
        cur_mesh = edge_collapse_simplify(
            mesh=cur_mesh,
            importance_undir=imp_undir,
            undirected_edges=undir_edges,
            edge_feature_weight=sharp,
            target_face_ratio=float(stage_ratio),

            global_start_faces=global_start_faces,
            global_target_faces=global_target_faces,
            strategy=strategy,
            hybrid_alpha=hybrid_alpha,

            gate_threshold=gate_c0,
            relaxation_power=relaxation_power,
            imp_hard_thresh=imp_hard_thresh,
            sharp_hard_thresh=sharp_hard_thresh,
            normal_cos_thresh=normal_cos_thresh,
            min_tri_quality=min_tri_quality,
            tri_quality_penalty=tri_quality_penalty,
            tri_penalty_w=tri_penalty_w,
            tri_q_target=tri_q_target,
            max_valence=max_valence,
            max_valence_increase=max_valence_increase,
            edge_len_penalty_w=edge_len_penalty_w,
            log_every=log_every,
        )
        total_cpu_time += (time.time() - t1)

    simplified = cur_mesh

    if out_path is None:
        stem = os.path.splitext(os.path.basename(obj_path))[0]
        out_path = os.path.join(os.path.dirname(obj_path), f"{stem}_simplified.obj")

    if write_vertex_normals:
        export_obj_with_vertex_normals(simplified, out_path)
    else:
        simplified.export(out_path)

    print(f"[Done] saved: {out_path}")

    stats_dict = {
        "verts_orig": orig_verts,
        "faces_orig": orig_faces,
        "verts_simp": int(len(simplified.vertices)),
        "faces_simp": int(len(simplified.faces)),
        "gpu_inference_time": round(total_gnn_time, 4),
        "cpu_time_seconds": round(total_cpu_time, 4),
        "total_time_seconds": round(total_gnn_time + total_cpu_time, 4),
        "gpu_vram_mb": round(peak_gpu_mem_mb, 2)
    }

    return simplified, stats_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="checkpoints/edge_gnn_tosc.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=None)

    parser.add_argument("--strategy", type=str, default="staged_local",
                        choices=["staged_local", "global_monotonic", "hybrid", "adaptive_hybrid"])
    parser.add_argument("--hybrid_alpha", type=float, default=0.5)

    parser.add_argument("--gate_c0", type=float, default=0.20)
    parser.add_argument("--relaxation_power", type=float, default=2.0)

    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--refresh_gate_each_stage", type=int, default=1)

    parser.add_argument("--min_tri_q", type=float, default=0.02)
    parser.add_argument("--normal_cos", type=float, default=0.5)
    parser.add_argument("--imp_hard", type=float, default=0.95)
    parser.add_argument("--sharp_hard", type=float, default=0.98)

    parser.add_argument("--tri_quality_penalty", type=int, default=0)
    parser.add_argument("--tri_penalty_w", type=float, default=0.35)
    parser.add_argument("--tri_q_target", type=float, default=0.08)
    parser.add_argument("--max_valence", type=int, default=0)
    parser.add_argument("--max_valence_increase", type=int, default=0)
    parser.add_argument("--edge_len_penalty_w", type=float, default=0.0)

    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--no_vn", action="store_true")

    args = parser.parse_args()

    simplify_with_model(
        obj_path=args.obj,
        ckpt_path=args.ckpt,
        device=args.device,
        target_face_ratio=args.ratio,
        out_path=args.out,

        strategy=args.strategy,
        hybrid_alpha=args.hybrid_alpha,

        gate_c0=args.gate_c0,
        relaxation_power=args.relaxation_power,
        stages=args.stages,
        refresh_gate_each_stage=bool(args.refresh_gate_each_stage),
        min_tri_quality=args.min_tri_q,
        normal_cos_thresh=args.normal_cos,
        imp_hard_thresh=args.imp_hard,
        sharp_hard_thresh=args.sharp_hard,
        tri_quality_penalty=bool(args.tri_quality_penalty),
        tri_penalty_w=args.tri_penalty_w,
        tri_q_target=args.tri_q_target,
        max_valence=args.max_valence,
        max_valence_increase=args.max_valence_increase,
        edge_len_penalty_w=args.edge_len_penalty_w,
        log_every=args.log_every,
        write_vertex_normals=(not args.no_vn),
    )
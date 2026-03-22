# simplify_mesh.py
# -*- coding: utf-8 -*-
"""
GNN-guided QEM simplification (Ultimate Research Version - Continuous Barrier).

Methodology:
1. QEM Cost is SACRED: The base geometric error is calculated purely and normalized independently.
2. Continuous Barrier Gating:
   - We introduce a SOFT, gate-aware reweighting barrier rather than a hard cutoff.
   - If importance > gate, a penalty is applied scaling with the violation magnitude.
   - Formula: Cost *= (1 + lambda * ((imp - gate)/(1-gate))^2)
   - As the gate relaxes to 1.0 (continuously based on collapse progress), the barrier naturally vanishes.
3. Dual-Timeline Dynamics:
   - Gate Relaxation: Continuous (per-collapse).
   - Importance Refresh: Staged (per-stage) to balance efficiency and accuracy.

Robustness:
- Quadric Accumulation, Topology Guards, Valence Guards.
"""

from __future__ import annotations

import os
import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

from mesh_dataset import mesh_to_graph
from models import EdgeImportanceGNN

# Penalty Strength for the Barrier (Soft Constraint Strength)
# A value of 100.0 means a fully violated edge (imp=1.0 when gate=0.0) gets 100x cost penalty.
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
    """Visualization: Export importance map as vertex colors."""
    v = np.asarray(mesh.vertices)
    n = v.shape[0]
    e = np.asarray(undirected_edges, dtype=np.int64)
    imp = np.asarray(importance_undir, dtype=np.float32).reshape(-1)

    vsum = np.zeros((n,), dtype=np.float32)
    vcnt = np.zeros((n,), dtype=np.float32)
    if e.shape[0] > 0 and imp.shape[0] == e.shape[0]:
        a = e[:, 0];
        b = e[:, 1]
        np.add.at(vsum, a, imp);
        np.add.at(vsum, b, imp)
        np.add.at(vcnt, a, 1.0);
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
    v0 = vertices[faces[:, 0]];
    v1 = vertices[faces[:, 1]];
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
    A = Q[:3, :3];
    b = -Q[:3, 3]
    try:
        A_reg = A + np.eye(3) * 1e-12
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


def tri_quality(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a;
    bc = c - b;
    ca = a - c
    s = float(np.dot(ab, ab) + np.dot(bc, bc) + np.dot(ca, ca))
    if s < 1e-30: return 0.0
    area = tri_area(a, b, c)
    return float(4.0 * np.sqrt(3.0) * area / s)


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
    affected_faces = (vertex_faces[u] | vertex_faces[v]).copy()
    for fidx in affected_faces:
        if not face_alive[fidx]: continue
        tri = faces[fidx]
        if u in tri and v in tri: continue

        a, b, c = tri.tolist()
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        new_a = new_pos if a in (u, v) else pa
        new_b = new_pos if b in (u, v) else pb
        new_c = new_pos if c in (u, v) else pc

        if tri_area(new_a, new_b, new_c) < area_eps: return False
        if float(tri_quality(new_a, new_b, new_c)) < min_tri_quality: return False

        old_n = tri_normal(pa, pb, pc)
        new_n = tri_normal(new_a, new_b, new_c)
        if np.allclose(old_n, 0) or np.allclose(new_n, 0): continue
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
        edge_feature_weight: np.ndarray,
        target_face_ratio: float = 0.5,

        # ✅ Methodology Pure: Only Gate Parameters
        gate_threshold: float = 0.20,
        relaxation_power: float = 2.0,  # ✅ Fixed: Now receiving this param

        min_tri_quality: float = 0.02,
        normal_cos_thresh: float = 0.5,
        imp_hard_thresh: float = 0.95,
        sharp_hard_thresh: float = 0.85,
        log_every: int = 200,

        # Robustness
        max_valence: int = 0,
        max_valence_increase: int = 0,
        edge_len_penalty_w: float = 0.0,
        tri_quality_penalty: bool = False,
        tri_penalty_w: float = 0.35,
        tri_q_target: float = 0.05,
) -> trimesh.Trimesh:
    """
    QEM edge-collapse simplification with GNN-guided Continuous Barrier.
    """
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
    edge_feature_weight = np.asarray(edge_feature_weight, dtype=np.float32).reshape(-1)

    # ---- Initialization ----
    face_alive = np.ones((nF,), dtype=bool)
    vert_alive = np.ones((nV,), dtype=bool)
    vertex_faces: List[set] = [set() for _ in range(nV)]
    for fi in range(nF):
        a, b, c = map(int, faces[fi])
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
        if u > v: u, v = v, u; undirected_edges[eid] = [u, v]
        key = (u, v)
        if key in edge_map: edge_alive[eid] = False; continue
        edge_map[key] = eid
        vertex_edges[u].add(eid);
        vertex_edges[v].add(eid)

    # ---- QEM Quadrics ----
    Qv = np.zeros((nV, 4, 4), dtype=np.float64)

    def _plane_quadric(fi: int) -> np.ndarray:
        a, b, c = map(int, faces[fi])
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        n = np.cross(pb - pa, pc - pa)
        nn = float(np.linalg.norm(n))
        if nn < 1e-12: return np.zeros((4, 4), dtype=np.float64)
        n = n / nn;
        d = -float(np.dot(n, pa))
        plane = np.array([n[0], n[1], n[2], d], dtype=np.float64)
        return quadric_from_plane(plane) * (0.5 * nn)

    for fi in range(nF):
        if face_alive[fi]:
            K = _plane_quadric(fi)
            a, b, c = map(int, faces[fi])
            Qv[a] += K;
            Qv[b] += K;
            Qv[c] += K

    # ---- Helpers ----
    def _one_ring(v: int) -> set:
        nbrs = set()
        for fi in vertex_faces[v]:
            if not face_alive[fi]: continue
            a, b, c = map(int, faces[fi])
            if a == v:
                nbrs.add(b); nbrs.add(c)
            elif b == v:
                nbrs.add(a); nbrs.add(c)
            elif c == v:
                nbrs.add(a); nbrs.add(b)
        return {x for x in nbrs if 0 <= x < nV and vert_alive[x]}

    def _common_faces(u: int, v: int) -> List[int]:
        return [fi for fi in vertex_faces[u].intersection(vertex_faces[v]) if face_alive[fi]]

    def _link_ok(u: int, v: int, common: List[int]) -> bool:
        if len(common) != 2: return False
        Nu = _one_ring(u);
        Nv = _one_ring(v);
        Nu.discard(v);
        Nv.discard(u)
        opp = set()
        for fi in common:
            for w in faces[fi]:
                if w != u and w != v: opp.add(w)
        return Nu.intersection(Nv) == opp

    def _get_valence(v: int) -> int:
        return sum(1 for eid in vertex_edges[v] if edge_alive[eid])

    # ---- Core Logic: Pure QEM Cost ----
    mean_c = 1e-12
    mean_cnt = 0

    def _soft_penalty(u, v, pos, common):
        if (not tri_quality_penalty) or tri_penalty_w <= 0: return 0.0
        min_q = 1.0
        affected = vertex_faces[u] | vertex_faces[v]
        for fi in affected:
            if not face_alive[fi] or fi in common: continue
            a, b, c = map(int, faces[fi])
            pa = vertices[a] if a not in (u, v) else pos
            pb = vertices[b] if b not in (u, v) else pos
            pc = vertices[c] if c not in (u, v) else pos
            if tri_area(pa, pb, pc) < 1e-12: return 1000.0  # avoid degeneracy
            q = tri_quality(pa, pb, pc)
            if q < min_q: min_q = q
        if min_q >= tri_q_target: return 0.0
        return ((tri_q_target - min_q) / tri_q_target) * tri_penalty_w

    def edge_score(u: int, v: int, common: List[int], elen: float, current_gate: float, imp: float) -> Tuple[
        float, np.ndarray]:
        nonlocal mean_c, mean_cnt

        # 1. Pure QEM Cost
        Q = Qv[u] + Qv[v]
        pos = solve_optimal_position(Q, vertices[u], vertices[v])
        c = compute_qem_cost(Q, pos)

        # 2. Geometric Penalties
        c += _soft_penalty(u, v, pos, common)
        if edge_len_penalty_w > 0: c += edge_len_penalty_w * (elen ** 2)

        # 3. Normalization (Purely numerical, ordering preserved)
        mean_cnt += 1
        mean_c = mean_c + (c - mean_c) / float(mean_cnt)
        c_norm = c / max(mean_c, 1e-12)

        # ✅ Logic B: Continuous Barrier Penalty
        # If importance > gate, apply a penalty scaled by the violation magnitude.
        # This keeps the edge in the heap but delays its collapse until gate relaxes.
        if imp > current_gate:
            # normalized violation: how far is imp from gate, relative to the remaining space
            violation = (imp - current_gate) / (1.0 - current_gate + 1e-6)
            # Apply multiplicative barrier penalty
            barrier = BARRIER_LAMBDA * (violation ** 2)
            c_norm *= (1.0 + barrier)

        return float(c_norm), pos

    heap: List[Tuple[float, float, int, int, int, int, float, float, float]] = []
    stats = DebugStats()

    # ✅ Helper to calculate dynamic gate based on collapse progress
    def get_current_gate() -> float:
        if start_faces <= target_faces_abs: return 1.0
        # Progress: 0.0 (Start) -> 1.0 (Target Reached)
        prog = 1.0 - (alive_faces - target_faces_abs) / (start_faces - target_faces_abs)
        prog = max(0.0, min(1.0, prog))
        # Quadratic Relaxation
        curve = prog ** relaxation_power
        # gate starts at gate_threshold (from CLI arg), ends at 1.0
        return gate_threshold + (1.0 - gate_threshold) * curve

    def push_edge(eid: int) -> None:
        if not edge_alive[eid]: return
        u, v = int(undirected_edges[eid, 0]), int(undirected_edges[eid, 1])
        if not vert_alive[u] or not vert_alive[v]: return

        common = _common_faces(u, v)
        k = len(common)
        if k != 2: return  # Strict Manifold
        if not _link_ok(u, v, common): return  # Topology

        if max_valence > 0:
            nv = _get_valence(u) + _get_valence(v) - 4
            if nv > max_valence: return
            if max_valence_increase > 0:
                if (nv - _get_valence(u) > max_valence_increase) or (
                        nv - _get_valence(v) > max_valence_increase): return

        imp = float(importance_undir[eid])
        sharp = float(edge_feature_weight[eid])

        # ✅ Hard Protection (Permanent Lock - Safety Guard)
        if imp > imp_hard_thresh: return
        if sharp > sharp_hard_thresh: return

        # ✅ Continuous Barrier Calculation
        curr_gate = get_current_gate()

        dx, dy, dz = vertices[u] - vertices[v]
        elen = (dx * dx + dy * dy + dz * dz) ** 0.5

        # Pass imp and curr_gate to score
        s, pos = edge_score(u, v, common, float(elen), curr_gate, imp)

        edge_rev[eid] += 1
        heapq.heappush(heap, (s, elen, eid, int(edge_rev[eid]), u, v, pos[0], pos[1], pos[2]))

    # Init Heap
    for eid in range(E):
        if edge_alive[eid]: push_edge(eid)

    # ---- Collapse Loop ----
    while alive_faces > target_faces_abs and heap:
        s, elen, eid, rev, u, v, px, py, pz = heapq.heappop(heap)
        stats.pop_total += 1

        if not edge_alive[eid] or rev != edge_rev[eid]: stats.skip_dead_edge += 1; continue
        if not vert_alive[u] or not vert_alive[v]: stats.skip_dead_vertex += 1; continue

        common = _common_faces(u, v)
        if len(common) != 2: stats.skip_nonmanifold += 1; continue
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

        # === Do Collapse ===
        keep, dead = u, v

        # ✅ Critical: Quadric Accumulation
        Qv[keep] += Qv[dead]

        vertices[keep] = new_pos

        # Update Topology
        for fi in common:
            face_alive[fi] = False;
            alive_faces -= 1
            for vv in map(int, faces[fi]): vertex_faces[vv].discard(fi)

        updated_faces = set()
        for fi in list(vertex_faces[dead]):
            if not face_alive[fi]: continue
            old_tri = faces[fi].copy()
            for vv in map(int, old_tri): vertex_faces[vv].discard(fi)
            tri = old_tri.copy();
            tri[tri == dead] = keep
            a, b, c = map(int, tri)

            if a == b or b == c or c == a or tri_area(vertices[a], vertices[b], vertices[c]) < 1e-12:
                face_alive[fi] = False;
                alive_faces -= 1;
                continue

            faces[fi] = tri
            for vv in (a, b, c): vertex_faces[vv].add(fi)
            updated_faces.add(fi)

        vertex_faces[dead].clear();
        vert_alive[dead] = False

        # Redirect Edges
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
                edge_alive[eid2] = False;
                continue

            a2, b2 = keep, int(other);
            if a2 > b2: a2, b2 = b2, a2
            key = (a2, b2)
            if key in edge_map and edge_alive[edge_map[key]]:
                edge_alive[eid2] = False;
                stats.dup_edge += 1;
                continue

            undirected_edges[eid2] = [a2, b2]
            edge_map[key] = eid2;
            edge_alive[eid2] = True
            vertex_edges[a2].add(eid2);
            vertex_edges[b2].add(eid2)

        vertex_edges[dead].clear()

        # Clean 1-ring (remove duplicate faces created)
        refresh_v = _one_ring(keep);
        refresh_v.add(keep)
        affected = set(vertex_faces[keep]) | updated_faces
        for nb in refresh_v: affected |= vertex_faces[nb]

        seen = {}
        for fi in list(affected):
            if not face_alive[fi]: continue
            key = tuple(sorted(faces[fi]))
            if key in seen:
                face_alive[fi] = False;
                alive_faces -= 1
                for vv in key: vertex_faces[vv].discard(fi)
                stats.dup_face += 1
            else:
                seen[key] = fi

        # Repush edges (Updates scores with new Gate!)
        for vv in refresh_v:
            if not vert_alive[vv]: continue
            for e in vertex_edges[vv]:
                if edge_alive[e]: push_edge(e)

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
        a, b, c = map(int, faces[fi])
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

        # Core B-Scheme Params
        gate_c0: float = 0.20,
        relaxation_power: float = 2.0,

        # Staged Inference
        stages: int = 1,
        bidirectional_graph: bool = True,
        refresh_gate_each_stage: bool = True,

        # Constraints
        min_tri_quality: float = 0.02,
        normal_cos_thresh: float = 0.5,
        imp_hard_thresh: float = 0.95,
        sharp_hard_thresh: float = 0.98,

        # Stability Parameters
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

    def _predict_importance(cur_mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if hasattr(data, "edge_feature_weight"):
            sharp = data.edge_feature_weight.detach().cpu().numpy().reshape(-1)

        undir_edges = data.undirected_edges.detach().cpu().numpy()
        return imp_undir, undir_edges, sharp

    stages = max(1, int(stages))
    orig_faces = int(len(mesh.faces))
    cur_mesh = mesh

    if stages == 1:
        stage_ratios = [target_face_ratio]
    else:
        step = target_face_ratio ** (1.0 / stages)
        stage_ratios = [step ** (i + 1) for i in range(stages)]
        stage_ratios[-1] = target_face_ratio

    for si, r in enumerate(stage_ratios, start=1):
        target_faces_abs = max(4, int(orig_faces * r))
        cur_faces = int(len(cur_mesh.faces))
        stage_ratio = target_faces_abs / max(cur_faces, 1)

        print(f"\n[Stage {si}/{len(stage_ratios)}] Faces: {cur_faces} -> {target_faces_abs} (r={stage_ratio:.4f})")

        # ✅ Staged Refresh: Importance is re-calculated per stage
        imp_undir, undir_edges, sharp = _predict_importance(cur_mesh)

        if export_importance_map:
            if importance_map_path is None:
                stem0 = os.path.splitext(os.path.basename(out_path or obj_path))[0]
                importance_map_path0 = os.path.join(os.path.dirname(out_path or obj_path), f"{stem0}_imp_stage{si}.ply")
            else:
                base0, ext0 = os.path.splitext(importance_map_path)
                ext0 = ext0 or ".ply"
                importance_map_path0 = f"{base0}_stage{si}{ext0}"
            save_importance_map(cur_mesh, undir_edges, imp_undir, importance_map_path0, cmap=importance_map_cmap)

        # ✅ Continuous Relaxation is handled INSIDE edge_collapse_simplify.
        cur_mesh = edge_collapse_simplify(
            mesh=cur_mesh,
            importance_undir=imp_undir,
            undirected_edges=undir_edges,
            edge_feature_weight=sharp,
            target_face_ratio=float(stage_ratio),

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

    simplified = cur_mesh

    if out_path is None:
        stem = os.path.splitext(os.path.basename(obj_path))[0]
        out_path = os.path.join(os.path.dirname(obj_path), f"{stem}_simplified.obj")

    if write_vertex_normals:
        export_obj_with_vertex_normals(simplified, out_path)
    else:
        simplified.export(out_path)

    print(f"[Done] saved: {out_path}")
    return simplified


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="checkpoints/edge_gnn_tosc.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--out", type=str, default=None)

    # Simplified Args for Paper Version
    parser.add_argument("--gate_c0", type=float, default=0.20, help="Initial strict admissibility threshold (e.g. 0.2)")
    parser.add_argument("--relaxation_power", type=float, default=2.0, help="Curve power for gate relaxation")

    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--refresh_gate_each_stage", type=int, default=1)

    parser.add_argument("--min_tri_q", type=float, default=0.02)
    parser.add_argument("--normal_cos", type=float, default=0.5)
    parser.add_argument("--imp_hard", type=float, default=0.95)
    parser.add_argument("--sharp_hard", type=float, default=0.98)

    # Stability Params
    parser.add_argument("--tri_quality_penalty", type=int, default=0, help="Enable soft penalty (1/0)")
    parser.add_argument("--tri_penalty_w", type=float, default=0.35)
    parser.add_argument("--tri_q_target", type=float, default=0.08)
    parser.add_argument("--max_valence", type=int, default=0, help="Max valence guard (0=disable)")
    parser.add_argument("--max_valence_increase", type=int, default=0, help="Max valence increase (0=disable)")
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
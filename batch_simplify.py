# -*- coding: utf-8 -*-
"""
Batch simplification driver (Final Research Version).

Cleaned up for B-Scheme:
- Removed obsolete score_mode/alpha/beta.
- Added stability params (max_valence, etc.).
- Added relaxation_power.
"""

import os
import argparse
from simplify_mesh import simplify_with_model


def _fmt_float(x: float) -> str:
    s = f"{x:g}"
    return s.replace(".", "p")


def batch_simplify(
        input_dir="data_tosc/toscahires_obj",
        output_dir="data_tosc/toscahires_gnn",
        ckpt_path="checkpoints/edge_gnn_tosc.pt",
        device="cpu",
        ratio=0.5,
        # ✅ Core B-Scheme Param
        gate_c0=0.20,
        relaxation_power=2.0,
        # Staged
        stages=1,
        refresh_gate_each_stage=True,
        # Graph
        bidirectional_graph=True,
        # Constraints
        min_tri_q=0.02,
        normal_cos=0.5,
        imp_hard=0.95,
        sharp_hard=0.98,
        # ✅ Stability / Open3D-like
        tri_quality_penalty=False,
        tri_penalty_w=0.35,
        tri_q_target=0.08,
        max_valence=0,
        max_valence_increase=0,
        edge_len_penalty_w=0.0,
        # Misc
        log_every=200,
        no_vn=False,
        export_imp_map=False,
        imp_map_dir=None,
):
    os.makedirs(output_dir, exist_ok=True)

    if export_imp_map:
        out_imp_dir = imp_map_dir or output_dir
        os.makedirs(out_imp_dir, exist_ok=True)

    objs = [f for f in os.listdir(input_dir) if f.lower().endswith(".obj")]
    if not objs:
        print("❌ No OBJ files found.")
        return

    print(f"🔍 Found {len(objs)} OBJ meshes. Starting batch simplification...")
    print(
        "🧾 Params: "
        f"ratio={ratio}, gate_c0={gate_c0}, power={relaxation_power}, stages={stages}, "
        f"refresh={int(bool(refresh_gate_each_stage))}, "
        f"tri_pen={int(bool(tri_quality_penalty))}, max_val={max_valence}, "
        f"bi={int(bidirectional_graph)}, vn={int(not no_vn)}"
    )

    for fname in objs:
        in_path = os.path.join(input_dir, fname)
        stem = os.path.splitext(fname)[0]

        rstag = f"_rg{int(bool(refresh_gate_each_stage))}"
        tqtag = "_tp1" if tri_quality_penalty else "_tp0"
        valtag = f"_mv{max_valence}" if max_valence > 0 else ""
        powtag = f"_p{_fmt_float(relaxation_power)}"

        out_name = (
            f"{stem}_GNN_r{int(ratio * 100)}"
            f"_st{stages}"
            f"_g{_fmt_float(gate_c0)}{powtag}"
            f"{rstag}{tqtag}{valtag}"
            f".obj"
        )
        out_path = os.path.join(output_dir, out_name)

        print(f"\n📌 Processing: {fname}  →  {out_name}")
        try:
            simplify_with_model(
                obj_path=in_path,
                ckpt_path=ckpt_path,
                device=device,
                target_face_ratio=ratio,
                out_path=out_path,
                gate_c0=gate_c0,
                relaxation_power=relaxation_power,
                stages=stages,
                bidirectional_graph=bidirectional_graph,
                min_tri_quality=min_tri_q,
                normal_cos_thresh=normal_cos,
                imp_hard_thresh=imp_hard,
                sharp_hard_thresh=sharp_hard,
                log_every=log_every,
                write_vertex_normals=(not no_vn),
                export_importance_map=bool(export_imp_map),
                importance_map_path=(
                    None
                    if not export_imp_map
                    else os.path.join((imp_map_dir or output_dir), f"{stem}_imp.ply")
                ),
                refresh_gate_each_stage=bool(refresh_gate_each_stage),
                tri_quality_penalty=bool(tri_quality_penalty),
                tri_penalty_w=float(tri_penalty_w),
                tri_q_target=float(tri_q_target),
                max_valence=int(max_valence),
                max_valence_increase=int(max_valence_increase),
                edge_len_penalty_w=float(edge_len_penalty_w),
            )
        except Exception as e:
            print(f"⚠️ Failed: {fname}, error: {e}")
            import traceback
            traceback.print_exc()

    print("\n✨ Done! Output dir:", output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data_tosc/toscahires_obj")
    parser.add_argument("--output_dir", type=str, default="data_tosc/toscahires_gnn")
    parser.add_argument("--ckpt", type=str, default="checkpoints/edge_gnn_tosc.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ratio", type=float, default=0.5)

    # Core B-Scheme Param
    parser.add_argument("--gate_c0", type=float, default=0.20, help="Initial admissibility threshold")
    parser.add_argument("--relaxation_power", type=float, default=2.0)

    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--bidirectional_graph", type=int, default=1)
    parser.add_argument("--refresh_gate_each_stage", type=int, default=1)

    # Constraints
    parser.add_argument("--min_tri_q", type=float, default=0.02)
    parser.add_argument("--normal_cos", type=float, default=0.5)
    parser.add_argument("--imp_hard", type=float, default=0.95)
    parser.add_argument("--sharp_hard", type=float, default=0.98)

    # Stability
    parser.add_argument("--tri_quality_penalty", type=int, default=0)
    parser.add_argument("--tri_penalty_w", type=float, default=0.35)
    parser.add_argument("--tri_q_target", type=float, default=0.08)

    parser.add_argument("--max_valence", type=int, default=0)
    parser.add_argument("--max_valence_increase", type=int, default=0)
    parser.add_argument("--edge_len_penalty_w", type=float, default=0.0)

    # Misc
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--no_vn", action="store_true")
    parser.add_argument("--export_imp_map", action="store_true")
    parser.add_argument("--imp_map_dir", type=str, default=None)

    args = parser.parse_args()

    batch_simplify(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ckpt_path=args.ckpt,
        device=args.device,
        ratio=args.ratio,
        gate_c0=args.gate_c0,
        relaxation_power=args.relaxation_power,
        stages=args.stages,
        bidirectional_graph=bool(args.bidirectional_graph),
        min_tri_q=args.min_tri_q,
        normal_cos=args.normal_cos,
        imp_hard=args.imp_hard,
        sharp_hard=args.sharp_hard,
        tri_quality_penalty=bool(args.tri_quality_penalty),
        tri_penalty_w=args.tri_penalty_w,
        tri_q_target=args.tri_q_target,
        max_valence=args.max_valence,
        max_valence_increase=args.max_valence_increase,
        edge_len_penalty_w=args.edge_len_penalty_w,
        log_every=args.log_every,
        no_vn=args.no_vn,
        export_imp_map=args.export_imp_map,
        imp_map_dir=args.imp_map_dir,
    )
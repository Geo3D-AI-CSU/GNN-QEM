# -*- coding: utf-8 -*-
"""
Multi-Ratio Batch simplification driver.
Automatically iterates through a list of ratios, creating separate output directories,
CSV metrics files, and importance map directories for each ratio.
"""

import os
import argparse
import pandas as pd
from simplify_mesh import simplify_with_model  # 确保这里和你的主环境一致


def _fmt_float(x: float) -> str:
    s = f"{x:g}"
    return s.replace(".", "p")


def batch_simplify(
        input_dir,
        output_dir,
        csv_out,
        ckpt_path,
        device="cpu",
        ratio=0.5,
        strategy="staged_local",  # <--- [新增修改]: 接收松弛策略参数
        hybrid_alpha=0.5,  # <--- [新增修改]: 接收混合策略的权重参数
        gate_c0=0.20,
        relaxation_power=2.0,
        stages=1,
        refresh_gate_each_stage=True,
        bidirectional_graph=True,
        min_tri_q=0.02,
        normal_cos=0.5,
        imp_hard=0.95,
        sharp_hard=0.98,
        tri_quality_penalty=False,
        tri_penalty_w=0.35,
        tri_q_target=0.08,
        max_valence=0,
        max_valence_increase=0,
        edge_len_penalty_w=0.0,
        log_every=200,
        no_vn=False,
        export_imp_map=False,
        imp_map_dir=None,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)

    if export_imp_map:
        out_imp_dir = imp_map_dir or output_dir
        os.makedirs(out_imp_dir, exist_ok=True)

    objs = [f for f in os.listdir(input_dir) if f.lower().endswith(".obj")]
    if not objs:
        print(f"❌ No OBJ files found in {input_dir}.")
        return

    print(f"🔍 Found {len(objs)} OBJ meshes for Ratio = {ratio}. Starting batch...")

    results_list = []

    for fname in objs:
        in_path = os.path.join(input_dir, fname)
        stem = os.path.splitext(fname)[0]

        rstag = f"_rg{int(bool(refresh_gate_each_stage))}"
        tqtag = "_tp1" if tri_quality_penalty else "_tp0"
        valtag = f"_mv{max_valence}" if max_valence > 0 else ""
        powtag = f"_p{_fmt_float(relaxation_power)}"

        # 为了在 Ablation Study 中方便对比结果，对 strategy 名字做个简写处理
        # staged_local -> sl, global_monotonic -> gm, hybrid -> hy, adaptive_hybrid -> ahy
        strat_tag = strategy
        if strategy == "staged_local":
            strat_tag = "sl"
        elif strategy == "global_monotonic":
            strat_tag = "gm"
        elif strategy == "hybrid":
            strat_tag = f"hy{_fmt_float(hybrid_alpha)}"
        elif strategy == "adaptive_hybrid":
            strat_tag = "ahy"

        out_name = (
            f"{stem}_GNN_r{int(ratio * 100)}"
            f"_st{stages}"
            f"_g{_fmt_float(gate_c0)}{powtag}"
            f"_{strat_tag}"  # <--- [新增修改]: 在输出文件名中记录当前策略，避免结果混淆
            f"{rstag}{tqtag}{valtag}"
            f".obj"
        )
        out_path = os.path.join(output_dir, out_name)

        print(f"\n📌 Processing: {fname}  →  {out_name}")

        row = {
            "model": fname,
            "ratio_target": ratio,
            "method": "GNN_Guided_QEM",
            "status": "fail",
            "error": ""
        }

        try:
            simplified_mesh, stats = simplify_with_model(
                obj_path=in_path,
                ckpt_path=ckpt_path,
                device=device,
                target_face_ratio=ratio,
                out_path=out_path,
                strategy=strategy,  # <--- [新增修改]: 传入 strategy 给 simplify_mesh
                hybrid_alpha=hybrid_alpha,  # <--- [新增修改]: 传入 hybrid_alpha
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
                    None if not export_imp_map else os.path.join((imp_map_dir or output_dir), f"{stem}_imp.ply")
                ),
                refresh_gate_each_stage=bool(refresh_gate_each_stage),
                tri_quality_penalty=bool(tri_quality_penalty),
                tri_penalty_w=float(tri_penalty_w),
                tri_q_target=float(tri_q_target),
                max_valence=int(max_valence),
                max_valence_increase=int(max_valence_increase),
                edge_len_penalty_w=float(edge_len_penalty_w),
            )

            row.update(stats)
            row["status"] = "ok"

        except Exception as e:
            print(f"⚠️ Failed: {fname}, error: {e}")
            row["error"] = str(e)
            import traceback
            traceback.print_exc()

        results_list.append(row)

    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(csv_out, index=False, encoding='utf-8-sig')
        print("\n======================================")
        print(f"✨ Ratio {ratio} Done!")
        print(f"📂 Output dir: {output_dir}")
        print(f"📊 Metrics saved to: {csv_out}")
        print("======================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Ratio Batch Simplification")

    # 修改1: 统一的基础输出路径和多个比例列表
    parser.add_argument("--input_dir", type=str, required=True, help="输入模型目录")
    parser.add_argument("--base_output_dir", type=str, required=True, help="所有比例结果的基础输出目录")
    parser.add_argument("--ratios", type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.5],
                        help="需要执行的简化率列表，例如: 0.05 0.1 0.2 0.5")

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    # <--- [新增修改开始]: 暴露 Strategy 参数到命令行
    parser.add_argument("--strategy", type=str, default="staged_local",
                        choices=["staged_local", "global_monotonic", "hybrid", "adaptive_hybrid"],
                        help="选择松弛进度的计算策略 (默认: staged_local)")
    parser.add_argument("--hybrid_alpha", type=float, default=0.5,
                        help="在使用 hybrid 策略时，全局进度的权重 alpha (默认: 0.5)")
    # [新增修改结束] --->

    parser.add_argument("--gate_c0", type=float, default=0.20)
    parser.add_argument("--relaxation_power", type=float, default=2.0)
    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--bidirectional_graph", type=int, default=1)
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
    parser.add_argument("--export_imp_map", action="store_true")

    args = parser.parse_args()

    # 修改2: 循环遍历每一个简化率
    for r in args.ratios:
        ratio_str = int(r * 100)  # 将 0.05 转为 5，0.1 转为 10

        # 自动构建当前比例专属的输出路径
        current_out_dir = os.path.join(args.base_output_dir, f"ratio_{ratio_str}")
        current_csv_out = os.path.join(current_out_dir, f"metrics_ratio{ratio_str}.csv")
        current_imp_map_dir = os.path.join(current_out_dir, "imp_maps") if args.export_imp_map else None

        print(f"\n" + "#" * 60)
        print(f"🚀 STARTING RATIO: {r} (Targeting {ratio_str}%) | Strategy: {args.strategy}")
        print(f"#" * 60)

        batch_simplify(
            input_dir=args.input_dir,
            output_dir=current_out_dir,
            csv_out=current_csv_out,
            ckpt_path=args.ckpt,
            device=args.device,
            ratio=r,
            strategy=args.strategy,  # <--- [新增修改]: 透传参数
            hybrid_alpha=args.hybrid_alpha,  # <--- [新增修改]: 透传参数
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
            imp_map_dir=current_imp_map_dir,
        )
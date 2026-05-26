#!/usr/bin/env python3
"""Plot kernel agent NCU-guided optimization trajectory across problems.

Reads opt_logs/program_database.json and opt_logs/artifacts/benchmark_results.json
from each problem directory. Produces:
  1. A two-panel PNG (per-problem lines + aggregate mean/median/IQR)
  2. A text summary table printed to stdout

Usage:
    python plot_perf_trajectory.py [PROBLEM_DIR] [-o OUTPUT.png]

    PROBLEM_DIR  Directory containing problem subdirectories (default: script's parent dir)
    -o           Output PNG path (default: PROBLEM_DIR/perf_trajectory.png)
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_problems(root_dir):
    problems = []
    for prob_id in sorted(os.listdir(root_dir)):
        prob_dir = os.path.join(root_dir, prob_id)
        db_path = os.path.join(prob_dir, "opt_logs", "program_database.json")
        init_bench = os.path.join(prob_dir, "opt_logs", "artifacts", "benchmark_results.json")
        bench_path = os.path.join(prob_dir, "benchmark.json")

        if not os.path.isfile(db_path):
            continue

        with open(db_path) as f:
            db = json.load(f)

        initial_ms = None
        if os.path.isfile(init_bench):
            with open(init_bench) as f:
                ib = json.load(f)
            initial_ms = ib.get("kernels", {}).get("initial_kernel", {}).get("time_ms")

        eager_ms = compile_ms = None
        if os.path.isfile(bench_path):
            with open(bench_path) as f:
                bench = json.load(f)
            eager_ms = bench.get("eager_ms")
            compile_ms = bench.get("torch_compile", {}).get("time_ms")

        round_times = []
        for prog in db.get("programs", []):
            t = prog.get("metrics", {}).get("time_ms")
            gen = prog.get("generation", 0)
            round_times.append((gen, t))
        round_times.sort(key=lambda x: x[0])

        if not (initial_ms and initial_ms > 0):
            continue

        cum_best = []
        best = initial_ms
        for _gen, t in round_times:
            if t is not None:
                best = min(best, t)
            cum_best.append(best / initial_ms)

        problems.append({
            "id": prob_id,
            "initial_ms": initial_ms,
            "eager_ms": eager_ms,
            "compile_ms": compile_ms,
            "cum_best_ratio": cum_best,
            "cum_best_abs": [r * initial_ms for r in cum_best],
            "n_rounds": len(cum_best),
            "final_best_ms": best,
            "final_speedup": initial_ms / best if best > 0 else 1.0,
        })

    return problems


def print_table(problems):
    print(f"{'Problem':<14} {'Eager':>7} {'Compile':>8} {'Initial':>8}  ", end="")
    for i in range(1, 6):
        print(f"{'R'+str(i)+' best':>8}  ", end="")
    print(f"{'Final':>8} {'vs Eager':>9} {'vs Comp':>8} {'vs Init':>8}")
    print("=" * 145)

    for p in problems:
        cum = p["cum_best_abs"]
        while len(cum) < 5:
            cum.append(None)

        row = f"{p['id']:<14} "
        row += f"{p['eager_ms']:>7.3f} " if p['eager_ms'] else f"{'—':>7} "
        row += f"{p['compile_ms']:>8.3f} " if p['compile_ms'] else f"{'—':>8} "
        row += f"{p['initial_ms']:>8.3f}  "

        for i in range(5):
            if cum[i] is not None:
                row += f"{cum[i]:>8.4f}  "
            else:
                row += f"{'—':>8}  "

        final = p["final_best_ms"]
        row += f"{final:>8.4f} "
        row += f"{p['eager_ms']/final:>8.2f}x " if p['eager_ms'] else f"{'—':>9} "
        row += f"{p['compile_ms']/final:>7.2f}x " if p['compile_ms'] else f"{'—':>8} "
        row += f"{p['final_speedup']:>7.2f}x"
        print(row)

    print("=" * 145)
    n = len(problems)
    beat_eager = sum(1 for p in problems if p['eager_ms'] and p['final_best_ms'] < p['eager_ms'])
    beat_compile = sum(1 for p in problems if p['compile_ms'] and p['final_best_ms'] < p['compile_ms'])
    improved = sum(1 for p in problems if p['final_speedup'] > 1.05)
    print(f"\nSUMMARY ({n} problems)")
    print(f"  Improved >1.05x over initial: {improved}/{n}")
    print(f"  Beat torch.compile:           {beat_compile}/{n}")
    print(f"  Beat eager:                   {beat_eager}/{n}")

    sps = [p['final_speedup'] for p in problems]
    print(f"  Speedup vs initial — mean: {np.mean(sps):.2f}x  median: {np.median(sps):.2f}x")

    print("\nPER-ROUND TRAJECTORY (cumulative best / initial):")
    for rnd in range(5):
        ratios = [p["cum_best_ratio"][rnd] for p in problems if rnd < len(p["cum_best_ratio"])]
        if ratios:
            print(f"  R{rnd+1}: mean {np.mean(ratios):.3f}  med {np.median(ratios):.3f}  ({len(ratios)} problems)")


def plot(problems, out_path):
    problems_sorted = sorted(problems, key=lambda p: p["final_speedup"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left: per-problem lines ---
    ax1 = axes[0]
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=0.8, vmax=max(p["final_speedup"] for p in problems_sorted))

    for p in problems_sorted:
        x = list(range(len(p["cum_best_ratio"])))
        y = p["cum_best_ratio"]
        color = cmap(norm(p["final_speedup"]))
        label = f'{p["id"][:6]}.. ({p["final_speedup"]:.1f}x)'
        ax1.plot(x, y, '-o', color=color, markersize=4, linewidth=1.5, alpha=0.8, label=label)

    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Initial kernel')
    ax1.set_xlabel('Optimization Round', fontsize=12)
    ax1.set_ylabel('Best Time / Initial Time', fontsize=12)
    ax1.set_title('Per-Problem Optimization Trajectory', fontsize=14)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels([f'R{i+1}' for i in range(5)])
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # --- Right: aggregate ---
    ax2 = axes[1]
    max_rounds = 5
    stats = {"avg": [], "med": [], "p25": [], "p75": [], "lo": [], "hi": []}
    for rnd in range(max_rounds):
        ratios = [p["cum_best_ratio"][rnd] for p in problems if rnd < len(p["cum_best_ratio"])]
        if ratios:
            stats["avg"].append(np.mean(ratios))
            stats["med"].append(np.median(ratios))
            stats["p25"].append(np.percentile(ratios, 25))
            stats["p75"].append(np.percentile(ratios, 75))
            stats["lo"].append(np.min(ratios))
            stats["hi"].append(np.max(ratios))

    x = list(range(len(stats["avg"])))

    ax2.fill_between(x, stats["lo"], stats["hi"], alpha=0.1, color='steelblue', label='Min–Max range')
    ax2.fill_between(x, stats["p25"], stats["p75"], alpha=0.25, color='steelblue', label='IQR (25th–75th)')
    ax2.plot(x, stats["avg"], '-s', color='navy', linewidth=2.5, markersize=8, label='Mean', zorder=5)
    ax2.plot(x, stats["med"], '-D', color='darkorange', linewidth=2.5, markersize=8, label='Median', zorder=5)

    for i in range(len(stats["avg"])):
        ax2.annotate(f'{stats["avg"][i]:.3f}', (i, stats["avg"][i]),
                     textcoords="offset points", xytext=(12, 8), fontsize=9, color='navy', fontweight='bold')
        ax2.annotate(f'{stats["med"][i]:.3f}', (i, stats["med"][i]),
                     textcoords="offset points", xytext=(12, -12), fontsize=9, color='darkorange', fontweight='bold')

    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Initial kernel (1.0x)')
    ax2.set_xlabel('Optimization Round', fontsize=12)
    ax2.set_ylabel('Best Time / Initial Time (lower = better)', fontsize=12)
    n = len(problems)
    ax2.set_title(f'Aggregate Optimization Trajectory (N={n})', fontsize=14)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels([f'R{i+1}' for i in range(5)])
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)

    textstr = (f'Round 1 → {len(stats["avg"])} improvement:\n'
               f'  Mean:   {(1-stats["avg"][0])*100:.1f}% → {(1-stats["avg"][-1])*100:.1f}%\n'
               f'  Median: {(1-stats["med"][0])*100:.1f}% → {(1-stats["med"][-1])*100:.1f}%')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax2.text(0.02, 0.02, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem_dir", nargs="?", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory containing problem subdirectories")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    parser.add_argument("--no-plot", action="store_true", help="Print table only, skip PNG generation")
    args = parser.parse_args()

    out_path = args.output or os.path.join(args.problem_dir, "perf_trajectory.png")

    problems = load_problems(args.problem_dir)
    if not problems:
        print(f"No problems found in {args.problem_dir}", file=sys.stderr)
        sys.exit(1)

    print_table(problems)

    if not args.no_plot:
        plot(problems, out_path)


if __name__ == "__main__":
    main()

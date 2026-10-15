# v8 GB300 benchmark comparison

Fresh measurements for all 20 problems. The [consolidated JSON](benchmark_comparison.json) contains every backend entry, hardware/software provenance, strict parity results, and timing ranges.

`simple_prompt_kernel` is **3.50x faster than eager** by geometric mean. For the fastest other backend passing its numerical tolerance, the geometric mean of other/simple timing ratios is **1.25x** across 20 paired problems. Simple has the lower median in 17/20; differences close to 1x should be read alongside the timing ranges.

- All timings are fresh NVIDIA GB300 measurements. For each problem, eager and all available backends used the same physical GPU, identical seed-0 inputs, software environment, and CUDA-graph timing method.
- Time is the median of 15 interleaved CUDA-event samples, with 50 calls per graph replay. Compilation, five function warmups, five graph warmups, and one replay before each sample are excluded. Outputs are released after each captured call.
- Both compiled backends use torch.compile with inductor, fullgraph=True, dynamic=False and internal CUDA graphs disabled. The max-autotune entry enables max_autotune. Each backend uses the same external CUDA-graph timing procedure.
- Other-backend PASS requires numerical agreement on three seeds under the previously stored atol_eff/rtol, or 0.05/0.01 where absent. Compiled outputs receive the same check. Historical bitwise tolerance labels are renamed pointwise; they never imply byte equality.
- simple_prompt_kernel PASS requires strict byte equality and memory parity. The consolidated report's per_problem_measurements section contains environment and parity details; per-problem benchmark.json files retain their original result format. Checks use seeds 17, 113, 2027, three calls per seed, retained-output checks, unchanged inputs, matching output metadata/aliases, and peak allocation no higher than eager.
- Ratios include only PASS entries with positive timings. Timings for numerical failures are retained but excluded from those ratios. Very small differences can fall within measurement variation.
- Roofline notes retain source-matched prior profiling; this rerun measures latency and parity, not new roofline profiles. Nine simple kernels still lack an established roofline result.

All times are milliseconds. Failed results show their timing but do not count as passing alternatives.

| Problem | eager | kernelagent_triton | kernelagent_triton_opt | torch_compile | torch_compile_max_autotune | simple_prompt_kernel | Best passing other / simple |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [12545c1e9b06](12545c1e9b06/benchmark.json) | 0.255284 | 0.088367 | 0.085546 | 0.089069 | 0.085417 | 0.067279 | 1.27x |
| [385e1beb9bb9](385e1beb9bb9/benchmark.json) | 0.316645 | 0.192184 | 0.194565 | 0.195340 | 0.195550 | 0.195878 | 0.98x |
| [3c6c589cdf81](3c6c589cdf81/benchmark.json) | 0.171651 | 0.264055 | 0.141314 | 0.132675 | 0.131154 | 0.095955 | 1.37x |
| [43f0c930291e](43f0c930291e/benchmark.json) | 0.180123 | 0.129681 | SKIP | 0.082025 (FAIL) | 0.071130 (FAIL) | 0.104634 | 1.24x |
| [493cad38cac9](493cad38cac9/benchmark.json) | 0.162557 | 0.132841 | 0.021676 | 0.107568 | 0.127310 | 0.018378 | 1.18x |
| [4b90e2d83144](4b90e2d83144/benchmark.json) | 0.022720 | 0.018010 | 0.018419 | 0.021286 | 0.024727 | 0.016576 | 1.09x |
| [622c35f5d1c6](622c35f5d1c6/benchmark.json) | 3.913127 | 0.935764 | SKIP | 1.165141 | 1.077404 | 0.865498 | 1.08x |
| [71f1d2c9f38b](71f1d2c9f38b/benchmark.json) | 5.033227 | 0.851540 | 0.850762 | 1.073788 | 1.082963 | 0.517264 | 1.64x |
| [7815074de963](7815074de963/benchmark.json) | 0.392527 | 0.130132 | 0.130225 | 0.135994 | 0.133454 | 0.120100 | 1.08x |
| [7884328a5557](7884328a5557/benchmark.json) | 0.052922 | 0.026748 | 0.026317 | 0.029718 | 0.029738 | 0.025663 | 1.03x |
| [7e38211c7f1c](7e38211c7f1c/benchmark.json) | 0.034925 | SKIP | SKIP | 0.030133 | 0.025894 | 0.004164 | 6.22x |
| [8e33a91b7364](8e33a91b7364/benchmark.json) | 1.107343 | 0.328815 | 0.329020 | 0.324881 | 0.324801 | 0.326849 | 0.99x |
| [94985bdb7630](94985bdb7630/benchmark.json) | 0.392814 | 0.130170 | 0.216645 | 0.135620 | 0.133204 | 0.119850 | 1.09x |
| [97032595d968](97032595d968/benchmark.json) | 0.217553 | 0.048873 | 0.042747 | 0.151786 | 0.169234 | 0.025608 | 1.67x |
| [c68dae22972b](c68dae22972b/benchmark.json) | 0.188286 | 0.097068 | 0.122032 | 0.103866 | 0.103868 | 0.096986 | 1.00x |
| [c8c60d8f7015](c8c60d8f7015/benchmark.json) | 0.210628 | 0.167519 | 0.032719 | 0.151585 | 0.169116 | 0.025428 | 1.29x |
| [d8f596b3df34](d8f596b3df34/benchmark.json) | 0.147550 | 0.201493 | 0.125842 | 0.071469 | 0.059979 | 0.082036 | 0.73x |
| [e29b92927600](e29b92927600/benchmark.json) | 0.035007 | 0.007967 | 0.004698 | 0.026345 | 0.030564 | 0.004124 | 1.14x |
| [f4e9e58fef73](f4e9e58fef73/benchmark.json) | 0.022739 | 0.030886 | 0.024501 | 0.021347 | 0.024623 | 0.017230 | 1.24x |
| [f6251e84e3fa](f6251e84e3fa/benchmark.json) | 1.917115 | 0.469736 | SKIP | 0.939078 (FAIL) | 0.939038 (FAIL) | 0.441925 | 1.06x |

PASS uses each backend's declared tolerance tier. Bitwise and memory columns are independent strict checks over the same three seeds; a numerical PASS alone does not establish either.

| Backend | PASS | FAIL | ERROR | SKIP | Bitwise PASS | Memory PASS | Both strict checks PASS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kernelagent_triton` | 19 | 0 | 0 | 1 | 2 | 12 | 2 |
| `kernelagent_triton_opt` | 16 | 0 | 0 | 4 | 1 | 10 | 1 |
| `torch_compile` | 18 | 2 | 0 | 0 | 7 | 19 | 7 |
| `torch_compile_max_autotune` | 18 | 2 | 0 | 0 | 6 | 19 | 6 |
| `simple_prompt_kernel` | 20 | 0 | 0 | 0 | 20 | 20 | 20 |

Ratios use `other["time_ms"] / simple_prompt_kernel["time_ms"]`, with only passing pairs.

| Other backend | Paired problems | Geomean ratio | Minimum | Maximum | Simple lower / tied / higher |
| --- | --- | --- | --- | --- | --- |
| kernelagent_triton | 19 | 1.61x | 0.98x | 7.23x | 18 / 0 / 1 |
| kernelagent_triton_opt | 16 | 1.28x | 0.99x | 1.81x | 15 / 0 / 1 |
| torch_compile | 18 | 1.90x | 0.87x | 7.24x | 15 / 0 / 3 |
| torch_compile_max_autotune | 18 | 1.94x | 0.73x | 7.41x | 15 / 0 / 3 |
| Best passing other | 20 | 1.25x | 0.73x | 6.22x | 17 / 0 / 3 |
| Eager | 20 | 3.50x | 1.32x | 9.73x | 20 / 0 / 0 |

Failures and missing implementations:

- `43f0c930291e`: `kernelagent_triton_opt`: SKIP (no optimized_kernel.py); `torch_compile`: FAIL (3 seeds; bitwise FAIL; memory PASS; numerical tolerance exceeded); `torch_compile_max_autotune`: FAIL (3 seeds; bitwise FAIL; memory PASS; numerical tolerance exceeded).
- `622c35f5d1c6`: `kernelagent_triton_opt`: SKIP (no optimized_kernel.py).
- `7e38211c7f1c`: `kernelagent_triton`: SKIP (no kernel.py); `kernelagent_triton_opt`: SKIP (no optimized_kernel.py).
- `f6251e84e3fa`: `kernelagent_triton_opt`: SKIP (no optimized_kernel.py); `torch_compile`: FAIL (3 seeds; bitwise FAIL; memory FAIL; numerical tolerance exceeded); `torch_compile_max_autotune`: FAIL (3 seeds; bitwise FAIL; memory FAIL; numerical tolerance exceeded).

All 20 simple kernel sources also match the earlier full validation: 161 cases, including edge inputs, passed bitwise and memory checks.

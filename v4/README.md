# KernelProblems v4 — Benchmark Review

## Per-problem comparison (all times in ms)

| problem | eager | triton | tri_opt | compile | comp_ma | **best** | **speedup** | key ops |
|---|---|---|---|---|---|---|---|---|
| `0ed358ecaed2` | 0.199 | 0.172 | 0.172 | 0.172 | 0.310 | **triton** | 1.2x | _to_copy, t |
| `18b859c50de2` | 1.486 | 1.500 | 1.066 | 1.051 | 2.382 | **compile** | 1.4x | _fused_rms_norm, _unsafe_view, add, clone... |
| `1e49beb8a123` | 0.306 | 0.424 | 0.100 | 0.166 | 0.992 | **triton_opt** | 3.0x | embedding, split_with_sizes, view |
| `2a844503364a` | 0.915 | 0.605 | 0.529 | 0.403 | 1.929 | **compile** | 2.3x | _fused_rms_norm, _unsafe_view, clone, reshape... |
| `363b7b58c971` | 0.945 | 0.772 | FAIL | 0.357 | 1.089 | **compile** | 2.7x | _fused_rms_norm, _fused_rms_norm_backward, _to_copy, _unsafe_view... |
| `4757b36375ae` | 0.128 | 0.096 | 0.095 | 0.160 | 0.259 | **triton_opt** | 1.3x | add, reshape |
| `4d1dc9709a0e` | 1.728 | 1.489 | 1.490 | 1.491 | 2.680 | **triton** | 1.2x | _to_copy, t |
| `65b4b565ed67` | 0.534 | 0.315 | 0.315 | 0.317 | 0.854 | **triton_opt** | 1.7x | mul, reshape, silu |
| `680e4c57cdc8` | 3.045 | 1.808 | 1.184 | 1.192 | 1.881 | **triton_opt** | 2.6x | _fused_rms_norm, _fused_rms_norm_backward, _to_copy, _unsafe_view... |
| `6b92d6e665ab` | 2.911 | 1.773 | 0.164 | 1.648 | 0.861 | **triton_opt** | 17.7x | _to_copy, index, mul, reshape... |
| `80ad8e737b06` | 23.109 | 90.312 | 29.984 | 2.985 | 4.899 | **compile** | 7.7x | _log_softmax, _log_softmax_backward_data, _to_copy, div... |
| `84496d824942` | 0.374 | 0.216 | FAIL | 0.201 | 0.503 | **compile** | 1.9x | _fused_rms_norm, _unsafe_view, clone, reshape... |
| `84c0a03c4374` | 0.786 | 2.192 | 0.361 | 0.290 | 0.946 | **compile** | 2.7x | _fused_rms_norm, _fused_rms_norm_backward, _to_copy, _unsafe_view... |
| `9c745ea697a7` | 0.922 | 4.328 | 0.362 | 0.357 | 1.076 | **compile** | 2.6x | _fused_rms_norm, _fused_rms_norm_backward, _to_copy, _unsafe_view... |
| `a1318bc813d2` | 1.561 | 0.785 | 0.773 | 0.769 | 0.497 | **compile_ma** | 3.1x | _to_copy, clone, mul, reshape... |
| `bca49a4df51c` | 0.160 | 0.071 | 0.068 | 0.160 | 0.214 | **triton_opt** | 2.4x | _to_copy, t |
| `c72358b8111e` | 1.479 | 2.119 | FAIL | 0.652 | 1.441 | **compile** | 2.3x | mul, reshape, silu, silu_backward... |
| `e00050757618` | 0.160 | 0.062 | 0.066 | 0.155 | 0.224 | **triton** | 2.6x | _to_copy, t |
| `f16356222984` | 0.199 | 0.172 | 0.172 | 0.172 | 0.308 | **triton** | 1.2x | _to_copy, t |
| `f7ad66842c19` | 0.659 | FAIL | 0.187 | 0.233 | 0.482 | **triton_opt** | 3.5x | _fused_rms_norm, _fused_rms_norm_backward, _to_copy, add... |
| `f91538eb2b54` | 0.246 | 0.095 | 0.095 | 0.168 | 0.291 | **triton_opt** | 2.6x | add, reshape |

## Summary

**Best backend distribution:** compile=8, compile_ma=1, triton=4, triton_opt=8

**Pass rates:** triton 20/21, triton_opt 18/21, compile 21/21, compile_ma 21/21

**Geomean speedup vs eager:**

| backend | geomean | range | n |
|---|---|---|---|
| triton | 1.12x | 0.21x – 2.58x | 20 |
| triton_opt | 2.08x | 0.77x – 17.70x | 18 |
| compile | 1.82x | 0.80x – 7.74x | 21 |
| compile_ma | 0.92x | 0.31x – 4.72x | 21 |

**Key findings:**

- **triton_opt** (NCU-optimized) delivers the highest geomean speedup (2.08x)
- **compile** wins on fused_rms_norm chains and loss computation (inductor cross-op fusion)
- **compile_max_autotune** wins on 1/21 (RoPE mul+clone), generally slower than default compile
- triton initial kernels have 1 correctness failure (`f7ad6`); triton_opt has 3 (`363b`, `84496`, `c723`)
- `6b92` (RoPE complex math) is the standout: triton_opt **17.7x** faster than eager, **10x** vs compile
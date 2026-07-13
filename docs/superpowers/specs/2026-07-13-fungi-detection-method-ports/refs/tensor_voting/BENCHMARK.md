# Tensor-voting complexity benchmark

The helper streams source fields and has time complexity
`O(H*W + A*(2*r + 1)^2)`, where `H x W` is the array shape, `A` is the count of
strictly positive tokens, and `r` is the source support radius. Initialization and the
source-order voter scan both traverse all `H*W` pixels. The helper allocates three image-sized
float64 accumulators and no `H x W x support` stack.

Measured after one JIT warm-up on 2026-07-13:

| Platform | Python arrays | Numba | Shape | Sigma | Active tokens | Wall time |
|---|---|---|---:|---:|---:|---:|
| Apple M4 Pro, macOS 26.5.1 | NumPy 2.3.5 | 0.62.1 | 800 x 600 | 2.25 | 480 | 0.007368 s |
| Apple M4 Pro, macOS 26.5.1 | NumPy 2.3.5 | 0.62.1 | 800 x 600 | 2.25 | 480,000 | 0.700010 s |

These timings are reproducibility observations, not cross-platform performance guarantees.
Runtime scales with support area as well as token count, so the selected source's default
`sigma=18.25` is not suitable for an everywhere-positive 800 x 600 response. The public helper
therefore documents a sparse-token input contract. Choosing a detector threshold or candidate
mask is a future detector adaptation and is intentionally not hidden inside the numerical port.

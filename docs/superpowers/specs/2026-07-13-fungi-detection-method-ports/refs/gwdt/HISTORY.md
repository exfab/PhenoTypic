# Vaa3D GWDT history audit

The immutable corpus is pinned at Vaa3D commit
`475e4ca92d4e51de10f1c05d80cef6615432c087` (2024-04-25). Its upstream path is
`released_plugins/v3d_plugins/neurontracing_vn2/app2/fastmarching_dt.h`.

`git log --follow` records the relevant history:

```text
5990590 2013-04-19 initial paper-era APP2 source addition
a7210aa5 2013-06-27 clarify one sqrt call as sqrt(double(...)) for Windows
54ec4967 2014-03-26 rename vn2 to neurontracing_vn2
```

The selected `fastmarching_dt` function at lines 33-199 was extracted from both
paper-era commit `a7210aa5` at
`released_plugins/v3d_plugins/vn2/app2/fastmarching_dt.h` and the pinned 2024 path.
Both extracts have SHA-256:

```text
5a4d930f2e4901b770a62e61ed5eb89f3583e4c3651c5abbb793e22430fda8c4
```

Reproduce the equivalence in a Vaa3D clone containing the pinned history:

```bash
git show a7210aa5:released_plugins/v3d_plugins/vn2/app2/fastmarching_dt.h \
  | sed -n '33,199p' | shasum -a 256
sed -n '33,199p' \
  released_plugins/v3d_plugins/neurontracing_vn2/app2/fastmarching_dt.h \
  | shasum -a 256
```

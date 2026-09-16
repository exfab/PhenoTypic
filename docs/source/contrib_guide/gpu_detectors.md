# Writing a GPU detector

This page is the reference for **authoring a `GpuDetector`** — the hooks you
implement, the capability fields you declare, and the two rules that decide
whether your detector can run inside another operation.

Read it before subclassing `GpuDetector`, and again before adding an
`OperationField` to any operation that will contain one.

```{admonition} Why a GPU detector is not just an ObjectDetector
:class: note
A `GpuDetector` in a CLI run does not execute per image. It triggers the
**staged engine**: CPU preprocess → resident-model GPU inference → CPU measure,
so a GPU node is never occupied by CPU work. Your detector is driven by the same
hooks in both the notebook path and the staged engine, but the staged engine
calls them from three different processes, on three different machines, at three
different times. Everything below follows from that.
```

## The hook surface

Subclass `GpuDetector` and implement **two** methods. Do not override
`_operate` — it is concrete, and the staged engine bypasses it entirely.

```python
from pydantic import PrivateAttr
from phenotypic.abc_ import GpuDetector


class MyDetector(GpuDetector):
    input_layer: str = "gray"
    model_size: str = "small"
    _model: object = PrivateAttr(default=None)

    def _ensure_model_loaded(self) -> None:
        """Build the model on first use. Idempotent."""
        if self._model is not None:
            return
        import torch                      # lazy: keep it out of import time
        self._model = ...

    def _infer_one(self, sample):
        """One preprocessed (H, W, 3) uint8 sample -> one result.

        Return a uint16 labeled objmap for output_kind="instance", or a bool
        mask for "semantic".
        """
        return ...
```

The base class supplies the rest of the chain:

| Hook | Supplied behaviour | Override when |
|---|---|---|
| `_preprocess(array)` | scales to `uint8` per `input_scaling`, stacks 2-D to `(H, W, 3)` | rarely — your model wants a different tensor layout |
| `_collate(samples)` | returns the list unchanged | your model takes a true `(N, C, H, W)` batch |
| `_infer_batch(batch)` | loops `_infer_one`, and is the **sole caller** of `_ensure_model_loaded` | you overrode `_collate` for real batching |
| `_write_object_output(image, result)` | writes `objmap`/`objmask`, applies `drop_frame_background` then the connectivity relabel | essentially never — the staged replay path depends on this exact behaviour |

**Defer model construction to `_ensure_model_loaded`, never `__init__`.** The
CLI constructs your detector to inspect it, serialise it, and split the pipeline
around it, in processes that have no GPU and may not have `torch` installed. A
model built in `__init__` makes `to_json()` require a GPU.

## Capability fields

These are pydantic fields, so they serialise and round-trip. Declare them on the
class; the engine reads them to decide how to drive you.

| Field | Meaning |
|---|---|
| `input_layer` | `rgb` / `gray` / `detect_mat` — the layer your model was trained on |
| `input_scaling` | `image_max` (per-image maximum) or `dtype_range` |
| `supports_batching` | whether `_infer_batch` accepts more than one sample |
| `output_kind` | `instance` (labeled objmap) or `semantic` (boolean mask) |
| `drop_frame_background` | zero the border-plurality label before relabeling |
| `split_disconnected_labels` | relabel by connected components after that |
| `connectivity` | `1` (4-neighbour) or `2` (8-neighbour) |

`drop_frame_background` matters more than it looks. A class-agnostic segmenter
can emit the plate background as a positive-labelled mask framing the image;
left in place it survives into measurement and bridges every colony it touches,
collapsing all instances into one blob.

## Determinism is a hard requirement

The staged engine may run your detector's *inputs* more than once (see the
container contract below), and a retried Stage 3 replays a recorded result rather
than re-inferring. Seed anything stochastic. A detector that returns different
labels for the same input produces artifacts that describe a state that never
existed, and nothing fails to say so.

## Do not assume you see one image

Your detector is invoked once per image today, but **a GPU round is a
full-dataset sweep with the model resident**, and the engine is free to
accumulate samples and issue one batched forward. Write `_infer_one` so it
depends only on its argument — no state carried between calls, no counters, no
"first image" special-casing.

```{admonition} Batching headroom
:class: tip
The engine currently calls `_collate([sample])` with a one-element list, so
`supports_batching` buys nothing *yet*. That is a gap, not a decision. If your
model benefits from batching, implement `_collate` and `_infer_batch` properly
anyway — the cost is small and the sweep structure exists so this can be turned
on without touching detectors.
```

## Nesting: the container contract

A `GpuDetector` may sit **inside** another operation — most usefully inside a
`CompositeDetector`, where its mask is merged with a CPU detector's. The staged
engine supports this by addressing the detector with a **tree path** and
substituting a replay stub at that path in Stage 3.

To split the pipeline it must answer one question about every container on the
path:

> **What image does this container hand its children?**

It cannot infer the answer — containers drive children in imperative `_operate`
bodies, and some pass a local intermediate that has no name in the operation
graph. So the engine keeps a small, closed table, and **refuses anything not in
it**.

```{admonition} Introduced with nested GPU staging
:class: important
See `docs/superpowers/specs/2026-09-15-nested-gpu-staging/design.md` §4.3.
```

### The two contracts

`"same"` — every child receives the container's **own input**
: Children are parallel branches; order does not affect what any of them sees.
  `CompositeDetector`, `CompositeEnhance`, `FilamentousFungiDetector`.

`"sequence"` — each child receives the **previous child's output**
: `ImagePipeline`, handled by the engine directly.

### This is a contract, not an observation

**A `CompositeDetector` is never sequential.** Its branches each receive the
image the composite itself was given, and are combined afterwards by `union`,
`intersection` or `overlap`. That is what makes it a *composite* rather than a
pipeline — PhenoTypic already has `ImagePipeline` for the sequential case, and a
composite whose branches chained would be one. `CompositeEnhance` carries the
same contract for the same reason.

So the entry in the engine's table is not a cached observation about today's
`_operate`; it restates the type's contract. **If you are changing a composite
so its branches no longer see the same input, you are not editing a composite —
stop and use a pipeline.**

### The contract is tested, not just asserted

Each container with a table entry carries a behavioural test that puts two
recording probe operations in its children and asserts what they received:

```python
def test_composite_branches_each_receive_the_composites_own_input():
    """The `"same"` contract, verified rather than declared."""
    seen = []

    class _Probe(ObjectDetector):
        tag: str
        def _operate(self, image):
            seen.append((self.tag, int(image.objmap[:].max())))
            image.objmask[:] = ...   # set a mask this probe alone would produce
            return image

    CompositeDetector(ops=[_Probe(tag="a"), _Probe(tag="b")],
                      mode="union").apply(load_synth_yeast_plate())

    # If branches were sequential, "b" would observe "a"'s objmap.
    assert seen == [("a", 0), ("b", 0)]
```

A declaration can lie and still pass every test. A probe cannot: if someone
makes the branches chain, this fails immediately and points at the contract.

### Adding a container

If you add an `OperationField` to an operation, a guard test enumerates every
`OperationField`-bearing class and requires each to be **either** in the
engine's table **or** on an explicit unsupported list with a reason. The suite
fails until you choose, so the decision lands at authoring time rather than in a
30,000-image run.

Put it on the unsupported list when no single answer is true for all of its
children. `TwoKFilamentousDetector` is the worked example: its `center_detector`
gets the original image, its `background_subtractor` gets a derived
`enhanced.copy()`, and its `branch_base` **mutates** `enhanced` in place.

The consequence is not silence. A `GpuDetector` nested inside an unsupported
container is refused, by name:

```text
GpuDetector at CompositeDetector/ops[0]/branch_base cannot be staged:
TwoKFilamentousDetector has no declared child-input contract — its children read
internal intermediates (_two_k_filamentous_detector.py:154,164).
```

That is the intended outcome. A wrong answer here means Stage 2 reads a layer
that was never prepared, and the run completes with different numbers.

## Testing your detector

Use the shared CPU fake rather than writing another one:

- `tests/_fakes/fake_gpu_detector.py` — `FakeGpuDetector`, a threshold detector
  with the full hook surface and no `torch`.
- For a test that round-trips a pipeline through JSON, register it first.
  `ImagePipeline.from_json` resolves classes **by bare name against the
  `phenotypic` namespace**, so a module-local fake raises `AttributeError`. The
  in-process pattern is the autouse fixture at
  `tests/unit/cli/test_staged_routing.py:21`; `tests/_fakes/register_fake_gpu.py`
  exists for the live SLURM dispatch test, where worker processes cannot reach a
  fixture.

## Gated weights and custom namespaces

- `PHENOTYPIC_ACCEPT_MODEL_LICENSE` plus `require_license_acceptance`
  (`detect/nn/_checkpoint_manager.py`) gate downloads of licence-restricted
  checkpoints.
- `PHENOTYPIC_PRELOAD_MODULES` lets a fresh SLURM worker import a
  self-registering module before `from_json`, so a detector defined outside the
  `phenotypic` namespace can be deserialised.

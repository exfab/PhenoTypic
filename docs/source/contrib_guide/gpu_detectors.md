# Writing a GPU detector

This page is the reference for **authoring a `GpuDetector`** — the hooks you
implement, the capability fields you declare, and the rules that decide where in
a pipeline your detector may be placed.

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
from phenotypic.sdk_.typing_ import GpuInputLayer


class MyDetector(GpuDetector):
    # Keep the alias, don't widen to `str` -- the Literal is what rejects a
    # typo'd layer name at construction instead of at first inference.
    input_layer: GpuInputLayer = "gray"
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
| `_infer_batch(batch)` | loops `_infer_one`, calling `_ensure_model_loaded` first | you overrode `_collate` for real batching — **keep the `_ensure_model_loaded()` call** |
| `_write_object_output(image, result)` | writes `objmap`/`objmask`, applies `drop_frame_background` then the connectivity relabel | essentially never — the staged replay path depends on this exact behaviour |

**Defer model construction to `_ensure_model_loaded`, never `__init__`.** The
CLI constructs your detector to inspect it, serialise it, and split the pipeline
around it, in processes that have no GPU and may not have `torch` installed. A
model built in `__init__` makes `to_json()` require a GPU.

**And make it idempotent, because it is called more than once and from more than
one place.** The base `_infer_batch` calls it before every batch, and the staged
engine *also* calls it directly, once per worker, before streaming a shard
(`_cli_staged_strategy.py`, `_cli_staged_slurm_worker.py`) — that up-front call
is what "resident model" means. Several shipped detectors call it from their own
helpers too. The `if self._model is not None: return` guard above is the whole
contract; without it the model is rebuilt per image on the GPU node.

:::{admonition} If you override `_infer_batch`, the `_ensure_model_loaded()` call is yours to keep
:class: warning
It is not an implementation detail of the base method. In the notebook path
(`op.apply(image)` → `_operate` → `_collate` → `_infer_batch`) there is **no
other caller**, so an override that drops it runs against an unbuilt model.

The staged CLI engine will not tell you: it loads the model up front, before
streaming a shard, so the same detector works there. A bug that passes every
batch run and fails the first interactive `apply()` is close to the worst shape
available — the CLI is where the large runs happen, and the notebook is where
detectors get developed.

If you would rather not carry the obligation, override `_infer_one` and leave
`_infer_batch` alone; the base loops it for you.
:::

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

The staged engine runs your detector's *inputs* more than once. When your
detector is nested, the operations ahead of it **inside its own branch** are
applied twice: once by Stage 2, to a provenance-detached copy, purely to build
the model input; and again by Stage 3, inside the enclosing operation, where
they are the ones actually recorded in the journal. A retried Stage 3 replays
the recorded result rather than re-inferring.

Seed anything stochastic — in your detector *and* in any operation you put ahead
of it in the same branch. A non-deterministic prefix means Stage 2 inferred from
an image Stage 3 never reconstructs, and nothing fails to say so.

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
on without touching detectors. An `_infer_batch` override must still call
`_ensure_model_loaded()` itself; see the warning under *The hook surface*.
```

## Nesting: the container contract

A `GpuDetector` may sit **inside** another operation, at any depth — most
usefully inside a `CompositeDetector`, where its mask is merged with a CPU
detector's. The staged engine supports this by addressing the detector with a
**tree path** and substituting a replay stub at that path in Stage 3.

That path — `("CompositeDetector", "ops[0]")`, say — is a single value used
three ways: it is the detector's recorded `pipeline_step_path`, the address
Stage 3 substitutes at, and (sanitised, plus 8 hex of the exact path) the
`<slot>` directory its Stage-2 signal is written under, e.g.
`.phenotypic/progress/stage2_raw/<dataset>/CompositeDetector__ops-0__dddf3676/<stem>.npy`.
One addressing scheme, not three.

To split the pipeline it must answer one question about every container on the
path:

> **What image does this container hand its children?**

It cannot infer the answer — containers drive children in imperative `_operate`
bodies, and some pass a local intermediate that has no name in the operation
graph. So the engine keeps a small, closed table, and **refuses anything not in
it**.

The table is populated lazily on first use (`_populate_child_contract`), because
the module is imported by CLI argument validation that may never ask a GPU
question and `phenotypic.detect`/`phenotypic.enhance` are heavy. If you ever add
an entry, add it **inside the existing single `update()`** rather than assigning
key by key: the guard is `if _CHILD_CONTRACT: return`, so a half-built table
lets a second thread return early and refuse a placement the design permits —
and in the GUI, which runs threaded, that shows up as a refusal alert and a
disabled Run button on a pipeline that is perfectly valid.

```{admonition} Introduced with nested GPU staging
:class: important
See `docs/superpowers/specs/2026-09-15-nested-gpu-staging/design.md` §4.3.
```

### Only composition primitives may carry one

Three classes, and nothing else:

`ImagePipeline` — `"sequence"`
: Each child receives the previous child's output. Handled directly in
  `_child_contract`, not by a table entry.

`CompositeDetector`, `CompositeEnhance` — `"parallel"`
: Every child receives the container's own input. Branches are parallel; order
  does not affect what any of them sees. These are the only two entries in
  `_CHILD_CONTRACT` (`phenotypic._cli._cli_validation`).

A `GpuDetector` anywhere else is **refused**. That includes domain detectors
whose current code would classify cleanly — see below.

**Entries match by `isinstance`, not by exact type.** A subclass a user actually
holds — `class Plotted(CompositeDetector, PlotImage)` — is accepted; an
exact-type lookup refused every such subclass, which was the same defect the
table avoided for `ImagePipeline`. The safety argument is preserved by a second
rule: **a subclass that overrides `_operate` is refused**, because the
child-input contract was verified against the *base's* `_operate` and a
replacement has been verified against nothing.

The contract answer is consumed by `_branch_prefix` in `_cli_pipeline_split.py`,
which dispatches on `_child_contract` and never on `isinstance`. A `"sequence"`
container contributes the ops preceding the step taken from it to
`StagePlan.stage2_prefix`; a `"parallel"` one contributes nothing, because its
children are branches and none runs "before" another; and the root pipeline
contributes nothing, because Stage 1 already applied and stored its ops.

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

Each of the two `"parallel"` primitives carries a behavioural probe in
`tests/unit/detect/test_container_child_contracts.py`, which puts two recording
operations in the container's children and asserts what each received:

```python
def test_composite_detector_branches_each_receive_the_composites_own_input():
    image = load_synth_yeast_plate()
    baseline = int(image.objmap[:].max())
    assert baseline != _PROBE_OBJECT_COUNT, "fixture collision"

    CompositeDetector(
        ops=[_DetectProbe(tag="a"), _DetectProbe(tag="b")], mode="union"
    ).apply(image)

    # Sequential branches would have "b" observing "a"'s objmap.
    assert _SEEN == [("a", baseline), ("b", baseline)]
```

**Assert the property, not a literal.** `load_synth_yeast_plate()` arrives with
its 8×12 ground-truth objmap already populated, so a branch observes 96 on
entry, not 0. A test hard-coding `0` would be testing the fixture rather than
the contract, and a spike on this change was derailed by exactly that number
once already. The probe writes one deliberate square instead of thresholding,
because a threshold on the synthetic plate also yields ~96 objects and so would
not discriminate chaining from parallelism — hence the explicit
fixture-collision guard.

A declaration can lie and still pass every test that reads the table. A probe
cannot: if someone makes the branches chain, this fails immediately and points
at the contract. A companion test in
`tests/unit/cli/test_gpu_detection_tree_wide.py` asserts coverage of the **table
itself**, so a third `"parallel"` entry cannot arrive without its own probe.

### Why domain detectors are excluded, even when they would classify

`FilamentousFungiDetector` passes `inoculum_detector` the container's own image
(`_filamentous_fungi_detector.py:398`), so it *reads* as `"parallel"`. It is
still refused.

The reason is that a composition primitive's child-input semantics is part of
**what the class is**, while a domain detector's is incidental to **what its
algorithm currently does**. `FilamentousFungiDetector._operate` also runs an
inline `ContrastStretching()` (`:418`), a destructive `_subtract_background`,
and a `del enhanced_work`. Nothing about being a fungus detector constrains it
to keep handing its child the raw image, and if it stopped, the table would be
wrong with nothing to say so.

`TwoKFilamentousDetector` shows how tangled this gets inside one algorithm:
`center_detector` gets the original image
(`_two_k_filamentous_detector.py:166`), `background_subtractor` a derived
`enhanced.copy()` (`:171`), and `branch_base` **mutates** `enhanced` in place
(`:204`). Three fields, three inputs, none of it visible from outside.

So the rule is by *kind of class*, not by whether today's code happens to be
classifiable. A new detector needs no entry and no decision — it is refused by
default, which is the right answer for it.

Both classes do route their children through `apply_child`, so their branches
record distinct `pipeline_step_path`s. That is a provenance-addressing fix and
**not** an admission to the staging table; do not read one from the other.

### What you get instead of silence

Every refusal below raises `UnstageableGpuDetectorError`, a `ValueError`
subclass, from `find_gpu_detectors`. The **placement** refusals are
unconditional, so they fire from `pipeline_requires_gpu` — *before* the run is
routed, rather than after it has already been sent somewhere. (The
more-than-one-detector refusal is the exception: it is gated on `strict=True`,
which only `split_pipeline_at_gpu` passes, so the GUI's "is this a GPU
pipeline?" probe reports `True` for a two-detector pipeline instead of raising.)

Nested in something that is not a composition primitive:

```text
a GpuDetector cannot be nested inside TwoKFilamentousDetector: only composition
primitives (ImagePipeline, CompositeDetector, CompositeEnhance) may carry one.
Lift the detector into a CompositeDetector branch, or into the top-level
pipeline.
```

Nested in a subclass of a listed composite that replaces `_operate` (here a
hypothetical `ChainingComposite(CompositeDetector)`):

```text
a GpuDetector cannot be nested inside ChainingComposite: it subclasses
CompositeDetector but overrides _operate, so the 'parallel' child-input contract
verified for CompositeDetector does not carry over to it. Lift the detector into
a plain CompositeDetector branch, or into the top-level pipeline.
```

In a `meas`, `post`, `filters` or `model` slot of any pipeline — the root, or
(here) a nested pipeline keyed `inner` — which runs after the op chain:

```text
GpuDetector at inner/meas:MeasureSymZones/center_detector cannot be staged: it
sits in the 'meas' slot of the pipeline at inner, which runs after the op chain
-- Stage 3 runs it on a CPU node, after GPU inference has finished. Move the
detector into that pipeline's ops.
```

And more than one detector anywhere in the tree — a deferred feature, not a
limit of the design — naming every offending path so you know which branches to
split:

```text
staged execution does not support more than one GpuDetector per pipeline; found
2 at: CompositeDetector/ops[0], CompositeDetector/ops[1]
```

Each message names the way forward, because there almost always is one: run the
GPU detector as a `CompositeDetector` branch and feed its mask to the domain
detector, rather than nesting it inside.

A wrong answer here would mean Stage 2 reads a layer that was never prepared,
and the run completes with different numbers. Refusing is the cheap outcome.

### Pipeline slots, at any depth

The walker descends every pipeline's `meas`, `post`, `filters` and `model`
slots as well as its `ops` — the root pipeline's and every nested one's — and
names a slot entry with the slot as a colon namespace: `meas:<key>`,
`post:<key>`, `filters:<key>`, and `model:<ClassName>` for the single model.
A bare `meas` segment would collide with an `ops` key a user is free to
choose. The flip side is that an `ops` key may itself contain a colon, so
never classify a segment by parsing the string —
`_operation_tree.pipeline_slot_of` decides against the live pipeline.

Those slots run after the op chain. In the staged engine that is Stage 3, on a
CPU node, after Stage 2's GPU inference has finished; and a measurement may
hand its nested detector a derived input, possibly once per object, that
Stage 2 has no way to reproduce. So any path through a slot is refused.

For every shipped slot type the ancestor check above would refuse the shape
anyway — `MeasureFeatures`, `PostMeasurement`, `SetAnalyzer` and `ModelFitter`
are not composition primitives — but its message would blame the slot entry's
class rather than the slot, which is why the slot check runs first. **For one
shape the slot check is the only guard:** a class that is *both* a slot type
and a composition primitive, such as
`class MeasuringComposite(CompositeDetector, MeasureFeatures)`. It inherits
`CompositeDetector._operate`, so the ancestor check admits it, and without the
slot refusal its detector would be staged and inferred on the stored image —
the wrong input, with no error. If you write such a class, that refusal is what
keeps it honest; `test_a_slot_entry_that_is_also_a_composite_is_refused` pins
it.

### How a refusal reaches the user

The CLI checks placement right after it has parsed its options, before it
loads a manifest, clears the output directory for `--overwrite`, or exits for
`--dry-run`, and prints the message as a one-line usage error. A refused
pipeline therefore never touches `--output`, and `--dry-run` refuses exactly
what a real run would.

The GUI run console asks the same question when a pipeline is selected. A
refused pipeline shows the message in a red alert, in Local and SLURM mode
alike, and Run is disabled; Validate and Run are also refused server-side
before any run is registered. If you ever edit the probe
(`_gui/run_console/_callbacks.py:_staged_gpu_capability`), keep its
`except UnstageableGpuDetectorError` clause **above** the generic
`except (OSError, ValueError, TypeError)`: the refusal is a `ValueError`, so
the other order silently treats it as "not a GPU pipeline".

### What Stage 3 actually runs: `ReplayDetector`

Stage 3 does not call your detector. It builds a `ReplayDetector`
(`_cli_replay_detector.py`) holding your instance plus Stage 2's recorded array,
substitutes it at `gpu_path`, and applies the resulting pipeline. Two
consequences for you:

- **The write still goes through *your* `_write_object_output`.** The stub
  delegates to it, so `drop_frame_background` and `split_disconnected_labels`
  apply exactly as in a single pass. This is why overriding that hook is
  "essentially never" in the table above — a stub that assigned `objmap`
  directly would skip both.
- **The journal records *your* detector, not the stub.** `ReplayDetector`
  overrides four provenance hooks (`operation_name`, `operation_class`,
  `parameters`, and a duration offset carrying Stage 2's inference time), so a
  staged run and a single-pass run of the same pipeline produce the same
  journal. The `parameters` override is not only a parity concern: the stub
  holds the recorded array in an `NdArrayField`, and the default `model_dump`
  would serialise the whole objmap into the journal.

`--mode process --layer objmap` uses the same substitution, so a process export
runs the post-detector operation chain rather than dumping the raw array.

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
- `PHENOTYPIC_PRELOAD_MODULES` names self-registering modules: modules whose
  import attaches the class to the `phenotypic` namespace
  (`phenotypic.MyDetector = MyDetector`). A module that only defines the class
  is not enough, because pipeline JSON records bare class names. Class
  resolution imports the listed modules in every process that deserializes a
  pipeline (the CLI, SLURM workers, local parallel workers, the finalizer), so a
  detector defined outside the `phenotypic` namespace deserializes everywhere.
- A runtime path must never prompt: pass `interactive=False` to
  `require_license_acceptance` and to any download helper that can prompt.

## Declaring what your detector needs

The CLI's run preflight reads `preflight_requirements()` from every operation
the run will execute, so a missing package, gated license, or uncached weight
is reported before any image is processed. For a fixed requirement, set the
class variables `_requires_modules` (import names, checked with `find_spec` and
never imported) and `_requires_extra` (the `pyproject` extra that provides
them). When a requirement depends on a field, override the method, call
`super()`, and extend the result with `dataclasses.replace`:

```python
import dataclasses

from phenotypic.abc_ import OperationRequirements, WeightRequirement


def preflight_requirements(self) -> OperationRequirements:
    requirements = super().preflight_requirements()
    return dataclasses.replace(
        requirements,
        modules=("my_model_pkg", "torch"),
        extra="torch",
        weights=(WeightRequirement(model="my-model:base", license_key=None,
                                   is_cached=lambda: my_cache_probe()),),
    )
```

`is_cached` must answer from the file system alone: no `torch` import and no
network access, because the preflight runs on the submitting node. Return
`None` when the answer cannot be known. `GpuDetector`'s own override adds the
RGB requirement for `input_layer="rgb"`; keep it by calling `super()`.

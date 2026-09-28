(measurement-classification)=

# Measurement Classification: Phenotypes vs. Features

PhenoTypic measures many columns per colony. This page explains *how to apply*
each one — which numbers you can report directly as a biological result, and
which are best used together as inputs to classification or clustering — without
needing the underlying math.

## Two questions place every measurement

- **Interpretability** — does the number name a *biological thing* (a diameter, a
  pigment, biomass), or is it a *mathematical descriptor* (a texture value, a
  colour coordinate)?
- **Analytical role** — do you use it *as a result* (quantify an effect), or *as a
  feature* (feed many of them into a classifier/clustering)?

## Four kinds, then three tiers

Every column is first one of four **kinds**:

- **Identity / design factors** — the variables you analyse *against* (metadata,
  locators). Not outcomes.
- **Quality** — gates whether to trust a row/plate. Never a biological claim.
- **Primary measurement** — the measured signal. These get a **tier** (below).
- **Derived / model output** — computed from primary measurements; classified by
  *how* they were derived.

Primary measurements fall on a three-tier spectrum:

(measurement-tiers)=

| Tier | Name | What it is | How to apply it |
|---|---|---|---|
| **1** | Direct phenotype | A real biological quantity with units/meaning (size, intensity/opacity) | Report a single value as a result; compare across conditions; dose–response. |
| **2** | Descriptive trait | A named, interpretable form/colour property, usually unitless (shape descriptors, Lab/HSV colour, radial/zone structure) | Interpret the *direction* of change against a control; also good clustering input. |
| **3** | Discriminative feature | A mathematical fingerprint with no single biological meaning (texture, XYZ/xy/composition colour) | Don't read one value; use the whole block together for classification/clustering. |

## The trust contract

The tier is a promise about what a single number licenses:

- **Tier 1** — pre-validated for direct biological claims; safe to report alone.
- **Tier 2** — interpret directionally, anchored to a control.
- **Tier 3** — make no single-value biological claim; its job is discrimination,
  judged by how well groups separate.

## Derived outputs inherit by *how* they were made

A model fit on a primary phenotype is classified by its transformation:

- **Parameterization** (e.g. logistic/softplus growth: growth rate, lag, carrying
  capacity) → same tier as the input phenotype. Colony size and growth rate are
  interchangeable fitness proxies, so growth parameters are Tier 1.
- **Normalization** (e.g. edge correction) → the input's tier, cleaned.
- **Fit diagnostics** (R², RMSE, optimizer state, regularization knobs) → Quality.

See also: [Measurement metrics and their biological meaning](measurement_metrics_biological_meaning.md)
for per-metric detail, and the
[measurement reference](../measurements_ref/measurements/index) for the Use/Tier badge on
every column.

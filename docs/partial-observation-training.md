# Partial-observation training guide

## Reference implementation

The notebooks in `PartialObservation` generate one wideband PSD-like tensor per
example. Its shape is `[batch, 1, frequency_points, channels]`; the matching
label is `[batch, channels]` with binary occupancy values.

`standalone_cnn` has ordinary convolutional layers and predicts all channels at
one node. `decouple_cnn` shares the early convolutional layers, then uses
grouped convolutions and a per-channel output mask so channel-specific paths
can be exchanged independently. `decouple_cnn_mod` expresses the final
per-channel paths as a `ModuleList`.

For collaborative training, `class_dir` assigns a fixed subset of channels to
each SU. Each SU receives only its local slice of the training tensor and
label. Local models train with `BCEWithLogitsLoss`; their shared and
band-specific weights are exchanged during the collaboration rounds. At
evaluation, each SU's logits are placed back into the global channel positions
and combined by the notebook's decision-fusion rule. The notebooks use logits,
not thresholded probabilities, during loss calculation.

## Collected hardware data

The hardware dataset differs from the notebook data in two important ways:

1. SU files contain complex IQ frames and local detector decisions, not PSD
   tensors.
2. PU files contain timestamped activity intervals and occupancy vectors, not
   labels attached to each SU frame.

`src/training/data_preprocessor.py` handles this conversion. For every SU
frame it computes a Hann-windowed, centered FFT, converts power to dB, bins the
10 MHz capture into the configured 10 channels, and interpolates each channel
to 64 frequency points. It labels a frame by OR-ing the occupancy vectors of
all PU intervals containing that frame timestamp. No nearest-event label is
invented, and the original SU id, timestamp, and local detector decision are
retained for auditing.

Run preprocessing only with:

```bash
python src/training/data_preprocessor.py \
  dataset/fusion/JSON data/partial_observation.npz
```

The output contains `data` with shape `[N, 1, 64, 10]`, `labels` with shape
`[N, 10]`, plus `local_detections`, `su_ids`, and `timestamps`. CSV input is
also supported by pointing the command at `dataset/fusion/CSV`.

## Our training design

Our partial observation is dynamic and sample-dependent. A PU can use any
channel, and an SU can miss a nearby transmission because of distance and
received SNR. Therefore, a fixed `class_dir` is the wrong abstraction. The
training sample should instead be represented as:

```text
x[sample]       [1, 64, 10]   measured PSD features
y[sample]       [10]          global PU ground truth
o[su, sample]   [10]          observability/reliability mask
```

`y` is the target and `o` is not the SU's occupancy decision. A false local
detection can mean either an empty channel or a missed PU, so
`local_detections` must not be used directly as `o`. The mask should be
estimated from the experiment's geometry/calibration or from a training-only
reliability rule. It must be created without using the validation or test
labels. If no defensible mask is available, use `o = 1` for every channel and
describe the experiment as multi-SU learning, not partial-observation
learning.

### Minimal model change

Keep the existing CNN geometry and use `nch=10` for every SU. Do not create a
different model width for each SU and do not remap channels into local indices.
The model always returns ten logits in the global channel order.

The PSD conversion remains the input adapter; the CNN does not need to know
whether a `[64, 10]` feature matrix came from a synthetic PSD or a measured
periodogram. Normalize the measured features using statistics computed on the
training split only. Initially keep one input channel. If the model needs to
know which columns are unreliable, the smallest architectural extension is to
concatenate `o` expanded from `[10]` to `[64, 10]` and change the first
convolution from `Conv2d(1, ...)` to `Conv2d(2, ...)`. This should be an
experiment, not the default: zeroing an unobserved band without an explicit
mask can be confused with a genuinely quiet band.

### Local objective

Each SU trains on its own frames, but contributes loss only for observable
channels:

```text
loss_s = sum(o[s,c] * BCEWithLogitsLoss(logit[s,c], y[s,c]))
         / max(sum(o[s,c]), 1)
```

The model still predicts all ten channels. Masking the loss prevents a missed
or unavailable observation from being treated as a trustworthy negative label.
The global PU labels remain the supervision; SU detector outputs are not
replacements for them.

### Weight sharing and aggregation

Use one local `decouple_cnn` (or `decouple_cnn_mod`) instance per SU, with all
instances configured identically as `nch=10`:

1. Aggregate the early/shared layers with ordinary sample-weighted FedAvg.
2. Aggregate each channel-specific grouped-convolution and output-head slice
   using only the number of valid observations for that channel:
   `weight[c] = sum(o[:, c])`.
3. Leave a channel-specific slice unchanged in a round when its valid count is
   zero.
4. Evaluate by producing ten global logits per SU and applying the same
   channel-wise masked metrics. A majority-vote fusion is optional and should
   be reported separately from the neural model's per-channel results.

This preserves the paper's shared-versus-band-specific idea without assuming
that a band permanently belongs to one SU. The random PU channel assignment
is handled by `y`; it does not require changing the convolutional channel
layout.

### What changes in the implementation

The next training implementation should add only these pieces around the
existing model:

1. A dataset loader for the `.npz` arrays, with a train/validation/test split
   performed by time or event, not by randomly duplicating adjacent frames.
2. A reliable `observation_mask` field and a masked BCE loss.
3. A local-training loop that returns model weights and per-channel valid
   counts.
4. A merger that applies ordinary FedAvg to shared parameters and weighted
   aggregation to channel-specific parameters.

No change to the PSD tensor dimensions or the core convolutional blocks is
needed for the first experiment. This is the smallest implementation that is
directly analogous to the paper while matching our moving PU bands and
distance-dependent SU observations.
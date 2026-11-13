# MATLAB hardware-equivalent experiment

[`main.m`](main.m) reads the simulation-local [`experiment-config.yaml`](experiment-config.yaml)
and writes the same JSON dataset shape as the hardware fusion pipeline:

- `su-001.jsonl` through `su-003.jsonl` contain one frame object per line with `timestamp`,
  `frame_id`, 1024 `[I,Q]` samples, global-truth `channels`, local-detector
  `local_channels`, and detector `snr_db`.
- `pu-001.jsonl` through `pu-003.jsonl` contain one occupation event per line with
  `timestamp`, `channel_occupation`, `channel_indices`, and `duration_s`.

Run from MATLAB with:

```matlab
main()
```

Optional arguments are `configPath`, `outputPath`, `durationSeconds`,
`randomSeed`, and `frameRateHz`. The simulation YAML defines three SUs and
three PUs explicitly.
The SU receiver gains and PU transmitter gains are staggered so different SUs
observe the same PU activity at different strengths, producing partial
observation rather than identical local views.
The `channels` field is retained only as global ground truth. Local SU
observations are represented by `local_channels`, which uses the same
maximum-FFT-bin energy thresholding rule as the hardware receiver.

The default duration is 12 hours (`43200` seconds), and the default persistence
rate is 9.18 frames per second, measured from the collected SU dataset. Each
frame contains exactly 1024 samples, matching the hardware receiver buffer.
Override the duration or persistence rate when needed, for example
`main("", "", 60, 7, 9.18)` for a one-minute run. Frames are streamed directly
to JSONL files instead of accumulating the entire experiment in memory. IQ
values are quantized to signed 16-bit integers, matching the compact numeric
shape of the recorded SDR data.

The fixed positions in `create_layout.m` are in metres and are intentionally
separate from the paper's 10-node illustration. Propagation uses only the
paper-model concepts in [path-loss.md](path-loss.md): distance-dependent
power loss, log-normal shadowing, and the configured PU-to-SU geometry.
Shadowing is sampled once per link, as it would be for a fixed experiment,
rather than changing every frame.

## PU waveform and occupancy

`generate_pu_signal.m` directly reproduces the checked-in
`primary-user.py` waveform: one complex exponential per occupied channel,
summed and scaled by `0.5 * 2^14 / number_of_active_channels`. Those complex
IQ samples are the samples passed to `adi.Pluto.tx()`; no modulation toolbox
or implicit Pluto modulation is used.

Each PU schedule independently chooses zero, one, or two channels for an
occupation event. Zero channels is an idle event. Therefore, a PU never
occupies more than two channels at a time. The active channel frequencies are
the configured channel offsets in `create_channel_plan.m`, and the generated
PU event JSON and SU IQ frames use the same schedule.

To inspect PU content in an SU frame:

```matlab
inspect_dataset("src/simulation-experiment/data", "su-001", 1, 10e6)
```

The `channels` field in the title shows the ground-truth occupied channels,
and the FFT displays the propagated PU tones.

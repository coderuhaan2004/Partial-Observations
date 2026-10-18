# MATLAB hardware-equivalent experiment

[`main.m`](main.m) reads the repository-level [`experiment-config.yaml`](../../experiment-config.yaml)
and writes the same JSON dataset shape as the hardware fusion pipeline:

- `su-001.json` through `su-003.json` contain frame objects with `timestamp`,
  `frame_id`, 1024 `[I,Q]` samples, `channels`, and `snr_db`.
- `pu-001.json` through `pu-003.json` contain occupation events with
  `timestamp`, `channel_occupation`, `channel_indices`, and `duration_s`.

Run from MATLAB with:

```matlab
main()
```

Optional arguments are `configPath`, `outputPath`, `durationSeconds`, and
`randomSeed`. The current YAML contains one SU and two PU entries; the runner
keeps those settings and deterministically expands missing logical radios to
three SUs and three PUs. Add real device entries to the YAML when the hardware
experiment has distinct settings for all six radios.

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

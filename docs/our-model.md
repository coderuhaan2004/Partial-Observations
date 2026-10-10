## What features can we extract from our collected dataset
1. Power Spectrum Information using PSD (Power Spectrum Density) data from IQ samples.
2. A hint of Spatial information of which SU is closer to which PU (which we already have)

- Question: How can we get accurate predictions of channel prediction when the SU is not able to sense the channel occupation of far off PUs? How will SUs communicate and collaborate with each other for complete accurate sensing?

- What about the following architecture:
    -  **Sparse Bayesian Learning combined with Deep Networks**: Bayesian inference (uncertainty quantification and resilience to partial data characteristic) with Deep learning based feature extraction capabilities
    - But it should definitely need some sort of collaboration among the SUs for correct spectrum sensing.

- The collaboration seems a bit of blackbox here. I want to exactly know how collaborative learning would help predicting accurate wideband spectrum.

## Refer [Collaboration Idea guide](collaboration-ideas.md)
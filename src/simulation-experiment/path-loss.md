# Signal Generation and Path-Loss Model in *Collaborative Learning-Based Spectrum Sensing Under Partial Observations*

**Paper:** Weishan Zhang, Yue Wang, Xiang Chen, Lingjia Liu, and Zhi Tian, “Collaborative Learning-Based Spectrum Sensing Under Partial Observations,” *IEEE Transactions on Cognitive Communications and Networking*, vol. 10, no. 5, October 2024, pp. 1843–1855. DOI: [10.1109/TCCN.2024.3391320](https://doi.org/10.1109/TCCN.2024.3391320).

This note focuses on the paper's signal/PSD data-generation method, propagation model, and what those choices mean for reproducing the experiment. The main details are in Section III-A (signal model, Eqs. 1–3, pp. 1845–1846) and Section V-A–V-B (data generation and simulation settings, pp. 1849–1850).

## 1. Simulation setup at a glance

The authors generate **synthetic wideband PSD observations**, rather than describing a complete SDR waveform-capture pipeline.

| Parameter | Paper setting |
|---|---|
| Number of spectrum bands, `N_f` | 20 |
| PSD frequency points per band, `N_w` | 64 |
| PSD input shape per SU observation | `64 × 20` |
| Number of primary users (PUs) | 10 |
| Number of secondary users (SUs) | 10 |
| Reference distance, `d_0` | 1 m |
| Path-loss exponent, `α` | 3.71 |
| Path-gain constant, `β` | `10^-3.154` |
| Log-normal shadow-fading variable | Gaussian with mean 0 and standard deviation `3.65 dB` |
| SNR range reported for sensing evaluation | `−16 dB` to `−2 dB` |

The locations of the 10 PUs and 10 SUs are illustrated in Fig. 5 (p. 1849). The paper uses the path loss and shadowing model to make a band's signal strong at some SUs and weak at others. This is how some SUs obtain useful observations of only part of the overall band pool.

## 2. How the PUs' signals and occupancy labels are generated

### 2.1 Band occupancy

The full spectrum is divided uniformly into `N_f = 20` bands. The occupancy state is represented by a binary vector:

\[
\mathbf y=[y_1,y_2,\ldots,y_{N_f}],\qquad y_n\in\{0,1\},
\]

where `y_n = 1` means band `n` is occupied and `y_n = 0` means it is vacant. Each band is assigned to a PU. A PU may be assigned one band or several **non-contiguous aggregated bands**; the bands assigned to one PU share that PU's activity state. Thus, occupancy across different bands is correlated when those bands belong to the same PU.

For the sufficient-training-data experiment, the paper describes `2^N_PU = 2^10 = 1024` possible PU occupancy patterns. It also reports generating 20 local training samples per occupancy pattern per SU in Case 1 (p. 1850).

### 2.2 Modulated signal for an occupied band

When band `n` is occupied, the authors generate its original PU signal `x_n` by **modulating a random message sequence using a predefined modulation scheme** for that PU. Different modulation schemes produce different within-band PSD shapes. The paper does not specify the full PHY implementation details (such as symbol rate, pulse-shaping filter, oversampling factor, frame length, or exact bit-to-symbol mapping) in its data-generation description.

Table I (p. 1849) assigns these bands and modulation schemes:

| PU index | Assigned band indices | Modulation scheme |
|---:|---|---|
| 0 | 0 | BPSK |
| 1 | 1, 10 | MSK |
| 2 | 2, 11, 14 | 2FSK |
| 3 | 3 | 16PSK |
| 4 | 4, 19 | 4FSK |
| 5 | 5, 13 | 16QAM |
| 6 | 6, 15 | BPSK |
| 7 | 7, 12, 18 | MSK |
| 8 | 8, 16 | 2FSK |
| 9 | 9 | 16PSK |

The table uses zero-based band indices. In the equations, the band index is written from `1` to `N_f`.

**Important:** the paper's `64` is the number of PSD frequency points measured **within each band**. It does **not** mean the PU randomly activates 64 OFDM subcarriers, nor does the paper specify an OFDM waveform. BPSK, MSK, FSK, PSK, and QAM are the modulation schemes explicitly listed.

## 3. Received signal model

For SU `j`, the paper models the received time-domain signal as a sum of the signals arriving from all bands, with propagation gain and additive noise:

\[
 x^j=\sum_{n=1}^{N_f} y_n h_n^j x_n+w^j.
\tag{1}
\]

Here:

- `x_n` is the modulated PU signal associated with band `n`;
- `y_n` is its ground-truth occupancy bit;
- `h_n^j` is the channel-gain coefficient between the PU assigned to band `n` and SU `j`;
- `w^j` is the noise vector at SU `j`.

This equation expresses the physical idea: an unoccupied band contributes no PU signal (`y_n = 0`); an occupied band contributes a signal scaled by the PU-to-SU channel gain.

## 4. The path-loss and shadowing model

### 4.1 Formula used in the paper

The channel-gain magnitude is defined as

\[
\boxed{
 h_n^j=\sqrt{\beta\left(\frac{d_0}{d_n^j}\right)^\alpha
 10^{-\psi_n^j/10}}
}
\]

where:

- `d_n^j` is the distance between SU `j` and the PU assigned to band `n`;
- `d_0 = 1 m` is the reference distance;
- `α = 3.71` is the path-loss exponent;
- `β = 10^-3.154` is the constant associated with antenna characteristics and average attenuation;
- `ψ_n^j` is the shadow-fading random variable, Gaussian distributed with mean zero and standard deviation `σ_ψ = 3.65 dB`.

Equivalently, the **power gain** applied to the PSD is

\[
\boxed{|h_n^j|^2=\beta\left(\frac{d_0}{d_n^j}\right)^\alpha
 10^{-\psi_n^j/10}}
\]

This is a distance-dependent path-loss model with log-normal shadowing. The distance term causes average received power to decrease approximately as `d^-α`; the random shadowing term varies the gain around that distance-dependent average.

### 4.2 What the parameters do

- **Distance (`d_n^j`):** increasing the separation between a PU and an SU reduces the received signal power.
- **Path-loss exponent (`α = 3.71`):** controls how quickly the power falls with distance. A larger exponent means faster decay.
- **Constant (`β = 10^-3.154`):** sets the reference gain scale. With `d_n^j = d_0` and `ψ_n^j = 0`, the model's power gain is `β`.
- **Shadow fading (`ψ_n^j`):** adds a random attenuation variation in dB. The paper uses zero mean and `3.65 dB` standard deviation.

Ignoring shadow-fading variation, moving from `1 m` to `5 m` changes the path-gain factor by

\[
10\alpha\log_{10}(5)\approx 25.9\text{ dB}
\]

of additional attenuation. This illustrates why the same PU can be readily observable at a nearby SU but effectively hidden beneath the noise at a more distant SU.

The paper's stated model includes path loss and log-normal shadowing; it does not spell out a separate small-scale multipath fading model in this formula.

## 5. How the authors create each SU's PSD measurement

The authors form a synthetic PSD vector for each individual band, then assemble all bands into the SU's wideband observation.

### 5.1 Model spectral leakage across adjacent bands

For each original modulated signal, they form an “overflowed” PSD vector with `3N_w` points:

\[
\tilde{s}_n=[s'_{n,L};\;s'_{n,M};\;s'_{n,R}]\in\mathbb R^{3N_w}.
\tag{9}
\]

Each component has `N_w = 64` values:

- `s'_{n,M}`: the PSD main lobe in the signal's own band;
- `s'_{n,L}`: the left sidelobe that can leak into the preceding band;
- `s'_{n,R}`: the right sidelobe that can leak into the following band.

This is how the simulation models **power leakage between contiguous bands**. A PU transmitting in one band can affect PSD measurements in its neighboring bands even when those neighbors are not themselves occupied.

### 5.2 Build the received PSD for band `n` at SU `j`

The paper gives the following PSD construction (Eq. 10):

\[
\boxed{
\begin{aligned}
s_n^j={}&y_n|h_n^j|^2s'_{n,M}\\
&+y_{n-1}|h_{n-1}^j|^2s'_{n-1,R}\\
&+y_{n+1}|h_{n+1}^j|^2s'_{n+1,L}+w_n.
\end{aligned}}
\tag{10}
\]

In words, the PSD observed in band `n` contains:

1. The main-lobe PSD from its own PU, scaled by that PU-to-SU power gain, if the band is occupied.
2. The right-sidelobe leakage from the preceding band, scaled by the preceding band's occupancy and channel gain.
3. The left-sidelobe leakage from the following band, similarly scaled.
4. An additive AWGN noise vector `w_n` for that band.

At the spectrum boundaries, an implementation must decide how to handle the nonexistent outside neighbor (for example, treat its contribution as zero); that boundary convention is not spelled out in the displayed equation.

The authors repeat this operation for every band and stack the results to form

\[
S^j\in\mathbb R^{N_w\times N_f}=\mathbb R^{64\times20},
\]

which is one PSD input sample for SU `j`. In the signal-model section, they also describe obtaining the wideband PSD by taking the Fourier transform of the received signal's autocorrelation, i.e. `s^j = FT(Corr(x^j))` (Eq. 2), and arranging it into the `N_w × N_f` matrix (Eq. 3).

## 6. How path loss creates “partial observations”

The key mechanism is **not that every SU is assigned an arbitrary subset of frequencies by the transmitter**. Instead, the SUs experience different path gains because the PUs have different locations relative to them.

For an occupied band, the useful main-lobe component is proportional to

\[
y_n|h_n^j|^2s'_{n,M}.
\]

When the PU is far from SU `j`, `|h_n^j|^2` becomes small. The authors explain that the received main-lobe PSD can become much smaller than the noise contribution, so the PSD distributions for occupied and unoccupied cases become nearly indistinguishable. That band is then effectively **unobservable** to that SU, even though the band is physically occupied.

The paper therefore obtains heterogeneous local observations from:

1. PU activity and the per-PU grouping of bands;
2. modulation-dependent PSD shapes;
3. leakage from adjacent bands;
4. distance-dependent path loss and log-normal shadowing;
5. additive AWGN.

The authors also define which bands are observable to each SU based on the placement of nearby PUs (Table II, p. 1850).

## 7. What is and is not specified for reproduction

### Explicitly specified by the paper

- 20 bands and 64 PSD points per band;
- 10 PUs and 10 SUs, with locations illustrated in Fig. 5;
- the per-PU band assignments and modulation types in Table I;
- a random message sequence modulated using each PU's predefined scheme;
- the received-signal model and path-loss/shadowing formula;
- three-part PSD shape per band (left sidelobe, main lobe, right sidelobe);
- the adjacent-band PSD mixing rule and AWGN term;
- `α = 3.71`, `β = 10^-3.154`, `d_0 = 1 m`, shadowing standard deviation `3.65 dB`;
- 1024 possible PU activity patterns and a `64 × 20` PSD input matrix;
- evaluation SNRs in the range `−16 dB` to `−2 dB`.

### Not fully specified in the paper's data-generation description

- Exact time-domain signal-generation implementation and symbol-rate/oversampling settings;
- pulse-shaping and filtering details used to produce the modulation-specific PSDs;
- exact PSD normalization and the numeric method used to construct each main-lobe/sidelobe template;
- whether each shadow-fading realization is held fixed for a PU–SU link or redrawn for every generated sample;
- exact numeric coordinates of all users in prose (their relative positions are drawn in Fig. 5);
- the exact implementation details for constructing the AWGN vector at every SNR.

The paper points to the authors' implementation repository for further implementation details: <https://github.com/FrancisZWS/PartialObservation>.

## 8. Implications for a PlutoSDR implementation

The paper's simulator is a **PSD-domain synthetic-data generator**. A PlutoSDR experiment will instead transmit and receive actual IQ waveforms, so matching the paper is not simply a matter of setting `tx_hardwaregain` and `rx_hardwaregain`.

A practical adaptation would be:

1. Generate a PU waveform using one of the paper's modulation types, and retain its known band-level activity label.
2. Transmit through the RF environment and capture IQ at each SU.
3. Estimate PSD from each SU's IQ samples and map the spectrum into 64 PSD points per logical band.
4. Preserve the PU ground-truth labels separately from the SU's detected occupancy.
5. Measure how received PSD changes with distance and environment. If the physical setup cannot reproduce the paper's path-gain spread, use controlled RF attenuation for a repeatable experiment; RX hardware gain should remain a receiver-setting variable, not a substitute for the paper's propagation model.
6. For a simulation that must match the paper's mathematical model closely, apply the formula above to PSD power and add its modeled leakage and noise terms. Keep clear whether results are real RF measurements or synthetic PSD samples.

**Do not interpret the paper's 64 PSD points as 64 randomly activated OFDM subcarriers.** The 64 values are the sensing representation; the authors describe modulated PU signals and compute/construct their PSDs. Per-bin random PU occupancy would be an extension to the paper, not a feature explicitly specified by this simulation.
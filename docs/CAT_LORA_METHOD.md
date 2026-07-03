# CAT-LoRA Method Specification

## Research Target

We study whether audio-conditioned video generators use audio as a temporally
faithful control signal. The adaptation target is not generic video quality and
not lip synchronization. The target property is:

> When the temporal structure of the input audio changes, the temporal dynamics
> of the generated video should change correspondingly, while prompt semantics
> and visual appearance remain stable.

## Backbone

The representative backbone is MusicInfuser. It is the most appropriate target
because it is a recent audio-conditioned video diffusion model built by adapting
a pretrained video diffusion prior with audio modules. This isolates our
question: not whether audio can be injected, but whether injected audio exerts
faithful temporal control.

We freeze the pretrained video generator, VAE, text encoder, and audio encoder.
We train only parameter-efficient audio-temporal modules: existing MusicInfuser
audio adapters and LoRA weights on selected attention projections.

## Data

Main training uses AIST++/AIST music-dance clips with held-out music-track
splits. The split must avoid train/test music overlap because the desired
generalization is to unseen audio temporal structure.

Training clips are short, around 2.5 seconds, following MusicInfuser. Longer
videos are reserved for evaluation.

Counterfactual audio is generated on the fly:

- local silence around strong onsets
- zero-padded temporal shifts
- pitch-preserving tempo changes
- rare global silence

Mismatched audio is reserved for evaluation because it does not imply a unique
target video response.

## Audio Temporal Control

For audio `a`, compute a frame-rate-aligned control curve:

```text
C_a(t) = norm(0.4 RMS(t) + 0.4 onset(t) + 0.2 spectral_flux(t))
```

Beat activation is used for evaluation but not as the primary training signal,
because beat tracking is unstable on short clips.

## Visual Temporal Response

During training, compute motion in predicted clean latent space rather than RGB:

```text
M(t) = mean(|z_hat0[t] - z_hat0[t-1]|)
```

This is differentiable and avoids decoding every training step. During
evaluation, RGB frame difference and optical flow are used.

## Objective

The implementation uses detailed losses:

```text
L = L_diff
  + lambda_align L_align
  + lambda_peak L_peak
  + lambda_delta L_delta
  + lambda_sil L_sil
  + lambda_preserve L_preserve
  + lambda_smooth L_smooth
```

The paper can present the grouped objective:

```text
L = L_diff + lambda_AT L_AT + lambda_CF L_CF + lambda_P L_P
```

### Diffusion Reconstruction

Only the original-audio branch receives the standard diffusion reconstruction
loss. Counterfactual audio has no ground-truth video.

### Audio-Motion Alignment

Use Pearson correlation between audio control and latent motion:

```text
L_align = 1 - corr(C_a, M)
```

Use peak distribution KL to align event peaks:

```text
L_peak = KL(softmax(C_a/tau) || softmax(M/tau))
```

### Counterfactual Delta Response

The central loss compares how audio changes with how video dynamics change:

```text
Delta C = C_cf - C_orig
Delta M = M_cf - M_orig
L_delta = 1 - corr(Delta C, Delta M)
```

This is the key distinction from ordinary paired reconstruction.

### Silence Suppression

For local/global silence:

```text
L_sil = mean(max(0, M_cf[silent] - rho M_orig[silent]))
```

### Prior Preservation

Use the frozen MusicInfuser backbone as a teacher. Preserve temporal mean
latents:

```text
L_preserve = |mean_t(z_student) - stopgrad(mean_t(z_teacher))|
```

Temporal mean approximates appearance, scene, and identity, while temporal
residuals capture motion.

## Evaluation

All counterfactual evaluations hold prompt, seed, and reference fixed. Only the
audio changes.

Metrics:

- audio-motion correlation
- peak alignment / peak F1
- silence suppression
- shift lag response
- tempo response
- prompt dominance ratio

Main comparison:

- MusicInfuser
- MusicInfuser + CAT-LoRA

Diagnostic-only models:

- TempoTokens
- MM-Diffusion
- Wan-S2V
- TalkVerse
- text-only controls when applicable

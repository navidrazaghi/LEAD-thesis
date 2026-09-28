# Fourier amplitude augmentation: stopped at stage 0

The idea was to augment each camera frame by perturbing its Fourier amplitude
while keeping its phase (FACT / APR style, luminance only, per camera), on the
premise that phase carries the scene's geometry and amplitude its appearance, so
the waypoint label stays valid; a consistency loss would then make the planner
amplitude-invariant. Stage 0 asked, before any training, whether the trained
policies are sensitive to amplitude at all. They mostly are not, and the idea was
stopped on 2026-09-28.

`../scripts/server_home/fourier_diagnostic.py` drives two checkpoints over the
same 320 frames (40 batches of 8, seed 0) under each edit and measures how far
the predicted waypoints move (mean Euclidean displacement over the future
points). "Share" is that move as a fraction of replacing the whole camera image
with another frame's. Raw numbers: `fourier_diagnostic.csv`.

| Edit | mean \|d pixel\| (0-255) | seed 2 (585 logs) | LEAD reference (1.1 TB) |
| :--- | ---: | ---: | ---: |
| amplitude halfway to another frame's, phase kept | 38.1 | 0.065 m (11%) | 0.114 m (6%) |
| amplitude replaced, phase kept | 75.4 | 0.150 m (25%) | 0.293 m (15%) |
| phase replaced, amplitude kept | 34.3 | 0.290 m (48%) | 1.179 m (60%) |
| whole image replaced | 83.6 | 0.609 m (100%) | 1.966 m (100%) |
| no camera signal | 104.8 | 0.260 m (43%) | 1.186 m (60%) |
| thesis camera degradation 0.5 | 54.4 | 0.082 m (13%) | 0.333 m (17%) |
| thesis camera degradation 1.0 | 90.9 | 0.156 m (26%) | 1.105 m (56%) |

Swapping the amplitude changes the pixels almost as much as swapping the whole
image, but moves the plan a quarter as far or less; swapping the phase changes
the pixels less than half as much and moves it two to four times further. Both
models already read geometry from the phase, so the augmentation could at best
push the 15-25% toward zero -- judged not worth the GPU time.

Limits: the frames are in-sample (only the 585 training logs are on the machine);
plan displacement is not a driving score; the amplitude swap also moves the mean
brightness. The spectral edit itself was checked on synthetic frames: amplitude
matches the donor and phase the original to float32 precision per camera, and
chroma is preserved exactly.

# Result tables

MSE: normalised amplitude vs. ground truth on the central region (groups 7-9).
Finest element: finest USAF element in groups 7-9 such that it and every coarser one have gap-to-bar Michelson contrast >= 0.1 in both orientations.

### Realistic simulation (spherical illumination, background, noise)

Seeds: usaf, usaf_s2, usaf_s3

| Configuration | Frames | MSE (mean ± sd) | Finest element per seed | Contrast 8-5 (mean) | Contrast 8-6 (mean) |
|---|---|---|---|---|---|
| backprop_single | 1 | 0.0378 ± 0.0006 | none, none, none | -0.035 | -0.013 |
| grid_h1_a1 | 1 | 0.0448 ± 0.0007 | none, none, none | 0.031 | -0.000 |
| grid_h1_a2 | 2 | 0.0340 ± 0.0034 | 7-2, none, 8-1 | -0.055 | -0.064 |
| grid_h1_a3 | 3 | 0.0345 ± 0.0041 | 7-6, 7-2, none | -0.089 | -0.080 |
| grid_h2_a1 | 2 | 0.0160 ± 0.0009 | 8-6, 8-6, 8-6 | 0.200 | 0.141 |
| grid_h2_a2 | 4 | 0.0144 ± 0.0015 | 8-3, 8-3, 8-1 | 0.032 | -0.001 |
| grid_h2_a3 | 6 | 0.0140 ± 0.0016 | 8-3, 8-2, 8-1 | 0.002 | -0.011 |
| grid_h3_a1 | 3 | 0.0091 ± 0.0011 | 9-1, 9-1, 9-1 | 0.323 | 0.255 |
| grid_h3_a2 | 6 | 0.0111 ± 0.0014 | 8-6, 8-4, 8-5 | 0.158 | 0.077 |
| grid_h3_a3 | 9 | 0.0115 ± 0.0013 | 8-4, 8-2, 8-1 | 0.014 | 0.005 |
| LISA_1h9a | 9 | 0.0204 ± 0.0007 | 9-2, 9-2, 9-2 | 0.427 | 0.400 |
| MFAP_9h1a | 9 | 0.0056 ± 0.0010 | 9-2, 9-2, 9-2 | 0.664 | 0.549 |
| HSSA_3h3a | 9 | 0.0115 ± 0.0013 | 8-4, 8-2, 8-1 | 0.014 | 0.005 |
| abl_HSSA_no_p_update | 9 | 0.0115 ± 0.0013 | 8-4, 8-2, 8-1 | 0.014 | 0.007 |
| abl_HSSA_literal_update | 9 | 0.0128 ± 0.0009 | 8-4, 8-3, 8-1 | -0.063 | -0.089 |
| abl_HSSA_true_geometry | 9 | 0.0160 ± 0.0019 | 8-5, 8-3, 7-3 | 0.111 | 0.009 |
| abl_HSSA_shift_instead_of_tilt | 9 | 0.0056 ± 0.0007 | 9-2, 9-2, 9-2 | 0.594 | 0.519 |
| abl_MFAP_literal_update | 9 | 0.0102 ± 0.0011 | 8-4, 8-4, 8-5 | 0.069 | -0.026 |
| abl_LISA_registered_no_angle | 9 | 0.0352 ± 0.0028 | none, 7-6, none | -0.184 | -0.160 |
| ext_HSSA_tilt_from_shift | 9 | 0.0062 ± 0.0008 | 9-2, 9-2, 9-2 | 0.543 | 0.466 |
| ext_grid_h3_a2_tilt_from_shift | 6 | 0.0064 ± 0.0010 | 9-2, 9-2, 9-1 | 0.525 | 0.418 |
| ext_grid_h2_a3_tilt_from_shift | 6 | 0.0077 ± 0.0012 | 9-2, 9-2, 9-2 | 0.451 | 0.390 |

### Idealised simulation (plane waves, no background; noise kept)

Seeds: usaf_ideal

| Configuration | Frames | MSE (mean ± sd) | Finest element per seed | Contrast 8-5 (mean) | Contrast 8-6 (mean) |
|---|---|---|---|---|---|
| backprop_single | 1 | 0.0368 | none | -0.035 | -0.011 |
| grid_h1_a1 | 1 | 0.0439 | none | 0.041 | 0.011 |
| grid_h1_a2 | 2 | 0.0340 | none | 0.029 | -0.025 |
| grid_h1_a3 | 3 | 0.0335 | 7-2 | 0.070 | -0.014 |
| grid_h2_a1 | 2 | 0.0138 | 8-6 | 0.195 | 0.120 |
| grid_h2_a2 | 4 | 0.0131 | 8-3 | 0.069 | 0.067 |
| grid_h2_a3 | 6 | 0.0126 | 8-4 | 0.064 | 0.056 |
| grid_h3_a1 | 3 | 0.0067 | 9-1 | 0.323 | 0.259 |
| grid_h3_a2 | 6 | 0.0095 | 7-3 | 0.111 | 0.149 |
| grid_h3_a3 | 9 | 0.0093 | 8-4 | 0.080 | 0.073 |
| LISA_1h9a | 9 | 0.0184 | none | 0.431 | 0.446 |
| MFAP_9h1a | 9 | 0.0029 | 9-2 | 0.780 | 0.662 |
| HSSA_3h3a | 9 | 0.0093 | 8-4 | 0.080 | 0.073 |
| abl_HSSA_no_p_update | 9 | 0.0093 | 8-4 | 0.080 | 0.074 |
| abl_HSSA_literal_update | 9 | 0.0111 | 8-4 | 0.050 | -0.047 |
| abl_HSSA_true_geometry | 9 | 0.0128 | 8-4 | 0.097 | 0.042 |
| abl_HSSA_shift_instead_of_tilt | 9 | 0.0035 | 9-2 | 0.599 | 0.536 |
| abl_MFAP_literal_update | 9 | 0.0079 | 8-4 | 0.036 | -0.035 |
| abl_LISA_registered_no_angle | 9 | 0.0359 | none | -0.007 | -0.082 |
| ext_HSSA_tilt_from_shift | 9 | 0.0029 | 9-2 | 0.714 | 0.602 |
| ext_grid_h3_a2_tilt_from_shift | 6 | 0.0034 | 9-1 | 0.519 | 0.450 |
| ext_grid_h2_a3_tilt_from_shift | 6 | 0.0045 | 9-2 | 0.566 | 0.497 |

### Phase objects (0 to pi rad)

| Object / method | Phase RMSE (rad) | NMSE | Detail RMSE, < 10 um (rad) |
|---|---|---|---|
| camera/LISA_1h9a | 0.8986 | 0.08182 | 0.1963 |
| camera/MFAP_9h1a | 0.8523 | 0.07359 | 0.1736 |
| camera/HSSA_3h3a | 0.8568 | 0.07438 | 0.1789 |
| camera/HSSA_tilt_kernel | 0.8460 | 0.07252 | 0.1723 |
| astronaut/LISA_1h9a | 0.8986 | 0.08181 | 0.2781 |
| astronaut/MFAP_9h1a | 0.8057 | 0.06578 | 0.2151 |
| astronaut/HSSA_3h3a | 0.8145 | 0.06721 | 0.2302 |
| astronaut/HSSA_tilt_kernel | 0.7968 | 0.06432 | 0.2059 |

### Angle sweep (3 heights x 3 angles, realistic)

| Max angle | Method | MSE seed1 | MSE seed2 | MSE seed3 | Mean | Finest per seed |
|---|---|---|---|---|---|---|
| 2 deg | HSSA | 0.0079 | 0.0140 | 0.0079 | 0.0099 | 9-1, 8-6, 8-5 |
| 2 deg | HSSA_tilt_from_shift | 0.0063 | 0.0075 | 0.0052 | 0.0063 | 9-2, 9-2, 9-2 |
| 4 deg | HSSA | 0.0091 | 0.0171 | 0.0125 | 0.0129 | 9-1, 8-4, 8-3 |
| 4 deg | HSSA_tilt_from_shift | 0.0061 | 0.0080 | 0.0068 | 0.0070 | 9-2, 9-2, 9-2 |
| 8 deg | HSSA | 0.0105 | 0.0222 | 0.0108 | 0.0145 | 8-4, 8-2, 8-1 |
| 8 deg | HSSA_tilt_from_shift | 0.0058 | 0.0105 | 0.0058 | 0.0074 | 9-2, 9-2, 9-2 |

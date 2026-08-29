# Radio Map Reconstruction

This context describes the observations and evaluation concepts used to reconstruct dense channel maps from sparse measurements.

## Language

**Valid Receiving Area**:
The pixels where receiver measurements may exist, excluding building and transmitter pixels.
_Avoid_: Acceptable area, valid area

**Valid Sampling Point**:
A distinct observed pixel located within the valid receiving area. A sample count always counts valid sampling points.
_Avoid_: Sampling pixel, measurement pixel

**Sampling Mask**:
The binary map identifying valid sampling points for one channel map under one sampling condition.
_Avoid_: Point map

**Fixed Sampling Mask**:
A sampling mask that is reused for the same channel map and sample count across repeated evaluations. Masks for different sample counts are independently sampled and need not contain one another.
_Avoid_: Nested sampling mask

**Gradient-Distance Weighted Clustering Sampling Strategy**:
A deterministic sampling strategy that selects exactly the requested number of valid sampling points by balancing predicted channel-map variation, transmitter distance, and spatial coverage.
_Avoid_: Gradient-guided probabilistic sampler, Sampling model

**Evaluation Bundle**:
The ground truth, sampling mask, sparse channel map, reconstructed channel map, and absolute error retained together for later visualization.
_Avoid_: Result image, visualization output

**Evaluation Bundle Figure**:
A four-panel report figure derived from one evaluation bundle, showing the ground truth, sparse channel map, reconstructed channel map, and absolute error.
_Avoid_: Evaluation bundle, joint image

**Mean Per-Sample Normalized RMSE**:
For each channel-map sample, normalized RMSE is computed over that sample's valid receiving area; the reported metric is the arithmetic mean of those per-sample RMSE values.
_Avoid_: Global pixel RMSE, physical-unit RMSE

**City-Map Split**:
A dataset partition in which all transmitter samples sharing one city and building layout remain in the same train, validation, or test subset.
_Avoid_: Per-transmitter split, per-sample split

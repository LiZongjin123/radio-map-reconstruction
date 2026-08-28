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

**Sampling Score Map**:
An unconstrained, single-channel score map produced from the building map and transmitter map. Scores define preference only within the valid receiving area and do not depend on the requested sample count.
_Avoid_: Probability map, channel map

**Ranked Sampling Policy**:
A one-shot policy that orders the valid receiving area by a K-independent sampling score map. Selecting prefixes of this order for increasing sample counts produces nested learned sampling masks.
_Avoid_: Adaptive sampling policy, K-conditioned policy

**Learned Sampling Mask**:
A binary sampling mask containing exactly K valid sampling points chosen by a ranked sampling policy. Learned sampling masks for increasing K are nested and are distinct from independently drawn random fixed sampling masks.
_Avoid_: Fixed sampling mask, random sampling mask

**Sampling Decision Figure**:
A two-panel report figure showing a sampling score map on the valid receiving area and the corresponding learned sampling mask together with the transmitter and building map.
_Avoid_: Evaluation bundle figure

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

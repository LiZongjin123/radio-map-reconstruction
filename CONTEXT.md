# Radio Map Reconstruction

This context defines the language for reconstructing dense path-gain fields from sparse receiver observations in a known urban scene.

## Propagation Environment

**Scene**:
An urban propagation environment defined by its spatial extent and building footprints. A Scene may be paired with multiple Transmitters.
_Avoid_: Map, city map, environment map

**Building Footprint**:
The part of a Scene occupied by buildings and therefore excluded from outdoor reconstruction.
_Avoid_: Building map, building image

**Transmitter**:
The radio source associated with a Radio Map, identified by its location within a Scene.
_Avoid_: Transmitter map, antenna image

**Receiver Cell**:
A spatial location at which Path Gain is defined or observed.
_Avoid_: Pixel, point

**Outdoor Cell**:
A Receiver Cell outside every Building Footprint.
_Avoid_: Valid pixel, free-space pixel

## Radio Maps

**Path Gain**:
The ratio of received power to transmitted power expressed in dB; a larger, less-negative value represents a stronger link.
_Avoid_: Path loss, signal strength

**Radio Map**:
The dense spatial field of Path Gain for exactly one Scene and one Transmitter.
_Avoid_: Channel map, signal map, image, path-loss map

**Reference Radio Map**:
The complete dataset-provided Radio Map that serves as the reconstruction target.
_Avoid_: Ground-truth image, full channel map

**Reconstruction**:
A predicted Radio Map inferred from a Scene, a Transmitter, and an Observation Set.
_Avoid_: Completed image, generated map

## Observations

**Observation**:
A known Path Gain value at one Outdoor Cell.
_Avoid_: Sample point, known pixel

**Observation Set**:
The Observations available for reconstructing one Reference Radio Map.
_Avoid_: Sparse image, sparse channel map

**Sampling Candidate Region**:
All Outdoor Cells other than the Transmitter location that are eligible to become Observations.
_Avoid_: Valid mask, sampling area

**Sampling Rate**:
The fraction of the Sampling Candidate Region included in an Observation Set.
_Avoid_: Sparsity, mask density

**Observation Realization**:
A concrete Observation Set drawn for one Reference Radio Map at one Sampling Rate.
_Avoid_: Random mask, sampling pattern

## Reconstruction Contract

**Reconstruction Region**:
The unobserved Outdoor Cells on which reconstruction error is measured.
_Avoid_: Evaluation mask, unknown pixels

**Data Consistency**:
The invariant that a Reconstruction equals the corresponding Observation at every observed Receiver Cell.
_Avoid_: Copying samples, output masking

**Scene-Disjoint Split**:
A dataset partition in which each Scene belongs to exactly one of training, validation, or test, together with all of its Transmitters and Observation Realizations.
_Avoid_: Image split, sample split, random train-test split

**Scene-Level Evaluation**:
Evaluation that treats a Scene as the independent statistical unit and keeps its Transmitters and Observation Realizations grouped together.
_Avoid_: Per-pixel significance, per-sample independence

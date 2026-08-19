---
status: accepted
---

# Partition RadioMapSeer by Scene

RadioMapSeer pairs each urban Scene with many Transmitters, so Radio Maps from the same Scene share building geometry. We partition Scenes—not individual Radio Maps—into fixed training, validation, and test sets of 501, 100, and 100 Scenes, and keep every Transmitter and Observation Realization for a Scene in the same split. This trades some split flexibility for a leakage-resistant estimate of generalization to unseen Scenes; changing the partition would invalidate comparisons with existing experiment results.

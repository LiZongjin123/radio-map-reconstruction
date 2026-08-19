---
status: accepted
---

# Treat Synthetic Observations as Exact Constraints

The DPM baseline draws sparse Observations directly from each Reference Radio Map, so they contain no measurement noise. A Reconstruction must therefore reproduce every Observation exactly, and training and evaluation measure error only over the unobserved Outdoor Cells in the Reconstruction Region. Allowing the model to rewrite known values would conflate reconstruction with denoising, while including observed or building cells in metrics would dilute the error with trivially known or irrelevant values; a future noisy-observation task should supersede this decision with an explicit noise model and soft consistency rule.

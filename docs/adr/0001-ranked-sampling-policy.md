# ADR-0001: Use a K-independent ranked sampling policy

## Status

Accepted

## Context

The reconstructor consumes a sparse channel map but does not decide where measurements are collected. A learned sampler must choose exactly K valid sampling points without observing channel-map measurements, support several sampling budgets with one model, and remain trainable through the discrete physical decision.

The reconstructor and sampler also have different responsibilities and training lifecycles. Updating reconstructor weights while learning the sampling policy would confound improvements in sampling with improvements in reconstruction.

## Decision

Use a two-input, one-output ResUNet with 16 base channels as a ranked sampling policy. Its inputs are the building map followed by the transmitter map, and its output is an unconstrained sampling score map. The score map is independent of K.

For each sample, rank only the valid receiving area. Break equal scores by row-major position and select the first K positions. Consequently, learned sampling masks for increasing K are nested.

Use the hard binary learned sampling mask in the forward pass. During backpropagation, use the gradient of a sigmoid soft Top-K surrogate. Solve an adaptive threshold per sample by bisection, treat that threshold as constant during differentiation, and configure the positive temperature, tolerance, and maximum iteration count. Use the exact sigmoid derivative while it is representable; if an extreme positive temperature or score would make that derivative exceed the score dtype's finite range, clamp it to a dtype-safe finite value.

Train the sampler through a frozen reconstructor. Reconstructor parameters do not receive updates, but its forward pass remains in the gradient graph so reconstruction loss can reach the sampler.

Maintain independent checkpoint and run-artifact lifecycles for the sampler and reconstructor. Training or replacing either model must not overwrite the other model's artifacts.

## Consequences

- One sampling score map supports every requested K and creates predictable nested learned sampling masks.
- Physical training inputs use exact binary masks while the sampler still receives a stable surrogate gradient.
- The estimator is biased, gradients do not include the dependence of the adaptive threshold on the scores, and numerical gradient clipping adds further bias only at the dtype's representational limit.
- Freezing the reconstructor isolates the sampling research question but prevents joint adaptation.
- Independent artifacts require explicit orchestration of both checkpoints in later training and evaluation workflows.

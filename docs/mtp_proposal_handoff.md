# MTP Proposal Handoff

The default P/D proposal encoding is marker-only for deterministic top-1:
`proposal_is_point_mass=true`, without `propose_probs`. This is accepted by
the marker-aware receiver in `fcbb549e0` and newer receivers, including receivers
that allow both fields. No dense probability tensor is transferred in this mode.

For a Decode deployment that predates the point-mass marker, set
`RTP_LLM_MTP_LEGACY_DENSE_HANDOFF=1` on **Prefill** before starting the service.
This sends `proposal_is_point_mass=false` and dense probabilities in the draft
vocabulary. A reduced draft vocabulary uses its `d2t` map to preserve probability
positions; ordinary dense proposals retain their original probabilities.

Do not enable legacy mode when Decode is `fcbb549e0`: although that receiver can
read dense probabilities, its multi-step assembler cannot combine a dense first
proposal with subsequent point-mass steps. Use the default marker-only mode for
that receiver. New Decode supports legacy dense-first multi-step and mixed
batches, mapping dense probabilities into target vocabulary before assembly.

When a Prefill pool targets heterogeneous Decode versions, route requests through
separate Prefill pools configured for the corresponding proposal encoding. This
switch is explicit, not automatic peer-version negotiation. On rollback, align
the Prefill mode with the Decode destination before routing traffic.

This compatibility contract covers proposal encoding only. All peers must still
satisfy the grouped KV-cache topology and block-addressing protocol contract;
this switch does not enable arbitrary cross-version KV-cache deployments.

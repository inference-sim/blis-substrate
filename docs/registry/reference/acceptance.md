# Acceptance profiles

<p class="lede">How far speculative decoding advances a request per verification step, by
model, speculative method, K and thinking mode. Generated at build time from
<code>acceptance/</code>, at revision <!-- registry:ref -->; nothing below is typed in by
hand.</p>

An acceptance-profile set is the acceptance counterpart of a coefficient set. Its format,
and how a run resolves one value from it, are on the contracts'
[acceptance reference](../../contracts/reference/acceptance.md)
(blis-schemas#65). Each profile holds the mean **acceptance length** (AL): the tokens a
request gains per verification step, counting the target's bonus token, so it lies in
[1, K + 1]. The InferenceX set stores no per-position rates; a consumer that needs them
derives vLLM v0.31.0's minimum-variance schedule, which is how the InferenceX runs that
force a published AL realise it. A set measured from an engine's own counters stores them,
and may state a [chain schedule](../../contracts/reference/acceptance.md#schedules) that
draws each step's acceptance.

## `inferencex-golden-al`

SemiAnalysis InferenceX's golden SPEED-Bench curves, the values its AgentX runs force
through vLLM's `synthetic_acceptance_length`. A cell links to its profile in the file. A
dot (·) is a K with no profile, and a lookup for it is refused rather than interpolated.

<!-- registry:acceptance inferencex-golden-al values -->

### Where each curve was measured

One row per InferenceX file. The settings column lists every engine setting the file
states, including the `chat_template_kwargs` of each thinking mode.

<!-- registry:acceptance inferencex-golden-al provenance -->

### Caveats

The set's own `caveats`, as committed:

<!-- registry:acceptance inferencex-golden-al caveats -->

## How the set is made

`scripts/emit_acceptance_profiles.py` writes the file, and its `--check` is the drift
gate, as for the coefficient writers. The rules it follows are in
[methodology §12](../methodology.md#12-speculative-acceptance-profiles).

## `glm53-h200-agentic`

GLM-5.3 with its MTP layers at K 3, measured on one 8xH200 node under an eight-round
agentic workload (inference-sim/inference-sim#2094). Its per-position rates come from the
engine's per-position acceptance counters, and its `request_logit_sd` from how acceptance
spread across requests, so a run that resolves it draws each step's accepted drafts as a
chain rather than advancing every request by the mean.

<!-- registry:acceptance glm53-h200-agentic values -->

### Where it was measured

<!-- registry:acceptance glm53-h200-agentic provenance -->

### Caveats

<!-- registry:acceptance glm53-h200-agentic caveats -->

### How the set is made

`scripts/emit_acceptance_measured.py` writes the file from a summary it derives from the
run's archive, and its `--check` is the drift gate. The fit is described in
[methodology §12.6](../methodology.md#126-a-profile-measured-from-a-run).

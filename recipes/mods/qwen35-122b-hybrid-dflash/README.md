# Qwen3.5 122B hybrid vLLM patch mod

This mod applies the runtime patches from
[`Entrpi/qwen3.5-122B-A10B-on-spark`](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark)
at pinned commit `a77cbdab26956ef6ac9cdca544e5fb9ec1f3bb2a`.

The downloaded files are SHA-256 verified before execution. The hybrid INC
INT4/FP8 dispatch, INT8 lm-head, DFlash KV-page unification, and prefix-alignment
patches are mandatory and fail closed if the pinned container source differs.
The FLA shared-memory patch is an optional prefill optimization.

Upstream is MIT licensed. The patch sources are downloaded directly from the
pinned upstream revision rather than duplicated here.

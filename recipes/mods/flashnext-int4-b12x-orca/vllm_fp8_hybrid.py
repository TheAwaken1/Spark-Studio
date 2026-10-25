"""vllm_fp8_hybrid — int4(GPTQ-Marlin) + blockwise-fp8 hybrid dispatch.

The checkpoint's routed experts + lm_head are GPTQ int4/int8 (served by
AutoGPTQConfig), while dense side-layers (GDN in_proj/out_proj, QSA q/k/v/o,
shared experts) are stored as blockwise FP8-e4m3 (`weight` fp8 +
`weight_scale_inv`, block 128x128). Stock AutoGPTQConfig would init those as
unquantized bf16 and then fail to load fp8 tensors.

This patch (enabled by VLLM_FP8_HYBRID=1, no-op otherwise):
  * after ``maybe_update_config``, scans the checkpoint's safetensors metadata
    for F8_E4M3 weights that have a ``.weight_scale_inv`` sibling and records
    those layer names;
  * maps them through the model's hf->vllm mapper alongside
    ``modules_in_block_to_quantize``;
  * in ``get_quant_method``, dispatches matching LinearBase layers (including
    fused qkv_proj / gate_up_proj via ``packed_modules_mapping``) to a shared
    blockwise ``Fp8Config`` instead of the GPTQ/unquantized path.

Port of Saren's spark-dflash-hybrid-fp8 (vLLM 0.23 INCConfig) onto the
qwen38next build's AutoGPTQConfig.
"""
import logging
import os

logger = logging.getLogger("vllm.fp8_hybrid")

_SENTINEL = "_fp8_hybrid_patched"


def _index_filtered_metadata(model_name, revision=None):
    """flashnext-int4-b12x change to Saren's module: read each tensor's dtype from the file the INDEX names for it.

    The served-model view links PLE-table files that also carry the original checkpoint's layer-1 tensors, including a bf16
    shared expert. get_safetensors_params_metadata merges every file in the folder, so whichever file came last in directory
    order decided that layer's dtype -- different on the two nodes (812 vs 809 fp8 layers, 2026-09-21): the node that saw bf16
    built layer 1's shared expert unquantized and died loading its fp8 weights. No index (e.g. a hub id) -> stock behaviour.
    """
    import json
    import os
    import struct
    index = os.path.join(str(model_name), "model.safetensors.index.json")
    if not os.path.isfile(index):
        from vllm.transformers_utils.config import get_safetensors_params_metadata
        return get_safetensors_params_metadata(model_name, revision=revision)
    by_file = {}
    for name, fname in json.load(open(index))["weight_map"].items():
        by_file.setdefault(fname, set()).add(name)
    md = {}
    for fname, names in by_file.items():
        with open(os.path.join(str(model_name), fname), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name in names:
            if name in header:
                md[name] = header[name]
    return md


def apply() -> None:
    if os.environ.get("VLLM_FP8_HYBRID", "0").lower() not in ("1", "true", "yes"):
        return
    from vllm.model_executor.layers.quantization import auto_gptq as m

    cfg_cls = m.AutoGPTQConfig
    if getattr(cfg_cls, _SENTINEL, False):
        return

    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.transformers_utils.config import get_safetensors_params_metadata

    orig_update = cfg_cls.maybe_update_config
    orig_mapper = cfg_cls.apply_vllm_mapper
    orig_gqm = cfg_cls.get_quant_method

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        orig_update(self, model_name, hf_config=hf_config, revision=revision)
        md = _index_filtered_metadata(model_name, revision)
        self.fp8_layers = {
            name[: -len(".weight")]
            for name, info in md.items()
            if name.endswith(".weight")
            and info.get("dtype") == "F8_E4M3"
            and name[: -len(".weight")] + ".weight_scale_inv" in md
        }
        if self.fp8_layers:
            logger.info(
                "fp8 hybrid: %d blockwise-fp8 layers detected", len(self.fp8_layers)
            )

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        orig_mapper(self, hf_to_vllm_mapper)
        if getattr(self, "fp8_layers", None):
            self.fp8_layers = set(hf_to_vllm_mapper.apply_list(list(self.fp8_layers)))

    def _is_fp8_layer(self, prefix: str) -> bool:
        fp8_layers = getattr(self, "fp8_layers", None)
        if not fp8_layers:
            return False
        head, _, proj = prefix.rpartition(".")
        fused = self.packed_modules_mapping.get(proj)
        names = [f"{head}.{p}" for p in fused] if fused and head else [prefix]
        return all(any(l in n for l in fp8_layers) for n in names)

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and self._is_fp8_layer(prefix):
            fp8_cfg = getattr(self, "_fp8_cfg", None)
            if fp8_cfg is None:
                fp8_cfg = Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    activation_scheme="dynamic",
                    weight_block_size=[128, 128],
                )
                fp8_cfg.packed_modules_mapping = self.packed_modules_mapping
                self._fp8_cfg = fp8_cfg
            logger.debug("fp8 hybrid: %s -> Fp8LinearMethod", prefix)
            return fp8_cfg.get_quant_method(layer, prefix)
        return orig_gqm(self, layer, prefix)

    cfg_cls.maybe_update_config = maybe_update_config
    cfg_cls.apply_vllm_mapper = apply_vllm_mapper
    cfg_cls._is_fp8_layer = _is_fp8_layer
    cfg_cls.get_quant_method = get_quant_method
    setattr(cfg_cls, _SENTINEL, True)
    logger.info("fp8 hybrid patch applied to AutoGPTQConfig")

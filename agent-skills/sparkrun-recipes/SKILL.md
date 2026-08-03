---
name: sparkrun-recipes
description: Create, repair, review, and optimize sparkrun inference recipe YAML for DGX Spark.
---

# sparkrun recipes

Use this skill whenever a task creates, fixes, tunes, or evaluates a sparkrun
recipe or translates a model launch command into recipe YAML.

## Reference docs

- `${HERMES_SKILL_DIR}/../../Title Recipe Format.md` (repo root)

## Quick start

Before changing a recipe, read the canonical local reference at:

    `${HERMES_SKILL_DIR}/../../Title Recipe Format.md`

Then:

1. Inspect the existing recipe, registry conventions, and the active model or
   failure evidence before proposing changes.
2. Prefer explicit `runtime`, `min_nodes`, and `max_nodes`; preserve valid
   registry-specific conventions when they intentionally differ.
3. Keep overridable values in `defaults` and reference them with command
   placeholders instead of duplicating literals.
4. Never hardcode credentials. Use environment expansion where required.
5. Validate the YAML and run the narrowest relevant recipe, launch, doctor,
   or benchmark checks available in the workspace.

## Custom-engine pitfalls

- A custom command under `runtime: llama-cpp` is rendered correctly, including
  the pre-synced GGUF path, but setting `served_model_name` makes sparkrun append
  llama.cpp's `--alias`. Omit that default when the custom binary does not
  support `--alias`.
- Entrpi/ds4 with the DeepSeek V4 Flash 0731 base must use the matching DSpark
  drafter explicitly. Do not pass `--preset spark`: that preset also requires
  the legacy MTP GGUF, which is incompatible with the 0731 base.
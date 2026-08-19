---
name: sparkrun-recipes
description: Créer, réparer, relire et optimiser des recipes d'inférence sparkrun (YAML) pour DGX Spark.
---

# sparkrun recipes

Utilisez cette skill dès qu'une tâche crée, corrige, ajuste ou évalue une
recipe sparkrun, ou traduit une commande de lancement de modèle en YAML de
recipe.

## Documentation de référence

- `${HERMES_SKILL_DIR}/../../Title Recipe Format.md` (racine du dépôt)

## Démarrage rapide

Avant de modifier une recipe, lisez la référence canonique locale :

    `${HERMES_SKILL_DIR}/../../Title Recipe Format.md`

Puis :

1. Inspectez la recipe existante, les conventions de registry, et le modèle
   actif ou les preuves d'échec avant de proposer des changements.
2. Préférez des valeurs explicites pour `runtime`, `min_nodes` et
   `max_nodes` ; préservez les conventions spécifiques à un registry
   lorsqu'elles diffèrent intentionnellement.
3. Gardez les valeurs surchargeables dans `defaults` et référencez-les via
   des placeholders de commande au lieu de dupliquer des littéraux.
4. Ne codez jamais d'identifiants en dur. Utilisez l'expansion par variable
   d'environnement là où c'est nécessaire.
5. Validez le YAML et lancez les vérifications de recipe, de lancement, de
   doctor ou de benchmark les plus étroites disponibles dans le workspace.

## Pièges des engines custom

- Une commande custom sous `runtime: llama-cpp` est correctement rendue, y
  compris le chemin GGUF pré-synchronisé, mais définir `served_model_name`
  fait que sparkrun ajoute l'option `--alias` de llama.cpp. Omettez ce
  défaut si le binaire custom ne supporte pas `--alias`.
- Entrpi/ds4 avec la base DeepSeek V4 Flash 0731 doit utiliser
  explicitement le drafter DSpark correspondant. Ne passez pas
  `--preset spark` : ce preset requiert aussi le GGUF MTP legacy, qui est
  incompatible avec la base 0731.

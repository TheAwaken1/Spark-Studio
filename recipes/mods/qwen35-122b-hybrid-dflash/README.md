# Mod de patch vLLM hybride Qwen3.5 122B

Ce mod applique les patches de runtime depuis
[`Entrpi/qwen3.5-122B-A10B-on-spark`](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark)
au commit pinné `a77cbdab26956ef6ac9cdca544e5fb9ec1f3bb2a`.

Les fichiers téléchargés sont vérifiés par SHA-256 avant exécution. Les
patches de dispatch hybride INC INT4/FP8, lm-head INT8, unification des
KV-pages DFlash et alignement de préfixe sont obligatoires et échouent
strictement (fail closed) si la source du container pinné diffère. Le
patch FLA shared-memory est une optimisation de prefill optionnelle.

L'upstream est sous licence MIT. Les sources des patches sont téléchargées
directement depuis la révision upstream pinnée, plutôt que dupliquées ici.

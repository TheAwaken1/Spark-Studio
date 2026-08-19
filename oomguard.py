"""Garde le dashboard en vie quand la mémoire unifiée se remplit.

Sur une DGX Spark, l'assistant `sparkrun setup` installe earlyoom avec
`--prefer '(...|python3|python)'`. Le serveur de Spark Studio est un simple
process `python`, donc earlyoom traite le dashboard ~100 MB comme une cible
de kill préférée — au même titre que le modèle multi-GB. Un chargement de
modèle qui remplit les 128 GB de mémoire unifiée peut alors faire tomber
le control plane avec lui (SIGKILL → le `Killed` brut dans le terminal).

Deux mitigations best-effort, sans privilège au lancement :

- protect_self() : tente d'abaisser notre propre oom_score_adj. Un process
  non privilégié peut *augmenter* mais pas *abaisser* son score, donc cela ne
  prend effet que si l'app tourne avec privilège (par ex. une unit systemd
  avec `OOMScoreAdjust=-500`). Le résultat est loggé dans les deux cas pour
  que l'opérateur sache où il en est.

- deprioritize(pid) : augmente l'oom_score_adj d'un engine spawné pour que
  le *modèle* soit la victime OOM avant le dashboard. Augmenter le score
  d'un process qu'on possède est toujours permis, donc c'est la mitigation
  qui marche réellement sur un lancement non privilégié normal — pour les
  sous-processus d'engine que nous démarrons directement.

The robust, box-wide fix is removing `python` from earlyoom's --prefer list;
see the README (Memory / OOM section).
"""
from __future__ import annotations

# Push engine subprocesses well above the dashboard's ~800 base oom_score so the
# OOM killer / earlyoom takes the model first. Clamped to the kernel's max.
_ENGINE_OOM_SCORE_ADJ = 900
# Modest protective floor for our own process (only settable with privilege).
_SELF_OOM_SCORE_ADJ = -500


def protect_self() -> str:
    """Best-effort lower this process's OOM priority. Returns a status string
    suitable for logging (never raises)."""
    try:
        with open("/proc/self/oom_score_adj") as f:
            cur = int(f.read().strip() or "0")
    except OSError:
        return "OOM guard: procfs unavailable (not Linux?) — skipped"
    if cur <= _SELF_OOM_SCORE_ADJ:
        return f"OOM guard: already protected (oom_score_adj={cur})"
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(str(_SELF_OOM_SCORE_ADJ))
        return f"OOM guard: lowered own oom_score_adj to {_SELF_OOM_SCORE_ADJ}"
    except OSError:
        return (
            "OOM guard: can't lower own OOM priority without privilege. "
            "Engine subprocesses are still deprioritized so the model is killed "
            "before this dashboard. For full protection on a box with earlyoom, "
            "remove 'python' from its --prefer list (see README: Memory / OOM)."
        )


def deprioritize(pid: int) -> bool:
    """Make an engine `pid` a preferred OOM victim over the dashboard, so memory
    pressure kills the (relaunchable) model instead of the control plane.
    Best-effort; returns True on success."""
    try:
        with open(f"/proc/{pid}/oom_score_adj", "w") as f:
            f.write(str(_ENGINE_OOM_SCORE_ADJ))
        return True
    except OSError:
        return False

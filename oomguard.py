"""Keep the dashboard alive when unified memory fills up.

On a DGX Spark the `sparkrun setup` wizard installs earlyoom with
`--prefer '(...|python3|python)'`. Spark Studio's server is a plain `python`
process, so earlyoom treats the ~100 MB dashboard as a preferred kill target —
right alongside the multi-GB model. A model load that fills the 128 GB unified
memory can then take the control plane down with it (SIGKILL → the bare
`Killed` in the terminal).

Two best-effort mitigations, neither requiring privilege at launch:

- protect_self(): try to lower our own oom_score_adj. An unprivileged process
  can *raise* but not *lower* its score, so this only takes effect when the app
  runs with privilege (e.g. a systemd unit with `OOMScoreAdjust=-500`). It logs
  the outcome either way so the operator knows where they stand.

- deprioritize(pid): raise a spawned engine's oom_score_adj so the *model* is
  the OOM victim ahead of the dashboard. Raising the score of a process you own
  is always permitted, so this is the mitigation that actually works on a normal
  unprivileged launch — for engine subprocesses we start directly.

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

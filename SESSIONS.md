# Session Log

A running log of AI-assisted development sessions on Spark Studio (Claude Code,
Codex, or any other LLM). Newest entries at the top.

Every agent working on this repo should append an entry here at the end of a
session, using this format:

```
## YYYY-MM-DD — <agent> — <short title>
**Asked for:** what the user requested
**Done:** what actually changed (files, behavior)
**Notes:** decisions made, gotchas, follow-ups
```

---

## 2026-07-11 — Claude Code — Run ordering, favorites, Enter-to-send, session log

**Asked for:**
- Overview + vLLM/SGLang/llama.cpp tabs: recent/running runs should sort to the top (were at the bottom).
- Recipes & Community tab: add a favorite button; favorites pin to the front.
- Engine chat tab: Enter should send the message (was Ctrl+Enter); reverse the binding.
- A session log file documenting every AI session (this file).

**Done:**
- `runners.py` — `Runner.list()` now sorts running runs first, then newest-first
  by `started_at`. This fixes ordering everywhere at once (Overview, all engine
  tab "Recent runs" panels) since they all read `/api/runs`.
- `web/app.js` — favorites for both My Recipes (`recipeFavs`, keyed by recipe id)
  and Community (`communityFavs`, keyed by @ref), stored in localStorage.
  Star button on each card (`favBtn`/`getFavs`/`toggleFav`/`bindFavButtons`);
  favorited cards sort first and get a gold border (`.is-fav`).
- `web/app.js` + `web/index.html` — `#chatInput` (Engine chat) and `#wgInput`
  (WebLLM/web-grounded chat): Enter sends, Shift+Enter inserts a newline,
  Ctrl/Cmd+Enter no longer required. Placeholders updated to match.
- `web/style.css` — `.fav-btn` and `.recipe-card.is-fav` styles.
- Created this `SESSIONS.md` and `CLAUDE.md`/`AGENTS.md` pointing agents here.

**Notes:**
- Favorites are per-browser (localStorage), not in the SQLite DB — swapping
  browsers/machines won't carry them over. Move to the `recipes` table (e.g. a
  `favorite` column or a `favorite` tag) if cross-device favorites are wanted.
- User lost the previous conversation to a system restart. For continuity:
  `claude --continue` / `claude --resume` restore past Claude Code sessions
  (stored in `~/.claude/projects/`), and this file is the cross-tool fallback.

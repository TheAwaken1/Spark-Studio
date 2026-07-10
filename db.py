import sqlite3
import json
import time
from contextlib import contextmanager
from pathlib import Path

DB_DIR = Path(__file__).parent / "data"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "spark_studio.db"


def _connect():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_conn = _connect()


def _add_col_if_missing(table: str, col: str, decl: str) -> None:
    cols = [r[1] for r in _conn.execute(f"PRAGMA table_info({table})")]
    if col not in cols:
        _conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def init():
    _conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            engine TEXT NOT NULL,
            model TEXT,
            args_json TEXT NOT NULL,
            env_json TEXT NOT NULL DEFAULT '{}',
            notes TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            raw_cmd TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            recipe_id INTEGER,
            engine TEXT NOT NULL,
            status TEXT NOT NULL,
            pid INTEGER,
            port INTEGER,
            cmd TEXT NOT NULL,
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            exit_code INTEGER,
            FOREIGN KEY (recipe_id) REFERENCES recipes(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            recipe_id INTEGER,
            tokens_per_sec REAL,
            ttft_ms REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            memory_mb REAL,
            data_json TEXT,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
        CREATE INDEX IF NOT EXISTS idx_bench_recipe ON benchmarks(recipe_id);
        CREATE TABLE IF NOT EXISTS benchy_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            recipe_id INTEGER,
            model TEXT,
            base_url TEXT,
            params_json TEXT NOT NULL,
            result_json TEXT,
            exit_code INTEGER,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_benchy_recipe ON benchy_runs(recipe_id);
        CREATE TABLE IF NOT EXISTS tooleval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            recipe_id INTEGER,
            model TEXT,
            base_url TEXT,
            score REAL,
            results_json TEXT,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tooleval_model ON tooleval_runs(model);
        """
    )


def now():
    return int(time.time())


@contextmanager
def cur():
    c = _conn.cursor()
    try:
        yield c
    finally:
        c.close()


def recipes_list():
    with cur() as c:
        c.execute("SELECT * FROM recipes ORDER BY updated_at DESC")
        return [_recipe_row(r) for r in c.fetchall()]


def recipes_get(rid):
    with cur() as c:
        c.execute("SELECT * FROM recipes WHERE id=?", (rid,))
        r = c.fetchone()
        return _recipe_row(r) if r else None


def _recipe_row(r):
    if not r:
        return None
    d = dict(r)
    d["args"] = json.loads(d.pop("args_json") or "{}")
    d["env"] = json.loads(d.pop("env_json") or "{}")
    return d


def recipes_upsert(data):
    t = now()
    args_json = json.dumps(data.get("args") or {})
    env_json = json.dumps(data.get("env") or {})
    raw_cmd = data.get("raw_cmd")
    with cur() as c:
        if data.get("id"):
            c.execute(
                "UPDATE recipes SET name=?, engine=?, model=?, args_json=?, env_json=?, notes=?, tags=?, raw_cmd=?, updated_at=? WHERE id=?",
                (
                    data["name"],
                    data["engine"],
                    data.get("model"),
                    args_json,
                    env_json,
                    data.get("notes", ""),
                    data.get("tags", ""),
                    raw_cmd,
                    t,
                    data["id"],
                ),
            )
            return recipes_get(data["id"])
        else:
            c.execute(
                "INSERT INTO recipes (name, engine, model, args_json, env_json, notes, tags, raw_cmd, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    data["name"],
                    data["engine"],
                    data.get("model"),
                    args_json,
                    env_json,
                    data.get("notes", ""),
                    data.get("tags", ""),
                    raw_cmd,
                    t,
                    t,
                ),
            )
            return recipes_get(c.lastrowid)


def recipes_delete(rid):
    with cur() as c:
        c.execute("DELETE FROM recipes WHERE id=?", (rid,))


def recipes_find_sparkrun(ref):
    """Find the recipe auto-saved for a sparkrun community ref (dedupe key is
    args._sparkrun.ref so user renames don't break matching)."""
    for r in recipes_list():
        if ((r.get("args") or {}).get("_sparkrun") or {}).get("ref") == ref:
            return r
    return None


def recipes_set_status_tag(rid, ok):
    """Single writer for the working/fix status tags. ok=True marks the recipe
    working; ok=False marks it broken (fix). Safe to call from any thread."""
    rec = recipes_get(rid)
    if not rec:
        return
    tags = {t.strip() for t in (rec.get("tags") or "").split(",") if t.strip()}
    before = set(tags)
    if ok:
        tags.add("working")
        tags.discard("fix")
    else:
        tags.add("fix")
        tags.discard("working")
    if tags == before:
        return
    with cur() as c:
        c.execute("UPDATE recipes SET tags=?, updated_at=? WHERE id=?", (", ".join(sorted(tags)), now(), rid))


def runs_insert(run):
    with cur() as c:
        c.execute(
            "INSERT INTO runs (id, recipe_id, engine, status, pid, port, cmd, started_at, meta_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run["id"],
                run.get("recipe_id"),
                run["engine"],
                run["status"],
                run.get("pid"),
                run.get("port"),
                run["cmd"],
                run.get("started_at") or now(),
                run.get("meta_json"),
            ),
        )


def runs_update(rid, **fields):
    if not fields:
        return
    keys = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [rid]
    with cur() as c:
        c.execute(f"UPDATE runs SET {keys} WHERE id=?", vals)


def runs_list(limit=100):
    with cur() as c:
        c.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]


def runs_list_running():
    with cur() as c:
        c.execute("SELECT * FROM runs WHERE status='running' ORDER BY started_at DESC")
        return [dict(r) for r in c.fetchall()]


def runs_get(rid):
    with cur() as c:
        c.execute("SELECT * FROM runs WHERE id=?", (rid,))
        r = c.fetchone()
        return dict(r) if r else None


def bench_insert(run_id, recipe_id, metrics, engine_version=None):
    with cur() as c:
        c.execute(
            "INSERT INTO benchmarks (run_id, recipe_id, tokens_per_sec, ttft_ms, prompt_tokens, completion_tokens, memory_mb, data_json, engine_version, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                recipe_id,
                metrics.get("tokens_per_sec"),
                metrics.get("ttft_ms"),
                metrics.get("prompt_tokens"),
                metrics.get("completion_tokens"),
                metrics.get("memory_mb"),
                json.dumps(metrics),
                engine_version,
                now(),
            ),
        )


def bench_list(recipe_id=None, limit=100):
    with cur() as c:
        if recipe_id:
            c.execute(
                "SELECT * FROM benchmarks WHERE recipe_id=? ORDER BY created_at DESC LIMIT ?",
                (recipe_id, limit),
            )
        else:
            c.execute("SELECT * FROM benchmarks ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]


def benchy_insert(run_id, recipe_id, model, base_url, params, result, exit_code, engine_version=None):
    with cur() as c:
        c.execute(
            "INSERT INTO benchy_runs (run_id, recipe_id, model, base_url, params_json, result_json, exit_code, engine_version, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                recipe_id,
                model,
                base_url,
                json.dumps(params),
                json.dumps(result) if result is not None else None,
                exit_code,
                engine_version,
                now(),
            ),
        )
        return c.lastrowid


def benchy_list(limit=50):
    with cur() as c:
        c.execute("SELECT * FROM benchy_runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]


def benchy_get(bid):
    with cur() as c:
        c.execute("SELECT * FROM benchy_runs WHERE id=?", (bid,))
        r = c.fetchone()
        return dict(r) if r else None


def tooleval_insert(row):
    with cur() as c:
        c.execute(
            "INSERT INTO tooleval_runs (run_id, recipe_id, model, base_url, score, results_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                row.get("run_id"),
                row.get("recipe_id"),
                row.get("model"),
                row.get("base_url"),
                row.get("score"),
                row.get("results_json"),
                now(),
            ),
        )
        return c.lastrowid


def tooleval_list(limit=50):
    with cur() as c:
        c.execute("SELECT * FROM tooleval_runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]


init()
_add_col_if_missing("recipes", "raw_cmd", "TEXT")
_add_col_if_missing("benchmarks", "engine_version", "TEXT")
_add_col_if_missing("benchy_runs", "engine_version", "TEXT")
_add_col_if_missing("runs", "meta_json", "TEXT")

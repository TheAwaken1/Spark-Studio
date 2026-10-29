"""One opt-in launch pass per application boot; never owns/stops model runs."""
import asyncio
import math
import os
import time


class BootQueue:
    def __init__(self, recipes, memory_check, launch, runs):
        self._recipes, self._memory_check = recipes, memory_check
        self._launch, self._runs = launch, runs
        self._attempted = False
        self._stopped = False
        self._state = 'pending'
        self._current = None
        self._results = []
        self._reason = None

    def status(self):
        return dict(state=self._state, current_recipe_id=self._current,
                    reason=self._reason, results=[dict(r) for r in self._results])

    def stop(self):
        # Latch even before reconciliation completes. Never call runner.stop().
        self._stopped = True
        self._state = 'stopped'
        return self.status()

    def _record(self, recipe, status, reason):
        self._results.append(dict(recipe_id=recipe['id'], status=status, reason=reason))
        print(f"[autostart] recipe {recipe['id']}: {status} — {reason}", flush=True)

    async def run(self):
        if self._attempted:
            return
        self._attempted = True  # set before the first await: no concurrent/retry pass
        if self._stopped:
            return
        if os.environ.get('SPARK_STUDIO_AUTOSTART', '1') == '0':
            self._state, self._reason = 'disabled', 'SPARK_STUDIO_AUTOSTART=0'
            print(f'[autostart] disabled: {self._reason}', flush=True)
            return
        self._state = 'running'
        try:
            recipes = self._recipes()
            if not recipes:
                return
            memory = await asyncio.to_thread(self._memory_check)
            if memory.get('status') == 'error':
                self._state = 'blocked'
                self._reason = 'critical memory: ' + str(memory.get('detail', ''))
                print(f'[autostart] blocked: {self._reason}', flush=True)
                return
            try:
                timeout = float(os.environ.get('SPARK_STUDIO_AUTOSTART_TIMEOUT', '600'))
                if not math.isfinite(timeout) or timeout <= 0:
                    raise ValueError
            except ValueError:
                timeout = 600.0
            for recipe in recipes:
                if self._stopped:
                    break
                self._current = recipe['id']
                if any(r.status == 'running' and r.recipe_id == recipe['id'] for r in self._runs()):
                    self._record(recipe, 'skipped', 'already running (re-adopted or manually launched)')
                    continue
                try:
                    result = await asyncio.to_thread(self._launch, recipe)
                except Exception as exc:  # a refusal must not retry this recipe
                    reason = getattr(exc, 'detail', None) or str(exc)
                    self._record(recipe, 'skipped', str(reason))
                    continue
                deadline = time.monotonic() + timeout
                while not self._stopped:
                    run = next((r for r in self._runs() if r.id == result['id']), None)
                    if run is None or run.status != 'running':
                        self._record(recipe, 'failed', 'run exited before ready')
                        break
                    if run.ready:
                        self._record(recipe, 'ready', 'serving')
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._record(recipe, 'timeout', f'not ready after {timeout:g}s; run left up')
                        break
                    await asyncio.sleep(min(.05, remaining))
        except Exception as exc:
            self._state, self._reason = 'blocked', str(exc)
            print(f'[autostart] blocked: {exc}', flush=True)
        finally:
            self._current = None
            if self._stopped:
                self._state = 'stopped'
            elif self._state == 'running':
                self._state = 'completed'

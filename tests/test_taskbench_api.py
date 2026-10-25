import unittest
from unittest import mock

import server


class TaskBenchApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_endpoint_passes_selected_cases_and_limits(self):
        endpoint = {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "local-model",
            "studio_url": "http://127.0.0.1:7860",
            "models": ["local-model"],
        }
        request = server.TaskBenchReq(
            base_url="http://127.0.0.1:8000/v1",
            model="local-model",
            cases=["file-audit"],
            max_turns=22,
            timeout=120,
        )
        with (
            mock.patch.object(server.agentlab, "discover_endpoint", return_value=endpoint),
            mock.patch.object(server.taskbench, "start_eval", return_value={"running": True}) as start,
        ):
            result = await server.taskbench_run(request)

        self.assertTrue(result["running"])
        start.assert_called_once_with(
            endpoint,
            case_ids=["file-audit"],
            max_turns=22,
            timeout=120.0,
            unsafe_yolo=False,
        )


if __name__ == "__main__":
    unittest.main()

"""Import and entry-point regression tests using only the standard library.

Set EXPERIMENT_SOURCE_PATH to run this exact harness against another source file.
The scientific dependencies and multiprocessing constructors are boundary doubles;
the source is loaded and executed without rewriting its code or creating workers.
"""

import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest import mock


SOURCE_PATH = Path(
    os.environ.get(
        "EXPERIMENT_SOURCE_PATH", str(Path(__file__).resolve().parents[1] / "experiment.py")
    )
).resolve()

SCIENTIFIC_AST_SHA256 = {
    "estimate_channel_capacity": "39790c628f64db364fbd4ef89f4a6351a3cb31ce383206abc9378aedef909580",
    "generate_huffman_codes": "825b0df577264f8dcba5b58ba157f26f64d8eb9f22c514043a49c0b59abe6034",
    "mutual_information": "f639427e7e723e4a987117039bb63e7d7d534515aa4651f4a88e65c7e4098a5c",
    "complete_agent": "f561a2a2a554bd2f4886f1a2d513c37245f2878da3993c4bee3417959fe1f36a",
    "run_complete_experiment": "4cff5e6a77856f87bfdf60bb2db7db28d3b0b5fa2ef285c261715bcba3b90af2",
}

LOG1 = [(7, 0.125, "Agent1"), (2, 0.75, "Agent1"), (11, 0.125, "Agent1")]
LOG2 = [(3, 0.875, "Agent2"), (13, 0.25, "Agent2"), (5, 0.875, "Agent2")]
TIME1 = (7, 2, 11)
TIME2 = (3, 13, 5)
BEHAVIOR1 = (0.125, 0.75, 0.125)
BEHAVIOR2 = (0.875, 0.25, 0.875)
EXPECTED_MI = -sum(p * math.log(p) for p in (2 / 3, 1 / 3))


def _canonical_ast(value):
    """Ignore empty AST fields added by different Python parser versions."""

    if isinstance(value, ast.AST):
        return {
            "type": type(value).__name__,
            "fields": {
                name: _canonical_ast(field)
                for name, field in ast.iter_fields(value)
                if field is not None and field != []
            },
        }
    if isinstance(value, list):
        return [_canonical_ast(item) for item in value]
    return value


class _Array:
    """Only division and flattening needed by the original MI function."""

    def __init__(self, values):
        self.values = tuple(
            tuple(value) if isinstance(value, (list, tuple)) else value
            for value in values
        )

    def __truediv__(self, denominator):
        return _Array(
            [tuple(item / denominator for item in value)
             if isinstance(value, tuple) else value / denominator
             for value in self.values]
        )

    def ravel(self):
        return _Array(
            [item for value in self.values
             for item in (value if isinstance(value, tuple) else (value,))]
        )


class _Boundaries:
    """Fail on import-time work; explicitly enable small deterministic flows."""

    def __init__(self, *, allow_flow=False):
        self.allow_flow = allow_flow
        self.process_calls = []
        self.plot_calls = []
        self.science_calls = []
        self.processes = []
        self.queues = []

        multiprocessing = types.ModuleType("multiprocessing")
        multiprocessing.Manager = self.manager
        multiprocessing.Process = self.process
        multiprocessing.Queue = self.queue

        numpy = types.ModuleType("numpy")
        numpy.histogram = self.histogram
        numpy.histogram2d = self.histogram2d
        numpy.sum = self.array_sum
        numpy.log2 = self.unexpected_log2

        scipy = types.ModuleType("scipy")
        scipy.__path__ = []
        stats = types.ModuleType("scipy.stats")
        stats.entropy = self.entropy
        scipy.stats = stats

        matplotlib = types.ModuleType("matplotlib")
        matplotlib.__path__ = []
        pyplot = types.ModuleType("matplotlib.pyplot")
        for name in ("figure", "plot", "xlabel", "ylabel", "title", "legend", "grid", "show"):
            setattr(pyplot, name, self.plotter(name))
        matplotlib.pyplot = pyplot

        self.modules = {
            "multiprocessing": multiprocessing,
            "numpy": numpy,
            "scipy": scipy,
            "scipy.stats": stats,
            "matplotlib": matplotlib,
            "matplotlib.pyplot": pyplot,
        }

    def require_flow(self, boundary):
        if not self.allow_flow:
            raise AssertionError(f"Unexpected {boundary} call while importing {SOURCE_PATH}")

    def manager(self):
        self.process_calls.append(("Manager",))
        self.require_flow("Manager")
        boundaries = self

        class ManagerDouble:
            def list(self):
                boundaries.process_calls.append(("manager.list",))
                return []

        return ManagerDouble()

    def queue(self):
        self.process_calls.append(("Queue",))
        self.require_flow("Queue")
        queue = object()
        self.queues.append(queue)
        return queue

    def process(self, *, target, args):
        self.process_calls.append(("Process", target, args))
        self.require_flow("Process")
        boundaries = self

        class ProcessDouble:
            def start(self):
                boundaries.process_calls.append(("start", args[4]))
                fixture = {"Agent1": LOG1, "Agent2": LOG2}[args[4]]
                args[3].extend(fixture)

            def join(self):
                boundaries.process_calls.append(("join", args[4]))

        process = ProcessDouble()
        self.processes.append((target, args, process))
        return process

    def histogram(self, behavior, bins):
        self.science_calls.append(("histogram", behavior, bins))
        self.require_flow("numpy.histogram")
        if bins != 10:
            raise AssertionError(f"Unexpected histogram bins: {bins!r}")
        if behavior == BEHAVIOR1:
            counts = [2] + [0] * 8 + [1]
        elif behavior == BEHAVIOR2:
            counts = [1] + [0] * 8 + [2]
        else:
            raise AssertionError(f"Unexpected histogram data: {behavior!r}")
        return _Array(counts), None

    def histogram2d(self, behavior1, behavior2, bins):
        self.science_calls.append(("histogram2d", behavior1, behavior2, bins))
        self.require_flow("numpy.histogram2d")
        if (behavior1, behavior2, bins) != (BEHAVIOR1, BEHAVIOR2, 10):
            raise AssertionError("Unexpected joint histogram data or bins")
        counts = [[0] * 10 for _ in range(10)]
        counts[0][9] = 2
        counts[9][0] = 1
        return _Array(counts), None, None

    def array_sum(self, values):
        self.require_flow("numpy.sum")
        return sum(values.ravel().values)

    def entropy(self, values):
        self.require_flow("scipy.stats.entropy")
        values = values.ravel().values
        self.science_calls.append(("entropy", values))
        return -sum(value * math.log(value) for value in values if value)

    def unexpected_log2(self, *args, **kwargs):
        raise AssertionError("A worker's channel-capacity calculation must not run")

    def plotter(self, name):
        def record(*args, **kwargs):
            self.plot_calls.append((name, args, kwargs))
            self.require_flow(f"pyplot.{name}")

        return record


def _load_module(boundaries, name):
    spec = importlib.util.spec_from_file_location(name, SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load source: {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {**boundaries.modules, name: module}):
        spec.loader.exec_module(module)
    return module


class ImportSafetyTests(unittest.TestCase):
    def assert_original_plot(self, boundaries):
        self.assertEqual(
            boundaries.plot_calls,
            [
                ("figure", (), {"figsize": (12, 6)}),
                ("plot", (TIME1, BEHAVIOR1), {"label": "Agent 1", "marker": "o"}),
                ("plot", (TIME2, BEHAVIOR2), {"label": "Agent 2", "marker": "x"}),
                ("xlabel", ("Time Step",), {}),
                ("ylabel", ("Behavior Level",), {}),
                ("title", (
                    f"Complete Experiment Considering All Aspects (Mutual Information: {EXPECTED_MI:.2f})",
                ), {}),
                ("legend", (), {}),
                ("grid", (True,), {}),
                ("show", (), {}),
            ],
        )

    def assert_original_mi_inputs(self, boundaries):
        self.assertEqual(
            boundaries.science_calls[:3],
            [
                ("histogram", BEHAVIOR1, 10),
                ("histogram", BEHAVIOR2, 10),
                ("histogram2d", BEHAVIOR1, BEHAVIOR2, 10),
            ],
        )
        entropy_calls = boundaries.science_calls[3:]
        self.assertEqual(len(entropy_calls), 3)
        self.assertTrue(all(call[0] == "entropy" for call in entropy_calls))

    def assert_safe_import(self, name):
        boundaries = _Boundaries()
        module = _load_module(boundaries, name)
        self.assertEqual(module.__name__, name)
        self.assertEqual(boundaries.process_calls, [])
        self.assertEqual(boundaries.plot_calls, [])
        self.assertEqual(boundaries.science_calls, [])
        self.assertTrue(callable(getattr(module, "main", None)), "Source must define main")

    def test_import_experiment_is_safe(self):
        self.assert_safe_import("experiment")

    def test_import_mp_main_is_safe(self):
        self.assert_safe_import("__mp_main__")

    def test_main_runs_original_flow_once(self):
        boundaries = _Boundaries()
        module = _load_module(boundaries, "experiment")
        boundaries.allow_flow = True
        with mock.patch.object(
            module, "run_complete_experiment", return_value=(list(LOG1), list(LOG2))
        ) as run_experiment, mock.patch.object(
            module, "mutual_information", wraps=module.mutual_information
        ) as calculate_mi:
            module.main()
        run_experiment.assert_called_once_with(100, 0.1)
        calculate_mi.assert_called_once_with(BEHAVIOR1, BEHAVIOR2)
        self.assertEqual(boundaries.process_calls, [])
        self.assert_original_mi_inputs(boundaries)
        self.assert_original_plot(boundaries)

    def test_script_guard_runs_main_once(self):
        boundaries = _Boundaries(allow_flow=True)
        function_calls = []
        previous_profile = sys.getprofile()

        def record_calls(frame, event, argument):
            if event == "call" and Path(frame.f_code.co_filename).resolve() == SOURCE_PATH:
                if frame.f_code.co_name in ("main", "run_complete_experiment", "mutual_information"):
                    function_calls.append((frame.f_code.co_name, dict(frame.f_locals)))

        with mock.patch.dict(sys.modules, boundaries.modules):
            try:
                sys.setprofile(record_calls)
                namespace = runpy.run_path(str(SOURCE_PATH), run_name="__main__")
            finally:
                sys.setprofile(previous_profile)

        self.assertTrue(callable(namespace.get("main")), "Script must define main")
        self.assertEqual([name for name, _ in function_calls], [
            "main", "run_complete_experiment", "mutual_information"
        ])
        self.assertEqual(function_calls[1][1], {"iterations": 100, "noise_level": 0.1})
        self.assertEqual(function_calls[2][1], {
            "behavior1": BEHAVIOR1, "behavior2": BEHAVIOR2, "bins": 10
        })
        self.assertEqual([call[0] for call in boundaries.process_calls], [
            "Manager", "manager.list", "manager.list", "Queue", "Queue",
            "Process", "Process", "start", "start", "join", "join"
        ])
        self.assertEqual(len(boundaries.queues), 2)
        self.assertEqual(len(boundaries.processes), 2)
        queue1, queue2 = boundaries.queues
        for index, (target, args, _) in enumerate(boundaries.processes):
            with self.subTest(agent=index + 1):
                self.assertIs(target, namespace["complete_agent"])
                self.assertIs(args[0], (queue1, queue2)[index])
                self.assertIs(args[1], (queue2, queue1)[index])
                self.assertEqual(args[2], 100)
                self.assertEqual(args[3], (LOG1, LOG2)[index])
                self.assertEqual(args[4:], (f"Agent{index + 1}", 0.1, 0.1))
        self.assertEqual(boundaries.process_calls[-4:], [
            ("start", "Agent1"), ("start", "Agent2"),
            ("join", "Agent1"), ("join", "Agent2")
        ])
        self.assert_original_mi_inputs(boundaries)
        self.assert_original_plot(boundaries)

    def test_scientific_function_asts_match_fixed_baseline(self):
        tree = ast.parse(SOURCE_PATH.read_bytes(), filename=str(SOURCE_PATH))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in SCIENTIFIC_AST_SHA256]
        self.assertEqual(len(nodes), 5)
        self.assertEqual({node.name for node in nodes}, set(SCIENTIFIC_AST_SHA256))
        for node in nodes:
            with self.subTest(function=node.name):
                canonical = json.dumps(
                    _canonical_ast(node), sort_keys=True, separators=(",", ":")
                ).encode()
                self.assertEqual(hashlib.sha256(canonical).hexdigest(),
                                 SCIENTIFIC_AST_SHA256[node.name])


if __name__ == "__main__":
    unittest.main()

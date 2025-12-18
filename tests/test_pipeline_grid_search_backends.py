"""Tests for backend abstraction in pipeline grid search."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from phenotypic.util._pipeline_grid_search import (
    _execute_parallel_tasks,
    _create_submitit_executor,
    _validate_save_tiff_params,
)


def _test_multiply(x, y):
    """Module-level test function for pickling tests."""
    return x * y


def _test_compute(x):
    """Module-level test function for pickling tests."""
    return x ** 2 + x


def _test_identity(x):
    """Module-level test function for identity operations."""
    return x


class MockJob:
    """Mock submitit job for testing."""

    def __init__(self, func, *args, **kwargs):
        """Initialize mock job."""
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self.job_id = "test_job_12345"

    def result(self):
        """Execute function immediately and return result."""
        return self._func(*self._args, **self._kwargs)


class MockAutoExecutor:
    """Mock submitit AutoExecutor for testing."""

    def __init__(self, folder):
        """Initialize mock executor."""
        self.folder = folder
        self.params = {}

    def update_parameters(self, **kwargs):
        """Update parameters."""
        self.params.update(kwargs)

    def submit(self, func, *args, **kwargs):
        """Submit job (execute immediately for testing)."""
        return MockJob(func, *args, **kwargs)


@pytest.fixture
def mock_submitit(monkeypatch):
    """Mock submitit module for testing without SLURM."""

    # Create mock module
    mock_module = type("submitit", (), {"AutoExecutor": MockAutoExecutor})()

    # Inject into sys.modules
    monkeypatch.setitem(sys.modules, "submitit", mock_module)

    return mock_module


class TestExecuteParallelTasks:
    """Tests for _execute_parallel_tasks function."""

    def test_joblib_backend_basic(self):
        """Test joblib backend with simple function."""

        def square(x):
            return x * x

        task_args = [(2,), (3,), (4,)]

        results = _execute_parallel_tasks(
            func=square,
            task_args=task_args,
            backend="joblib",
            n_jobs=1,  # Use serial execution for testing
        )

        assert results == [4, 9, 16]

    def test_joblib_backend_preserves_order(self):
        """Test that joblib backend returns results in input order."""

        def add(x, y):
            return x + y

        task_args = [(1, 10), (2, 20), (3, 30)]

        results = _execute_parallel_tasks(
            func=add,
            task_args=task_args,
            backend="joblib",
            n_jobs=1,
        )

        assert results == [11, 22, 33]

    def test_submitit_backend_basic(self, mock_submitit):
        """Test submitit backend with mock SLURM."""

        task_args = [(2, 3), (4, 5), (6, 7)]

        results = _execute_parallel_tasks(
            func=_test_multiply,
            task_args=task_args,
            backend="submitit",
            slurm_params={"folder": "./test_logs"},
        )

        assert results == [6, 20, 42]

    def test_submitit_backend_preserves_order(self, mock_submitit):
        """Test that submitit backend returns results in input order."""

        task_args = [(1,), (2,), (3,), (4,)]

        results = _execute_parallel_tasks(
            func=_test_compute,
            task_args=task_args,
            backend="submitit",
            slurm_params={"folder": "./test_logs"},
        )

        assert results == [2, 6, 12, 20]

    def test_unpicklable_function_raises_error(self, mock_submitit):
        """Test that unpicklable functions raise ValueError with submitit."""

        # Lambda functions are not picklable
        func = lambda x: x * 2  # noqa: E731

        task_args = [(1,), (2,)]

        with pytest.raises(ValueError, match="not picklable"):
            _execute_parallel_tasks(
                func=func,
                task_args=task_args,
                backend="submitit",
            )

    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError."""

        def dummy(x):
            return x

        with pytest.raises(ValueError, match="Unknown backend"):
            _execute_parallel_tasks(
                func=dummy,
                task_args=[(1,)],
                backend="invalid_backend",
            )

    def test_joblib_with_n_jobs_parameter(self):
        """Test joblib backend respects n_jobs parameter."""

        def identity(x):
            return x

        task_args = [(i,) for i in range(4)]

        # Should work with different n_jobs values
        results_n1 = _execute_parallel_tasks(
            func=identity,
            task_args=task_args,
            backend="joblib",
            n_jobs=1,
        )

        results_n2 = _execute_parallel_tasks(
            func=identity,
            task_args=task_args,
            backend="joblib",
            n_jobs=2,
        )

        # Results should be identical regardless of n_jobs
        assert results_n1 == results_n2 == [0, 1, 2, 3]


class TestCreateSubmititExecutor:
    """Tests for _create_submitit_executor function."""

    def test_default_parameters(self, mock_submitit):
        """Test executor creation with default parameters."""
        executor = _create_submitit_executor()

        assert hasattr(executor, "folder")
        assert hasattr(executor, "params")

    def test_custom_slurm_params(self, mock_submitit):
        """Test executor creation with custom SLURM parameters."""
        custom_params = {
            "folder": "./custom_logs",
            "timeout_min": 120,
            "mem_gb": 32,
            "cpus_per_task": 4,
            "slurm_partition": "gpu",
        }

        executor = _create_submitit_executor(slurm_params=custom_params)

        assert executor.folder == "./custom_logs"
        # Check that custom params were applied
        assert executor.params.get("timeout_min") == 120
        assert executor.params.get("mem_gb") == 32

    def test_parameter_merging(self, mock_submitit):
        """Test that user params override defaults."""
        custom_params = {
            "timeout_min": 999,  # Override default
            "custom_param": "test",  # New param
        }

        executor = _create_submitit_executor(slurm_params=custom_params)

        assert executor.params.get("timeout_min") == 999
        assert executor.params.get("custom_param") == "test"
        # Check defaults are still present if not overridden
        assert executor.params.get("mem_gb") == 16  # Default value


class TestValidateSaveTiffParams:
    """Tests for _validate_save_tiff_params function."""

    def test_create_trial_view_requires_save_dir(self):
        """Test that create_trial_view=True without save_tiff_dir raises error."""
        with pytest.raises(ValueError, match="create_trial_view=True requires save_tiff_dir"):
            _validate_save_tiff_params(None, True, "joblib")

    def test_invalid_backend_raises_error(self, tmp_path):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="backend must be"):
            _validate_save_tiff_params(str(tmp_path), False, "invalid")

    def test_valid_joblib_parameters(self, tmp_path):
        """Test that valid parameters pass validation."""
        # Should not raise
        _validate_save_tiff_params(str(tmp_path), False, "joblib")

    def test_valid_submitit_parameters(self, tmp_path):
        """Test that valid submitit parameters pass validation."""
        # Should not raise
        _validate_save_tiff_params(str(tmp_path), False, "submitit")

    def test_directory_creation(self, tmp_path):
        """Test that directory is created if it doesn't exist."""
        new_dir = str(tmp_path / "new" / "nested" / "dir")
        _validate_save_tiff_params(new_dir, False, "joblib")

        assert Path(new_dir).exists()

    def test_none_save_dir_is_valid(self):
        """Test that save_tiff_dir=None is valid (napari mode)."""
        # Should not raise
        _validate_save_tiff_params(None, False, "joblib")


class TestBackendIntegration:
    """Integration tests for backend usage."""

    def test_joblib_backend_with_empty_tasks(self):
        """Test joblib backend with no tasks."""

        def dummy(x):
            return x

        results = _execute_parallel_tasks(
            func=dummy,
            task_args=[],
            backend="joblib",
        )

        assert results == []

    def test_submitit_backend_with_empty_tasks(self, mock_submitit):
        """Test submitit backend with no tasks."""

        results = _execute_parallel_tasks(
            func=_test_identity,
            task_args=[],
            backend="submitit",
        )

        assert results == []

    def test_submitit_backend_not_installed(self, monkeypatch):
        """Test helpful error message when submitit not installed."""

        # Remove submitit from sys.modules to simulate not installed
        if "submitit" in sys.modules:
            del sys.modules["submitit"]

        # Mock import to fail for submitit
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "submitit":
                raise ImportError("No module named 'submitit'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="submitit.*not installed"):
            _validate_save_tiff_params("./test", False, "submitit")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import torch
import pytest


def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require real data",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "large_model: marks tests that load large models (deselect with '-m \"not large_model\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests (run with --run-integration)")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        # --run-integration given: do not skip integration tests
        return
    skip_integration = pytest.mark.skip(reason="Requires real data - use pytest --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Automatically run after each test to clean up resources.
    This helps prevent file descriptor leaks when running many tests.
    """
    yield  # Run the test

    # Clean up after the test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection to close any lingering file descriptors
    gc.collect()


@pytest.fixture(autouse=True, scope="session")
def increase_file_limit():
    """
    Attempt to increase file descriptor limit at the session level.
    This runs once at the start of the test session.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Try to increase to 4096, but don't exceed hard limit
        new_soft = min(4096, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"\n✓ Increased file descriptor limit: {soft} -> {new_soft}")
    except Exception as e:
        print(f"\n⚠ Could not increase file descriptor limit: {e}")
        print("  This may cause 'Too many open files' errors with many tests")

    yield

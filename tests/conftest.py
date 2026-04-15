from __future__ import annotations

import pytest
import torch

from tests._smoke_helpers import SmokeWorkspace, build_smoke_workspace


@pytest.fixture(autouse=True)
def deterministic_torch_state() -> None:
    torch.manual_seed(7)


@pytest.fixture
def smoke_workspace(tmp_path) -> SmokeWorkspace:
    return build_smoke_workspace(tmp_path)


@pytest.fixture
def smoke_config(smoke_workspace: SmokeWorkspace) -> dict:
    return smoke_workspace.build_config()

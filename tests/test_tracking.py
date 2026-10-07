from unittest.mock import Mock

import pytest
import torch
import torchelie.callbacks as tcb
import trackio

from papote.train import LogCtxLoss
from papote.tracking import run_with_tracking


def test_context_loss_uses_torchelie_trackio_conversion():
    run = Mock(project="papote")
    logger = tcb.TrackioLogger(trackio_project="papote", run=run, prefix="train/")
    logger.log(0, {"loss_at_pos": LogCtxLoss(torch.tensor([2.0, 1.0]))})
    table = run.log.call_args.args[0]["train/loss_at_pos"]
    assert table.data == [
        {"position": 0, "value": 2.0},
        {"position": 1, "value": 1.0},
    ]
    assert run.log.call_args.kwargs == {"step": 0}


@pytest.mark.parametrize("fails", [False, True])
def test_named_run_is_shared_and_finished(monkeypatch, fails):
    run = Mock(project="papote")
    init = Mock(return_value=run)
    finish = Mock()
    monkeypatch.setattr(trackio, "init", init)
    monkeypatch.setattr(trackio, "finish", finish)
    recipe = Mock()
    if fails:
        recipe.run.side_effect = RuntimeError("training failed")
    kwargs = dict(rank=0, project="papote", name="tiny", config={"lr": 0.001})
    if fails:
        with pytest.raises(RuntimeError, match="training failed"):
            run_with_tracking(recipe, 1, **kwargs)
    else:
        assert run_with_tracking(recipe, 1, **kwargs) is recipe.run.return_value
    init.assert_called_once_with(
        project="papote", name="tiny", config={"lr": 0.001}, embed=False
    )
    training_logger = recipe.callbacks.add_epilogue.call_args.args[0]
    test_logger = recipe.test_loop.callbacks.add_epilogue.call_args.args[0]
    assert training_logger.run is test_logger.run is run
    finish.assert_called_once()


@pytest.mark.parametrize("rank,project", [(1, "papote"), (0, None)])
def test_disabled_or_non_primary_training_does_not_initialize(monkeypatch, rank, project):
    init = Mock()
    monkeypatch.setattr(trackio, "init", init)
    recipe = Mock()
    run_with_tracking(recipe, 1, rank=rank, project=project, name=None, config={})
    recipe.run.assert_called_once_with(1)
    recipe.callbacks.add_epilogue.assert_not_called()
    init.assert_not_called()

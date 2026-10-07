"""Own the Trackio run used by Torchelie's training and evaluation loggers."""

import trackio
import torchelie.callbacks as tcb


def run_with_tracking(recipe, epochs, *, rank, project, name, config):
    """Create a named run on rank zero and flush it even if training fails."""
    if rank != 0 or project is None:
        return recipe.run(epochs)

    run = trackio.init(project=project, name=name, config=config, embed=False)
    try:
        recipe.callbacks.add_epilogue(
            tcb.TrackioLogger(
                trackio_project=project, run=run, prefix="train/", log_every=10
            )
        )
        recipe.test_loop.callbacks.add_epilogue(
            tcb.TrackioLogger(
                trackio_project=project, run=run, prefix="test/", log_every=-1
            )
        )
        return recipe.run(epochs)
    finally:
        trackio.finish()

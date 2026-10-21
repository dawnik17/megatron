"""What the generic check needs from a model. A model family subclasses Config in config/<family>.py and registers
it in config/__init__.py."""

import runpy
import sys
from pathlib import Path

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import Float16Module

BRIDGE = Path(__file__).resolve().parents[3]  # bridge/


class Config:
    """recipe: Megatron-Bridge's pretrain config function. changes: (owner, attribute) of each OOTB function a change
    replaces; ours is what train.main() puts in its place, or what config_flag_changes() returns. train_dir: the
    folder of the train.py under test. hf: the checkpoint and tokenizer."""

    def __init__(self, recipe, changes, train_dir, hf, seq_length=1024):
        self.recipe, self.changes, self.train_dir = recipe, changes, train_dir
        self.hf, self.seq_length = hf, seq_length

    # Get "ootb" and "ours" functions
    def setup(self):
        """One-rank Megatron; the OOTB functions and train.py's replacements; the model as training builds it (random
        weights, TP 1) with train.py's patches applied. main() loads the pretrained weights after the gradient check.
        """
        sys.path.insert(0, str(self.train_dir))
        import train  # the code under test

        torch.distributed.init_process_group(
            "nccl", init_method="tcp://127.0.0.1:29791", world_size=1, rank=0
        )
        parallel_state.initialize_model_parallel(1, 1)
        tensor_parallel.model_parallel_cuda_manual_seed(1234)

        ootb = {key: getattr(*key) for key in self.changes}
        # train.main() ends by starting training; stop it there
        runpy.run_path = lambda *args, **kwargs: None
        train.main()
        ours = {key: getattr(*key) for key in self.changes}

        cfg = self.recipe()
        get_mixed_precision_config(cfg.mixed_precision).setup(cfg.model)
        provider = cfg.model
        provider.tensor_model_parallel_size, provider.seq_length = 1, self.seq_length
        provider.finalize()
        model = provider.provide().cuda()
        return provider, model, Float16Module(provider, model), ootb, ours

    def load_model(self, model):
        """Load the pretrained weights into model."""
        AutoBridge.from_hf_pretrained(self.hf).load_hf_weights([model])

    def config_flag_changes(self, provider, ootb):
        """{key: ours} for the changes that are a config flag, not a patched function. They take keyword arguments
        only, as compare() calls them, so they can't go in the model: main() adds them after the gradient check.
        """
        return {}

import os

# Compatibility wrapper to handle version conflicts between PyTorch and Transformers.
import torch.utils._pytree as _pytree


# Wrapper function for parameter handling.
def _register_pytree_wrapper(*args, **kwargs):
    # Remove the 'serialized_type_name' parameter if it is present.
    kwargs.pop('serialized_type_name', None)
    return _pytree._register_pytree_node(*args, **kwargs)


if not hasattr(_pytree, "register_pytree_node"):
    setattr(_pytree, "register_pytree_node", _register_pytree_wrapper)

from .hook import HookManager
from transformers import TrainerCallback


class HookHFCallback(TrainerCallback):
    """
    Callback for the HuggingFace Trainer.
    Collects parameter-level hook statistics: grad_* and param_* per parameter.
    
    Args:
        model: PyTorch model to attach hooks to.
        out_dir: Directory to save hook statistics CSV files.
        reduce_fn: Optional tensor reduction function.
    """

    def __init__(self, model, out_dir="artifacts", reduce_fn=None, capture_every_n_steps: int = 1, include_quantiles: bool = True):
        self.hm = HookManager(
            reduce_fn=reduce_fn,
            store_on_cpu=True,
            capture_every_n_steps=capture_every_n_steps,
            include_quantiles=include_quantiles,
        )
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Parameter-level hooks: collect grad_* and param_* statistics per parameter
        print("[HookHFCallback] Registering parameter-level hooks...")
        hook_count = 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                prefix = name if name else "root"
                self.hm.register_parameters(module, prefix=prefix)
                module_params = sum(1 for _ in module.named_parameters(recurse=False))
                hook_count += module_params
        print(f"[HookHFCallback] Registered parameter-level hooks for {hook_count} parameters")

    def on_step_begin(self, args, state, control, **kwargs):
        # Start a new "step" at the beginning of every batch.
        self.hm.next_step()

    def on_epoch_end(self, args, state, control, **kwargs):
        # Save at the end of each epoch.
        df = self.hm.to_dataframe()
        out_csv = os.path.join(self.out_dir, f"hook_stats_epoch{int(state.epoch)}.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[HookHFCallback] Saved: {out_csv}")

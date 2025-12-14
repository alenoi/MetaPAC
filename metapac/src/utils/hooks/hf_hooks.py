import os

# Kompatibilitási wrapper a PyTorch és Transformers közötti verziókonfliktus kezelésére
import torch.utils._pytree as _pytree


# Wrapper függvény a paraméterek kezelésére
def _register_pytree_wrapper(*args, **kwargs):
    # Eltávolítjuk a 'serialized_type_name' paramétert, ha jelen van
    kwargs.pop('serialized_type_name', None)
    return _pytree._register_pytree_node(*args, **kwargs)


if not hasattr(_pytree, "register_pytree_node"):
    setattr(_pytree, "register_pytree_node", _register_pytree_wrapper)

from .hook import HookManager
from transformers import TrainerCallback


class HookHFCallback(TrainerCallback):
    """
    HuggingFace Trainer-hez való callback.
    Parameter-level hook statistics gyűjtése: grad_* és param_* per parameter.
    
    Args:
        model: PyTorch model to attach hooks to.
        out_dir: Directory to save hook statistics CSV files.
        reduce_fn: Optional tensor reduction function.
    """

    def __init__(self, model, out_dir="artifacts", reduce_fn=None):
        self.hm = HookManager(reduce_fn=reduce_fn, store_on_cpu=True)
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
        # minden batch elején új "step"
        self.hm.next_step()

    def on_epoch_end(self, args, state, control, **kwargs):
        # epoch végén mentés
        df = self.hm.to_dataframe()
        out_csv = os.path.join(self.out_dir, f"hook_stats_epoch{int(state.epoch)}.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[HookHFCallback] Mentve: {out_csv}")

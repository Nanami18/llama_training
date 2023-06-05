import fairscale.nn.model_parallel.initialize as fs_init
import torch
from fairscale.nn.wrap.auto_wrap import wrap, enable_wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from llama import Transformer
import torch.distributed as dist
import os

def get_model_parallel_size() -> int:
    if torch.distributed.is_initialized() and fs_init.model_parallel_is_initialized():
        return fs_init.get_model_parallel_world_size()
    else:
        return 1
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def init_distributed(num_tasks, global_rank):
    torch.distributed.init_process_group(
        backend="nccl",
        init_method= "env://",
        world_size=num_tasks,
        rank=global_rank,
    )


def build_model(
    model_args,
    dtype = torch.float32,
    fp32_reduce_scatter = "all",
    reshard_after_forward = True,
):
    assert model_args.vocab_size > 0
    print("Starting build of model...")
    fsdp_cfg = {
        "process_group": fs_init.get_data_parallel_group(),
        "process_group_reduce_scatter": fs_init.get_data_parallel_group(),
        "compute_dtype": get_torch_dtype(dtype),
        "state_dict_device": torch.device("cpu"),
        "mixed_precision": True,
        "flatten_parameters": True,
        "fp32_reduce_scatter": fp32_reduce_scatter == "all",
        "reshard_after_forward": reshard_after_forward,
    }
    torch.set_default_tensor_type(torch.FloatTensor)
    with enable_wrap(wrapper_cls=FSDP, **fsdp_cfg):
        model = Transformer(model_args)
        model = wrap(
            model.cuda(),
            fp32_reduce_scatter=fp32_reduce_scatter in ("all", "only_input_output"),
        )
        model.train()
    print("Done with build of model...")
    return model


def get_torch_dtype(dtype: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
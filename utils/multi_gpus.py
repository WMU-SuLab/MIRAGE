import torch
import torch.distributed as dist


def init_distributed_mode(dist_backend, world_size, rank):
    dist.init_process_group(backend=dist_backend, init_method='env://',
                            world_size=world_size, rank=rank)
    dist.barrier()


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_available()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value

def barrier():
    dist.barrier()

def synchronize(device):
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

def cleanup():
    dist.destroy_process_group()

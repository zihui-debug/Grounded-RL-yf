import json
import logging
import torch
from accelerate import Accelerator

def gather_accelerator_results(results, accelerator: Accelerator):
    """
    Gather a list of Python dicts (`results`) across all processes using
    accelerator.gather(). Since accelerator.gather() must operate on
    tensors, we convert our list of dicts into JSON lines, encode them
    to bytes, and gather those byte tensors. Only rank=0 reassembles them.

    Args:
        results (list[dict]): The local process' list of dictionaries.
        accelerator (Accelerator): The huggingface Accelerate accelerator.

    Returns:
        list[dict]: On rank=0, the full combined list from all ranks.
                    On other ranks, an empty list.
    """
    device = accelerator.device

    # 1) Convert local results -> JSON lines string
    if len(results) == 0:
        # If this rank has no results, we create an empty "byte" tensor
        local_str = ""
    else:
        lines = [json.dumps(r) for r in results]
        local_str = "\n".join(lines)

    encoded = local_str.encode("utf-8")
    local_bytes = torch.tensor(list(encoded), dtype=torch.uint8, device=device)

    # 2) We'll also gather the length of each rank's string
    local_length = torch.tensor([local_bytes.shape[0]], dtype=torch.long, device=device)
    all_lengths = accelerator.gather(local_length)  # shape [world_size]
    max_length = all_lengths.max().item()

    # 3) Pad each rank's bytes to the same length
    padded = torch.zeros((max_length,), dtype=torch.uint8, device=device)
    padded[: local_bytes.shape[0]] = local_bytes

    # 4) Gather across ranks. The result will have shape [world_size * max_length]
    gathered = accelerator.gather(padded)

    # 5) Reshape gathered tensor into [world_size, max_length]
    world_size = accelerator.num_processes
    gathered = gathered.view(world_size, max_length)

    # 6) Only the main process reassembles them
    if accelerator.is_main_process:
        all_dicts = []
        gathered = gathered.cpu()
        for i in range(world_size):
            length_i = all_lengths[i].item()
            if length_i == 0:
                # That rank had no results
                continue
            raw_bytes = gathered[i, :length_i].numpy().tobytes()
            chunk_str = raw_bytes.decode("utf-8")
            # chunk_str is the JSON lines from rank i
            lines = chunk_str.split("\n")
            for line in lines:
                all_dicts.append(json.loads(line))
        return all_dicts
    else:
        # Return empty list on non-main processes
        return []
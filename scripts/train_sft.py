import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from nanollm.dataloader import LemonmindDataset, lemonmind_dataloader
from nanollm.gpt import GPTConfig, GPT, Block
from nanollm.tokenizer import get_tokenizer


def save_checkpoint(model, optimizer, step, config, vocab_size, save_path, ddp, master, buffer_round):
    if ddp:
        model_state, optimizer_state = get_state_dict(
            model,
            optimizer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
    else:
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()

    if not master:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optimizer_state,
            "config": config,
            "vocab_size": vocab_size,
            "buffer_round": buffer_round,
        },
        f"{save_path}.tmp"
    )
    os.replace(f'{save_path}.tmp', f'{save_path}')

    logger.success(f"Saved checkpoint to {save_path}")


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        decay = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


def main():
    # Distributed init
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)

        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(device)
        )
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    if master:
        logger.info(
            f"CPUs: {torch.cuda.device_count()}  |  World size: {world_size}  |  Device: {device}"
        )

    # Tokenizer
    tokenizer = get_tokenizer('/home/oem/ztw/nano-llm/gpt-neox-20b-tokenizer')
    vocab_size = len(tokenizer)

    if master:
        logger.info(f"Tokenizer: EleutherAI/gpt-neox-20b  |  Vocab size: {vocab_size}")

    # Hyperparameters
    max_seq_len = 1024
    micro_batch = 32
    target_tokens = 287_535_104 * 10
    grad_accum = max(1, 256 // (world_size * micro_batch))
    global_batch_tokens = world_size * micro_batch * grad_accum * max_seq_len
    total_steps = target_tokens // global_batch_tokens
    warmup_ratio = 0.01
    warmup_steps = int(warmup_ratio * total_steps)
    lr = 5e-5
    weight_decay = 0.1
    log_steps = 100
    save_steps = 500
    pretrain_model_path = 'model/pretrain/nano-llm-flash.bin'
    save_path = 'model/sft/nano-llm-flash.bin'

    if master:
        logger.info(
            f"max_seq_len={max_seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
            f"global_batch_tokens={global_batch_tokens:,} | total_steps={total_steps:,}"
        )

    # 4.Model
    config = GPTConfig()
    config.vocab_size = vocab_size
    config.max_seq_len = max_seq_len
    config.model_dim = 1024
    config.n_layers = 16
    config.n_heads = 8
    config.n_kv_heads = 4
    config.head_dim = 128

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = GPT(config)
    model.load_state_dict(torch.load(pretrain_model_path, map_location="cpu", weights_only=False)["model"])

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
        )
        wrap_policy = ModuleWrapPolicy({Block})
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank
        )
    else:
        model = model.to(device=device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device else nullcontext()
        )

    amp_ctx = nullcontext() if ddp else amp_ctx

    if master:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {total_params:,}  |  AMP dtype: {amp_dtype}")

    # Loss | Optimizer
    loss_func = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), fused=True)

    # Resume from latest checkpoint (if any)
    start_step = 0
    buffer_round = 0
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        if ddp:
            set_state_dict(
                model,
                optimizer,
                model_state_dict=ckpt["model"],
                optim_state_dict=ckpt["optimizer"],
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True
                )
            )
        else:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        buffer_round = ckpt.get("buffer_round", 0)
        if master:
            logger.info(f"Resuming from step {start_step:,}")

    # Dataset | Dataloader
    train_dataset = LemonmindDataset(
        # '/root/autodl-tmp/SmolTalk-GSM8K',
        '/home/oem/ztw/SmolTalk-GSM8K',
        buffer_size=10
    )
    train_dataset.load_data(buffer_round)
    train_loader, train_sampler = lemonmind_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len+1,
        batch_size=micro_batch,
        ddp=ddp,
        rank=rank,
        world_size=world_size
    )

    # train!
    model.train()
    start_time = time.perf_counter()
    step = start_step
    micro_step_in_accum = 0

    while step < total_steps:
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(buffer_round)

        if master:
            logger.info(
                f"buffer_round={buffer_round}  |  "
                f"train_dataset length={len(train_dataset)}  |  "
                f"train_loader length={len(train_loader)}"
            )

        for _, (x, y, loss_mask) in enumerate(train_loader):
            if step >= total_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)

            cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr*0.1)
            for p in optimizer.param_groups:
                p["lr"] = cur_lr

            is_accum_last = micro_step_in_accum == grad_accum - 1

            sync_ctx = (
                nullcontext()
                if (not ddp or is_accum_last)
                else model.no_sync()
            )

            with sync_ctx, amp_ctx:
                logits = model(x)
                loss = loss_func(
                    logits.view(-1, logits.shape[-1]),
                    y.view(-1)
                )
                loss_mask = loss_mask.view(-1)
                loss = loss * loss_mask
                loss = loss.sum() / loss_mask.sum().clamp_min(1.0)
                loss = loss / grad_accum
                loss.backward()

            micro_step_in_accum += 1

            if micro_step_in_accum < grad_accum:
                continue

            if ddp:
                grad_norm = model.clip_grad_norm_(1.0)
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            micro_step_in_accum = 0

            if master and step % log_steps == 0:
                elapsed = time.perf_counter() - start_time
                tokens_per_sec = global_batch_tokens * log_steps / elapsed
                tokens_seen = step * global_batch_tokens
                logger.info(
                    f"step: {step:6d}/{total_steps}  |  "
                    f"loss: {loss.item() * grad_accum:.4f}  |  "
                    f"grad_norm: {float(grad_norm):.2f}  |  "
                    f"lr: {cur_lr:.2e}  |  "
                    f"{tokens_per_sec / 1e6:.2f}M token/s |  "
                    f"{tokens_seen / 1e9:.1f}B tokens seen  |  "
                    f"total_time: {elapsed:.2f}s"
                )
                start_time = time.perf_counter()

            if step % save_steps == 0:
                save_checkpoint(
                    model, optimizer, step, config, vocab_size, save_path, ddp, master, buffer_round
                )
                if ddp:
                    dist.barrier(device_ids=[local_rank])

        if step < total_steps:
            buffer_round += 1
            train_dataset.load_data(buffer_round)
            train_loader, train_sampler = lemonmind_dataloader(
                train_dataset, tokenizer, max_seq_len+1, micro_batch, ddp, rank, world_size
            )
            if master:
                logger.info("load new data buffer")

    if step > start_step and step % save_steps != 0:
        save_checkpoint(model, optimizer, step, config, vocab_size, save_path, ddp, master, buffer_round)
        if ddp:
            dist.barrier(device_ids=[local_rank])

    if ddp:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()

    if master:
        logger.success("Training complete!")


if __name__ == '__main__':
    main()
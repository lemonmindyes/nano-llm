import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn

from nanollm.dataloader import TinyStoriesDataset, tiny_stories_dataloader
from nanollm.gpt import GPTConfig, GPT
from nanollm.tokenizer import get_tokenizer


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)

    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

    return min_lr + (max_lr - min_lr) * cosine_decay


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    autocast_ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=dtype)

    batch_size = 32
    max_lr = 5e-4
    min_lr = 5e-5
    epoch = 1
    accumulation_steps = 1
    warmup_ratio = 0.03
    log_steps = 100
    save_steps = 100
    save_path = './'

    tokenizer = get_tokenizer()

    config = GPTConfig()
    config.vocab_size = len(tokenizer)

    model = GPT(config).to(device=device, dtype=dtype)
    model.init_weights()
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    train_dataset = TinyStoriesDataset(
        'D:/think-dataset/TinyStories',
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len + 1,
        split='train',
    )
    train_dataloader = tiny_stories_dataloader(
        train_dataset,
        batch_size=batch_size
    )

    total_steps = epoch * len(train_dataloader) * accumulation_steps
    warmup_steps = math.ceil(warmup_ratio * total_steps)

    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
    opt = torch.optim.Muon(model.parameters(), lr=max_lr)
    loss_func = nn.CrossEntropyLoss(reduction='none')

    try:
        checkpoint = torch.load(f'{save_path}/pretrain1.bin', map_location=device)
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        scaler.load_state_dict(checkpoint['scaler'])
    except:
        checkpoint = {'step': 0}

    step = checkpoint['step']
    while step < total_steps:
        start_time = time.time()
        for _, (x, y, loss_mask) in enumerate(train_dataloader):
            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
            for param_group in opt.param_groups:
                param_group['lr'] = get_lr(step, warmup_steps, total_steps, max_lr, min_lr)

            with autocast_ctx:
                logits = model(x)
                loss = loss_func(
                    logits.view(-1, logits.shape[-1]),
                    y.view(-1)
                )
                loss_mask = loss_mask.view(-1)
                loss = loss * loss_mask
                loss = loss.sum() / loss_mask.sum()
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if (step + 1) % log_steps == 0:
                print(f'step: {step+1}/{total_steps}, loss: {loss.item():.4f}, time: {time.time() - start_time:.2f}s, '
                      f'lr: {opt.param_groups[0]["lr"]:.6f}')

            if (step + 1) % save_steps == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'scaler': scaler.state_dict(),
                    'step': step
                }
                torch.save(checkpoint, 'pretrain1.bin.tmp')
                os.replace('pretrain1.bin.tmp', f'pretrain1.bin')
            step += 1

    checkpoint = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step
    }
    torch.save(checkpoint, 'pretrain1.bin.tmp')
    os.replace('pretrain1.bin.tmp', f'pretrain1.bin')

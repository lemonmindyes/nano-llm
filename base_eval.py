import math

import torch
import torch.nn as nn
from tqdm import tqdm

from nanollm.dataloader import TinyStoriesDataset, tiny_stories_dataloader
from nanollm.gpt import GPTConfig, GPT
from nanollm.tokenizer import get_tokenizer


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = get_tokenizer()
    tiny_stories_dataset = TinyStoriesDataset(
        'D:/think-dataset/TinyStories',
        tokenizer=tokenizer,
        max_seq_len=256 + 1,
        split='val',
    )
    val_dataloader = tiny_stories_dataloader(tiny_stories_dataset, batch_size=64)

    config = GPTConfig()
    config.vocab_size = len(tokenizer)
    model = GPT(config).to(device=device)
    model.load_state_dict(torch.load('pretrain.bin', map_location=device)['model'])

    loss_func = nn.CrossEntropyLoss(reduction='none')
    model.eval()

    total_loss = 0
    total_token = 0
    for i, (x, y, loss_mask) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
        with torch.no_grad():
            logits = model(x)

            loss = loss_func(
                logits.view(-1, logits.shape[-1]),
                y.view(-1)
            ).view_as(loss_mask)

            loss = loss * loss_mask

            total_loss += loss.sum().item()
            total_token += loss_mask.sum().item()

    avg_loss = total_loss / total_token
    ppl = math.exp(avg_loss)

    print(f'avg_loss: {avg_loss:.2f}')
    print(f'ppl: {ppl:.2f}')



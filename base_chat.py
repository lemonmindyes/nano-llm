import pandas as pd

import torch

from nanollm.dataloader import ClimbMixDataset
from nanollm.gpt import GPTConfig, GPT
from nanollm.tokenizer import get_tokenizer


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = get_tokenizer()

    climb_mix_dataset = ClimbMixDataset(
        'D:/think-dataset/climbmix-400b-shuffle',
        buffer_size=1
    )
    print(climb_mix_dataset[1])

    config = GPTConfig()
    config.model_dim = 768
    config.n_layers = 14
    config.n_heads = 12
    config.n_kv_heads = 6
    config.vocab_size = len(tokenizer)
    model = GPT(config).to(device=device)
    model.load_state_dict(torch.load('model/pretrain_2.bin', map_location=device)['model'])

    model.eval()

    prompt = "Question: What is one of the main challenges mentioned in the text that sustainable power solutions aim to address?"
    print(prompt)
    print("=====================")
    x = tokenizer.encode(prompt)

    for o in model.generate(x, max_tokens=200, temperature=0.7, top_k=20, seed=42):
        if o == tokenizer.eos_token_id:
            break
        print(tokenizer.decode(o), end='')

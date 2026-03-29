import os
import random
from functools import partial

import pandas as pd
from torch.utils.data import Dataset, DataLoader


# https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle
class ClimbMixDataset(Dataset):
    def __init__(self, data_path, buffer_size):
        super().__init__()
        self.data_list = [f'{data_path}/{v}' for v in os.listdir(data_path)]
        self.buffer_size = buffer_size
        self.data = []
        self.load_data()

    def load_data(self):
        data_list = random.sample(self.data_list, k=min(self.buffer_size, len(self.data_list)))
        self.data = []
        for path in data_list:
            self.data.extend(pd.read_parquet(path).loc[:, 'text'].tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer, max_seq_len):
    texts = [
        tokenizer.bos_token + b + tokenizer.eos_token
        for b in batch
    ]

    encode = tokenizer(
        texts,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encode['input_ids']
    attention_mask = encode['attention_mask']

    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    loss_mask = attention_mask[:, 1:]

    return x, y, loss_mask


def climb_mix_dataloader(ClimbMixDataset, tokenizer, max_seq_len, batch_size=32):
    return DataLoader(ClimbMixDataset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True,
                      num_workers=4,
                      collate_fn=partial(collate_fn, tokenizer=tokenizer, max_seq_len=max_seq_len)
                      )


if __name__ == '__main__':
    from tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    climb_mix_dataset = ClimbMixDataset(
        'D:/think-dataset/climbmix-400b-shuffle',
        buffer_size=1
    )

    train_dataloader = climb_mix_dataloader(climb_mix_dataset,
                                            tokenizer,
                                            max_seq_len=256+1,
                                            batch_size=32
                                            )

    print(len(train_dataloader))
    for i, (x, y, loss_mask) in enumerate(train_dataloader):
        print(i, x.shape, y.shape, loss_mask.shape)

    climb_mix_dataset.load_data()
    print(len(train_dataloader))
    for i, (x, y, loss_mask) in enumerate(train_dataloader):
        print(i, x.shape, y.shape, loss_mask.shape)
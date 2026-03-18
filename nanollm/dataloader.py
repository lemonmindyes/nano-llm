import os
import random

import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TinyStoriesDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len,
                 split):
        super().__init__()
        self.data_list = [f'{data_path}/{v}' for v in os.listdir(data_path) if v.startswith(split)]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        for path in self.data_list:
            self.data.extend(pd.read_parquet(path).iloc[:, 0].tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.tokenizer.bos_token + self.data[idx] + self.tokenizer.eos_token
        encode = self.tokenizer(
            [text],
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encode['input_ids'][0]
        x = input_ids[:-1]
        y = input_ids[1:]
        attention_mask = encode['attention_mask'][0]
        loss_mask = attention_mask[1:]
        return x, y, loss_mask


def tiny_stories_dataloader(tiny_stories_dataset, batch_size=128):
    return DataLoader(tiny_stories_dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True,
                      num_workers=4,
                      )


if __name__ == '__main__':
    from tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    tiny_stories_dataset = TinyStoriesDataset(
        'D:/think-dataset/TinyStories',
        tokenizer=tokenizer,
        max_seq_len=256 + 1,
        split='train',
    )
    train_dataloader = tiny_stories_dataloader(tiny_stories_dataset, batch_size=256)

    for i, (x, y, loss_mask) in enumerate(train_dataloader):
        print(i, x.shape, y.shape, loss_mask.shape)
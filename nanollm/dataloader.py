import os
import random
from functools import partial
from pathlib import Path

import pandas as pd
import torch
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


def collate_fn_pretrain(batch, tokenizer, max_seq_len):
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
                      collate_fn=partial(collate_fn_pretrain, tokenizer=tokenizer, max_seq_len=max_seq_len)
                      )


class SmolTalkGSM8KDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_list = [str(v) for v in Path(data_path).rglob('*.parquet')]
        self.data = []
        self.load_data()

    def load_data(self):
        self.data = []
        for path in self.data_list:
            if path.split('\\')[-2] == 'GSM8K':
                for _, row in pd.read_parquet(path).iterrows():
                    self.data.append({
                        'messages': [
                            {'role': 'user', 'content': row['question']},
                            {'role': 'assistant', 'content': row['answer']}
                        ]
                    })
            elif path.split('\\')[-2] == 'SmolTalk':
                for _, row in pd.read_parquet(path).iterrows():
                    tmp = row['messages'].tolist()
                    if tmp[-1]['role'] == 'user':
                        tmp = tmp[:-1]
                    self.data.append({
                        'messages': tmp
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_sft(batch, tokenizer, max_seq_len):
    def norm_text(x):
        tmp = tokenizer.bos_token + '\n'
        for v in x:
            if v['role'] == 'system':
                tmp += '<|system_start|>' + '\n' + v['content'] + '\n' + '<|system_end|>' + '\n'
            elif v['role'] == 'user':
                tmp += '<|user_start|>' + '\n' + v['content'] + '\n' + '<|user_end|>' + '\n'
            else:
                tmp += '<|assistant_start|>' + '\n' + v['content'] + '\n' + '<|assistant_end|>' + '\n'
        tmp += tokenizer.eos_token
        return tmp

    def generate_loss_mask(ids):
        assistant_start_idx = tokenizer.convert_tokens_to_ids('<|assistant_start|>')
        assistant_end_idx = tokenizer.convert_tokens_to_ids('<|assistant_end|>')

        loss_mask = torch.zeros_like(ids, dtype=torch.bool)

        for i in range(len(ids)):
            start = torch.where(ids[i] == assistant_start_idx)[0]
            end = torch.where(ids[i] == assistant_end_idx)[0]

            for s, e in zip(start, end):
                loss_mask[i, s+1:e] = True
            if len(start) != len(end):
                loss_mask[i, start[-1]+1:] = True

        return loss_mask

    texts = [
        norm_text(b['messages']) for b in batch
    ]

    encode = tokenizer(
        texts,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encode['input_ids']

    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    loss_mask = generate_loss_mask(input_ids)[:, 1:]

    return x, y, loss_mask


def smoltalk_gsm8k_dataloader(SmolTalkGSM8KDataset, tokenizer, max_seq_len, batch_size=32):
    return DataLoader(SmolTalkGSM8KDataset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True,
                      num_workers=0,
                      collate_fn=partial(collate_fn_sft, tokenizer=tokenizer, max_seq_len=max_seq_len)
                      )


if __name__ == '__main__':
    from tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    climb_mix_dataset = ClimbMixDataset(
        'D:/think-dataset/climbmix-400b-shuffle/train',
        buffer_size=4
    )

    train_dataloader = climb_mix_dataloader(climb_mix_dataset,
                                            tokenizer,
                                            max_seq_len=256+1,
                                            batch_size=16
                                            )

    print(len(train_dataloader))
    for i, (x, y, loss_mask) in enumerate(train_dataloader):
        print(i, x.shape, y.shape, loss_mask.shape)
        break

    climb_mix_dataset.load_data()
    print(len(train_dataloader))
    for i, (x, y, loss_mask) in enumerate(train_dataloader):
        print(i, x.shape, y.shape, loss_mask.shape)
        break

    # from tokenizer import get_tokenizer
    #
    # tokenizer = get_tokenizer()
    # smoltalk_gsm8k_dataset = SmolTalkGSM8KDataset(
    #     'D:/think-dataset/SmolTalk-GSM8K'
    # )
    #
    # train_dataloader = smoltalk_gsm8k_dataloader(smoltalk_gsm8k_dataset,
    #                                              tokenizer,
    #                                              max_seq_len=512+1,
    #                                              batch_size=16
    #                                              )
    # print(len(train_dataloader))
    # for i, (x, y, loss_mask) in enumerate(train_dataloader):
    #     idx = 11
    #     print(tokenizer.decode(y[idx]))
    #     print('========================================')
    #     print(tokenizer.decode(y[idx][loss_mask[idx]]))
    #     break
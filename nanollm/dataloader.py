import os
import random
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle
class ClimbMixDataset(Dataset):
    def __init__(self, data_path, buffer_size=5, seed=42):
        super().__init__()
        self.data_list = [
            f'{data_path}/{v}'
            for v in os.listdir(data_path)
            if v.endswith('.parquet')
        ]
        self.data_list = sorted(self.data_list)

        self.buffer_size = buffer_size
        self.seed = seed
        self.data = []

    def load_data(self, buffer_round=0):
        if len(self.data_list) == 0:
            raise ValueError('No data found!')

        rng = random.Random(self.seed + buffer_round)

        data_list = rng.sample(
            self.data_list,
            k=min(self.buffer_size, len(self.data_list))
        )

        # remove selected files
        selected_set = set(data_list)
        self.data_list = [
            path for path in self.data_list
            if path not in selected_set
        ]

        self.data = []
        for path in data_list:
            self.data.extend(
                pd.read_parquet(path).loc[:, 'text'].tolist()
            )

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


def climb_mix_dataloader(dataset, tokenizer, max_seq_len, batch_size=32, ddp=False, rank=0, world_size=1, seed=42):
    if ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True, seed=seed
                                     )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=4,
        collate_fn=partial(collate_fn_pretrain, tokenizer=tokenizer, max_seq_len=max_seq_len)
    )
    return loader, sampler


class LemonmindDataset(Dataset):
    def __init__(self, data_path, buffer_size=10, seed=42):
        super().__init__()
        self.data_list = sorted(
            [str(v) for v in Path(data_path).rglob("*.parquet")]
        )
        self.letters = ['A', 'B', 'C', 'D']

        self.buffer_size = buffer_size
        self.seed = seed
        self.data = []

    def _clean_messages(self, messages):
        cleaned = []

        for msg in messages:
            role = msg.get('role', None)
            content = msg.get('content', None)

            if role not in {'system', 'user', 'assistant'}:
                continue

            if content is None:
                continue

            content = str(content).strip()

            if len(content) == 0:
                continue

            cleaned.append({
                'role': role,
                'content': content
            })

        while len(cleaned) > 0 and cleaned[-1]['role'] != 'assistant':
            cleaned.pop()

        if len(cleaned) == 0:
            return None

        # system 只允许出现在第一条
        start_idx = 0

        if cleaned[0]['role'] == 'system':
            start_idx = 1

        if start_idx >= len(cleaned):
            return None

        # system 后必须是 user，不能直接 assistant
        if cleaned[start_idx]['role'] != 'user':
            return None

        # user / assistant 必须交替
        expected_role = 'user'

        for msg in cleaned[start_idx:]:
            if msg['role'] != expected_role:
                return None

            if expected_role == 'user':
                expected_role = 'assistant'
            else:
                expected_role = 'user'

        # 最后一条必须是 assistant
        if cleaned[-1]['role'] != 'assistant':
            return None

        return cleaned

    def load_data(self, buffer_round=0):
        rng = random.Random(self.seed + buffer_round)

        data_list = rng.sample(
            self.data_list,
            k=min(self.buffer_size, len(self.data_list))
        )

        self.data = []
        for path in data_list:
            path_obj = Path(path)
            dataset_name = path_obj.parent.name
            df = pd.read_parquet(path)

            if dataset_name == 'gsm8k':
                for row in df.itertuples(index=False):
                    messages = [
                        {
                            'role': 'user',
                            'content': getattr(row, 'question')
                        },
                        {
                            'role': 'assistant',
                            'content': getattr(row, 'answer')
                        }
                    ]

                    messages = self._clean_messages(messages)

                    if messages is not None:
                        self.data.append({
                            'messages': messages
                        })
            elif dataset_name == 'smoltalk':
                for row in df.itertuples(index=False):
                    messages = getattr(row, 'messages')

                    if hasattr(messages, 'tolist'):
                        messages = messages.tolist()
                    else:
                        messages = list(messages)

                    messages = self._clean_messages(messages)

                    if messages is not None:
                        self.data.append({
                            'messages': messages
                        })
            elif dataset_name == 'mmlu':
                for row in df.itertuples(index=False):
                    question = getattr(row, 'question')
                    choices = getattr(row, 'choices')
                    answer = getattr(row, 'answer')
                    query = f'Multiple Choice question: {question}\n'
                    query += "".join([f"{letter}. {c}\n" for letter, c in zip(self.letters, choices)])
                    query += f'\nRespond only with the letter of the correct answer.'
                    messages = [
                        {
                            'role': 'user',
                            'content': query
                        },
                        {
                            'role': 'assistant',
                            'content': self.letters[answer]
                        }
                    ]
                    messages = self._clean_messages(messages)
                    if messages is not None:
                        self.data.append({
                            'messages': messages
                        })
            elif dataset_name == 'nano_llm_custom_identity':
                for row in df.itertuples(index=False):
                    messages = getattr(row, "text")

                    if hasattr(messages, 'tolist'):
                        messages = messages.tolist()
                    else:
                        messages = list(messages)

                    messages = self._clean_messages(messages)

                    if messages is not None:
                        self.data.append({
                            'messages': messages
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_sft(batch, tokenizer, max_seq_len):
    def norm_text(messages, tokenizer):
        tmp = [tokenizer.bos_token, '\n']
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == 'system':
                tmp.extend([
                    "<|system_start|>", '\n',
                    content, '\n',
                    "<|system_end|>", '\n'
                ])
            elif role == 'user':
                tmp.extend([
                    "<|user_start|>", '\n',
                    content, '\n',
                    "<|user_end|>", '\n'
                ])
            elif role == 'assistant':
                tmp.extend([
                    "<|assistant_start|>", '\n',
                    content, '\n',
                    "<|assistant_end|>", '\n'
                ])
            else:
                raise ValueError(f"Invalid role: {role}")
        tmp.append(tokenizer.eos_token)
        return ''.join(tmp)

    def generate_loss_mask(input_ids, attention_mask, tokenizer):
        vocab = tokenizer.get_vocab()

        if '<|assistant_start|>' not in vocab:
            raise ValueError("<|assistant_start|> not in vocab")
        if '<|assistant_end|>' not in vocab:
            raise ValueError("<|assistant_end|> not in vocab")

        assistant_start_id = tokenizer.convert_tokens_to_ids('<|assistant_start|>')
        assistant_end_id = tokenizer.convert_tokens_to_ids('<|assistant_end|>')

        loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            valid_len = int(attention_mask[i].sum().item())
            ids = input_ids[i, :valid_len]

            starts = torch.where(ids == assistant_start_id)[0]
            ends = torch.where(ids == assistant_end_id)[0]

            end_ptr = 0

            # For each assistant_start, find the nearest following assistant_end.
            # If the end token is missing, use valid_len to support truncated samples.
            for s in starts:
                # Ignore end tokens that are before or at the current start token.
                while end_ptr < len(ends) and ends[end_ptr] <= s:
                    end_ptr += 1

                # Use the next valid end token, or the sequence end if none exists.
                if end_ptr < len(ends):
                    e = ends[end_ptr]
                    end_ptr += 1
                else:
                    e = valid_len

                # Enable loss only for tokens inside the assistant response span:
                # after assistant_start and before assistant_end.
                if s + 1 < e:
                    loss_mask[i, s + 1:e + 1] = True

        loss_mask = loss_mask & attention_mask.bool()
        return loss_mask

    texts = [
        norm_text(b['messages'], tokenizer)
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
    loss_mask = generate_loss_mask(input_ids, attention_mask, tokenizer)[:, 1:]

    return x, y, loss_mask


def lemonmind_dataloader(dataset, tokenizer, max_seq_len, batch_size=32, ddp=False, rank=0, world_size=1, seed=42):
    if ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=seed
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        collate_fn=partial(collate_fn_sft, tokenizer=tokenizer, max_seq_len=max_seq_len)
    )
    return loader, sampler


if __name__ == '__main__':
    from tokenizer import get_tokenizer

    tokenizer = get_tokenizer('../gpt-neox-20b-tokenizer')
    lemonmind_dataset = LemonmindDataset(
        'D:/think-dataset/sft',
        buffer_size=10
    )
    lemonmind_dataset.load_data(buffer_round=0)
    print(len(lemonmind_dataset))

    train_dataloader, train_sampler = lemonmind_dataloader(lemonmind_dataset,
                                                 tokenizer,
                                                 max_seq_len=1024+1,
                                                 batch_size=32
                                                 )
    print(len(train_dataloader))
    for i, (x, y, loss_mask) in enumerate(train_dataloader):
        idx = 0
        print(tokenizer.decode(y[idx]))
        print('========================================')
        print(tokenizer.decode(y[idx][loss_mask[idx]]))
        break
import os
from pathlib import Path

from transformers import AutoTokenizer

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

SPECIAL_TOKENS = {
    'bos_token': '<|bos|>',
    'eos_token': '<|eos|>',
    'pad_token': '<|pad|>',
    'extra_special_tokens': [
        '<|system_start|>',
        '<|system_end|>',
        '<|user_start|>',
        '<|user_end|>',
        '<|assistant_start|>',
        '<|assistant_end|>',
    ]
}


def get_tokenizer(tokenizer_path: str = None):
    if tokenizer_path is not None:
        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


if __name__ == '__main__':
    tokenizer = get_tokenizer()
    text = '<|bos|>\n<|user_start|>请你介绍一下什么是 Transformer。<|user_end|>\n<|assistant_start|>Transformer 是一种基于注意力机制的神经网络架构，常用于自然语言处理和大语言模型。<|assistant_end|>\n<|eos|>'
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))
    print(tokenizer.encode('<|user_start|>'))
    print(tokenizer.decode(tokenizer.encode('<|user_start|>')))
    print(tokenizer.encode('<|user_end|>'))
    print(tokenizer.decode(tokenizer.encode('<|user_end|>')))
    print(tokenizer.encode('<|assistant_start|>'))
    print(tokenizer.decode(tokenizer.encode('<|assistant_start|>')))
    print(tokenizer.encode('<|assistant_end|>'))
    print(tokenizer.decode(tokenizer.encode('<|assistant_end|>')))
    print(tokenizer.encode('<|bos|>'))
    print(tokenizer.decode(tokenizer.encode('<|bos|>')))
    print(tokenizer.encode('<|eos|>'))
    print(tokenizer.decode(tokenizer.encode('<|eos|>')))
    print(tokenizer.encode('<|pad|>'))
    print(tokenizer.decode(tokenizer.encode('<|pad|>')))
import os

from transformers import AutoTokenizer

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    special_tokens = {
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
    tokenizer.add_special_tokens(special_tokens)
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
import os

from transformers import AutoTokenizer

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer.add_special_tokens({
        'bos_token': '<|bos|>',
        'eos_token': '<|eos|>',
        'pad_token': '<|pad|>'
    })
    return tokenizer


if __name__ == '__main__':
    tokenizer = get_tokenizer()
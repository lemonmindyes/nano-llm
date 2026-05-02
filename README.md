# nano-llm

nano-llm is a lightweight and easy-to-implement nano-scale LLM, primarily designed for learning and experimenting with new techniques in LLMs. It uses relatively small and lightweight training datasets, enabling training to be completed on a single GPU within a matter of hours.

| id |model| time | description                                                                                                               | PPL |
|----|----|-----|---------------------------------------------------------------------------------------------------------------------------|--|
| 1  |nano-llm| | model_dim=640, n_layers=12, n_heads=10, n_kv_heads=5, head_dim=64, total_steps=1000000, seq_len=300, total_parameter=118M | 34.54 |
| 2  |nano-llm| | model_dim=768, n_layers=14, n_heads=12, n_kv_heads=6, head_dim=64, total_steps=264000, seq_len=300                        | 29.40 |
| 3  |nano-llm| | model_dim=896, n_layers=16, n_heads=14, n_kv_heads=7, head_dim=64, total_steps=264000, seq_len=300                        | 28.47 |
| 4  |nano-llm| | model_dim=896, n_layers=16, n_heads=14, n_kv_heads=7, head_dim=64, total_steps=528000, seq_len=300                        | 26.35 |
| 5  |nano-llm| | model_dim=1280, n_layers=22, n_heads=20, n_kv_heads=10, head_dim=64, total_steps=2087500, seq_len=320                     | 19.33 |
| 6  |nano-llm| | model_dim=1280, n_layers=22, n_heads=20, n_kv_heads=10, head_dim=64, total_steps=6750000, seq_len=320                     |  |
batch_size=8, accumulation_steps=32

## Usage

### base_train
Perform pretraining
```bash
python base_train.py
```

eval nano-llm
```bash
python base_eval.py
```
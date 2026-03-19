# nano-llm

nano-llm is a lightweight and easy-to-implement nano-scale LLM, primarily designed for learning and experimenting with new techniques in LLMs. It uses relatively small and lightweight training datasets, enabling training to be completed on a single GPU within a matter of hours.

|id|model| time  | description                                                              | PPL  |
|----|----|-------|--------------------------------------------------------------------------|------|
|1|nano-llm| 1.01h | model_dim=512, n_layers=8, n_heads=8, n_kv_heads=4, head_dim=64          | 4.88 |
|2|nano-llm| 0.88h | model_dim=384, n_layers=12, n_heads=6, n_kv_heads=6, head_dim=64, 59.95M | 4.88 |

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
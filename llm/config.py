import os
from pathlib import Path
import torch

REPO_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

# Text extraction
url = 'https://www.gutenberg.org/cache/epub/9182/pg9182-images.html'
header_height = 60  # Main text distance from the top of the page: to remove header
footer_height = 540 # Remove footer
start_page = 45
end_page = 1055
extraction_path = REPO_DIR / "llm/data/book_text.jsonl"

# Text processing
min_length = 100

# HF repo
hf_repo = "iloraishaque/llm-bronte"

# Dataset
context_length = 2048
batch_size = 1000
test_size = 0.1
shuffle = True

# Training
model_name = 'NousResearch/Llama-2-7b-chat-hf'#'bigscience/bloom-3b'


load_in_4bit=True
bnb_4bit_use_double_quant=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16

lora_r = 8#16 # attention heads
lora_alpha = 32 # alpha scaling
lora_dropout = 0.05
lora_bias = "none"
lora_task_type = "CAUSAL_LM" # set this for CLM or Seq2Seq
target_modules=[
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.rotary_emb.inv_freq",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "input_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight"]

## Trainer config
per_device_train_batch_size = 2
gradient_accumulation_steps = 2
warmup_steps = 100
num_train_epochs=3
weight_decay=0.1
learning_rate = 2e-4 
fp16 = True
logging_steps = 1
overwrite_output_dir = True
evaluation_strategy = "no"
save_strategy = "no"
push_to_hub = False

## Data collator
mlm =False

## Generate
max_new_tokens = 50
temperature = 0.5
do_sample = False

import os
import logging
import argparse
import torch
from typing import Mapping, Any

from torch import cuda
from datasets import load_dataset
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig, HfArgumentParser

import config
from training_utils import prepare_model, print_trainable_parameters, compute_perplexity


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMBronte():

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = 'cuda' if cuda.is_available() else 'cpu'

    def train(
            self, 
            hf_repo: str, 
            bnb_config: Mapping[str, Any],
            lora_config: Mapping[str, Any],
            trainer_config: Mapping[str, Any],
            mlm: bool,
        ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name,quantization_config=BitsAndBytesConfig(**bnb_config), device_map={"": 0} )
  
        model = prepare_model(model)
        model = get_peft_model(model, LoraConfig(**lora_config))
        LOGGER.info(f"Model trainable parameters:\n {print_trainable_parameters(model)}")
        dataset = load_dataset("iloraishaque/tokenized-bronte-full")
        LOGGER.info(f"Train dataset downloaded:\n {dataset['train']}")
        LOGGER.info(f"Number of tokens for the training: {dataset['train'].num_rows*len(dataset['train']['input_ids'][0])}")
        trainer = Trainer(
            model=model,
            train_dataset=dataset['train'],
            args=TrainingArguments(**trainer_config),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
        )
        model.config.use_cache = False  # silence warnings
        trainer.train()
        model.config.use_cache = True
####################
        # model.save_pretrained("llm-bronte")
        # trainer.model.push_to_hub(repo_id=hf_repo, token=True)
        # trainer.tokenizer.push_to_hub(repo_id=hf_repo, token=True)
#####################

        # model.save_pretrained('./adapter', token=True)
        model.push_to_hub('iloraishaque/llm-bronte-full', token=True, safe_serialization=True)
        tokenizer.push_to_hub('iloraishaque/llm-bronte-full')

    def evaluate():
        pass

    def generate(self, prompt: str, hf_repo: str, max_new_tokens: int, temperature: float, do_sample: bool) -> None:
        # Import the model
        config = PeftConfig.from_pretrained(hf_repo)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, quantization_config=BitsAndBytesConfig(**bnb_config), device_map={"": 0})
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        # Load the Lora model
        self.model = PeftModel.from_pretrained(model, hf_repo)

        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        print(tokenizer.decode(tokens[0]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an LLM with Lora.")
    parser.add_argument("--model-name", type=str, default=config.model_name, help="The name of the model to train.")
    parser.add_argument("--hf-repo", type=str, default=config.hf_repo, help="The name of the HuggingFace repo to push the model to.")
    parser.add_argument("--mlm", type=bool, default=config.mlm, help="Whether to use MLM or not for the training.")
    parser.add_argument('--lora-r', type=float, default=config.lora_r, help="The Lora parameter r, the number of heads.")
    parser.add_argument('--lora-alpha', type=float, default=config.lora_alpha, help="Lora parameter.")
    parser.add_argument('--lora-dropout', type=float, default=config.lora_dropout, help="Lora dropout.")
    parser.add_argument('--lora-bias', type=str, default=config.lora_bias, help="Lora bias.")
    parser.add_argument('--lora-task-type', type=str, default=config.lora_task_type, help="Lora task type.")
    parser.add_argument('--per-device-train-batch-size', type=int, default=config.per_device_train_batch_size, help="The batch size per device for the training.")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=config.gradient_accumulation_steps, help="The number of gradient accumulation steps.")
    parser.add_argument('--warmup-steps', type=int, default=config.warmup_steps, help="The number of warmup steps.")
    parser.add_argument('--weight-decay', type=float, default=config.weight_decay, help="The weight decay.")
    parser.add_argument('--num-train-epochs', type=float, default=config.num_train_epochs, help="The number of training epochs.")
    parser.add_argument('--learning-rate', type=float, default=config.learning_rate, help="The learning rate.")
    parser.add_argument('--fp16', type=bool, default=config.fp16, help="Whether to use fp16 or not.")
    parser.add_argument('--logging-steps', type=int, default=config.logging_steps, help="The number of logging steps.")
    parser.add_argument('--output-dir', type=str, default=config.hf_repo, help="The output directory.")
    parser.add_argument('--overwrite-output_dir', type=bool, default=config.overwrite_output_dir, help="Whether to overwrite the output directory.")
    parser.add_argument('--save-strategy', type=str, default=config.save_strategy, help="The saving strategy.")
    parser.add_argument('--evaluation-strategy', type=str, default=config.evaluation_strategy, help="The evaluation strategy.")
    parser.add_argument('--push-to-hub', type=bool, default=config.push_to_hub, help="Whether to push the model to the HuggingFace Hub.")
    parser.add_argument("--load_in_4bit", type=bool, default=config.load_in_4bit, help="Whether to load data in 4-bit format.")
    parser.add_argument("--bnb_4bit_use_double_quant", type=bool, default=config.bnb_4bit_use_double_quant, help="Whether to use double quantization in 4-bit BnB.")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default=config.bnb_4bit_quant_type, help="The quantization type for 4-bit BnB (e.g., nf4).")
    parser.add_argument("--bnb_4bit_compute_dtype", type=torch.dtype, default=config.bnb_4bit_compute_dtype, help="The compute dtype for 4-bit BnB (e.g., torch.bfloat16).")
    args = parser.parse_args()

    bnb_config = {
        "load_in_4bit":args.load_in_4bit,
        "bnb_4bit_use_double_quant":args.bnb_4bit_use_double_quant,
        "bnb_4bit_quant_type":args.bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype":args.bnb_4bit_compute_dtype
    }

    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, 
        'bias': args.lora_bias,
        "task_type": args.lora_task_type,
        "target_modules":[
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
    }

    trainer_config = {
        "per_device_train_batch_size": args.per_device_train_batch_size, 
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate, 
        "fp16": args.fp16,
        "logging_steps": args.logging_steps, 
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "evaluation_strategy": args.evaluation_strategy,
        "save_strategy": args.save_strategy,
        "push_to_hub": args.push_to_hub
        # ,"max_steps":1
    }

    model = LLMBronte(args.model_name)
    model.train(
        hf_repo=args.hf_repo,
        bnb_config=bnb_config,
        lora_config=lora_config,
        trainer_config=trainer_config,
        mlm=args.mlm
    )


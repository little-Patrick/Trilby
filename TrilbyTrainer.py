#%% Trilby Trainer
import json
import time
import torch
from transformers.data import data_collator
import matplotlib.pyplot as plt
from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback)

from peft import LoraConfig, get_peft_model, TaskType
from pipeline.dataset import prepare_the_stack_v2_dedup


#%% Language config 
config_path = "./training_configs/local_training/Python/python_the_stack_v2_dedup.json"

with open(config_path, "r") as f:
    config = json.load(f)


#%% 
    # 4-bit quantization
bnb_config = BitsAndBytesConfig(
        load_in4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_bnb_nested_quant=True
    )

    # Load model with quantization config
model_name = config["model"]
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure pad token is set correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Add FIM special tokens BEFORE loading the model
fim_tokens = {
    "prefix": "<fim_prefix>",
    "middle": "<fim_middle>",
    "suffix": "<fim_suffix>"
}
num_added_tokens = tokenizer.add_tokens(list(fim_tokens.values()), special_tokens=True)

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
)

torch.backends.cuda.matmul.allow_tf32 = False
model.config.use_cache = False

# Resize model embeddings to match tokenizer
if num_added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to: {model.get_input_embeddings().num_embeddings}")

train_dataset = ""
eval_dataset = ""
test_dataset = ""

#%% Tokenize Dataset and Split Train/Test
DS = config["dataset"]["database"]

try: 
    if DS == "bigcode/the-stack-v2-dedup":
        start_time = time.time()
        dataset = prepare_the_stack_v2_dedup(config, tokenizer)
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']
        test_dataset = dataset['test']
        print("=" * 50)
        print("Data prep done")
        print("=" * 50)
    else:
        print("!" * 50)
        print("Database not supported")
        print("!" * 50)
except Exception as e:
    print("=" * 50)
    print(f"Error occurred while preparing dataset: {e}")
    print("=" * 50)


#%% Setup LoRA config
CA = config["adapter"]

lora_config = LoraConfig(
        r=CA["r"],
        lora_alpha=CA["lora_alpha"],
        target_modules=CA["target_modules"],
        lora_dropout=CA["lora_dropout"],
        bias=CA["bias"],
        task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)


#%% Model Training Args
TA = config["training_args"]

import random, numpy as np
seed = config.get("seed", 42)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); 
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

training_args = TrainingArguments(
        output_dir=TA["output_dir"],
        per_device_train_batch_size=TA["per_device_train_batch_size"],
        gradient_accumulation_steps=TA["gradient_accumulation_steps"], 
        learning_rate=TA["learning_rate"],
        num_train_epochs=TA["num_train_epochs"],
        fp16=TA["fp16"],
        logging_steps=TA["logging_steps"],
        eval_strategy=TA.get("eval_strategy", "steps"),
        eval_steps=TA["eval_steps"],
        save_strategy=TA["save_strategy"],
        save_steps=TA["save_steps"],
        save_total_limit=TA["save_total_limit"],
        load_best_model_at_end=TA["load_best_model_at_end"],
        metric_for_best_model=TA["metric_for_best_model"],
        greater_is_better=TA["greater_is_better"],
        dataloader_pin_memory=TA["dataloader_pin_memory"],
        remove_unused_columns=TA["remove_unused_columns"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        lr_scheduler_type=TA.get("lr_scheduler_type", "cosine"),
        warmup_steps=TA.get("warmup_steps", 0),
        seed=seed,
)

# If dataset small adjust eval/save dynamically
total_steps_est = (len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) + 1) * training_args.num_train_epochs
if training_args.save_steps > total_steps_est:
    print(f"Adjusting save/eval steps from {training_args.save_steps} to {max(10, total_steps_est//5)}")
    training_args.save_steps = max(10, total_steps_est//5)
    training_args.eval_steps = training_args.save_steps


#%% Trainer
    # Data collactor to handle variable length sequences
data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        )

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=config["early_stopping"]["patience"],
    early_stopping_threshold=config["early_stopping"]["threshold"],
)

    # Add learning rate scheduler callback
from transformers import get_cosine_schedule_with_warmup

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[early_stopping],
    )


#%% Train Model
    # Clear CUDA cache before training
torch.cuda.empty_cache()

    # Train model
try:
    print("Starting training...")
    # print(f"Total training steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    
    train_result = trainer.train()
    
    print("Training completed successfully!")
    
except KeyboardInterrupt:
    print("Training interrupted by user")
    print("Saving current model state...")
    trainer.save_model(config["save"] + "_interrupted")
    
except RuntimeError as e:
    print("=" * 50)
    print(f"CUDA Runtime Error: {e}")
    print("=" * 50)
    torch.cuda.empty_cache()
    raise


# Save final model with metadatr
print("Training completed successfully!")

# Evaluate best model on validation again (optional reassurance)
try:
    val_metrics = trainer.evaluate(eval_dataset)
    print(f"Best/Final validation metrics: {val_metrics}")

    # Evaluate on held-out test set
    test_metrics = trainer.evaluate(test_dataset)
    print(f"Test metrics: {test_metrics}")

    model.save_pretrained(config["save"], savr_embedding_layers=True)
    tokenizer.save_pretrained(config["save"])
except RuntimeError as e:
    print(f"Error saving model/tokenizer: {e}")


#%% Eval

# After validation + test eval add perplexity
import math
if "eval_loss" in val_metrics:
    print(f"Validation perplexity: {math.exp(val_metrics['eval_loss']):.3f}")
if "eval_loss" in test_metrics:
    print(f"Test perplexity: {math.exp(test_metrics['eval_loss']):.3f}")



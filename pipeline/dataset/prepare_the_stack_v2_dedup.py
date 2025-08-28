import os
import boto3
import random
from smart_open import open
from datasets import load_dataset

def prepare_the_stack_v2_dedup(config, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
    s3 = session.client("s3")

    def download_contents(blob_id, src_encoding):
        s3_url = f"s3://softwareheritage/content/{blob_id}"
        
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            content = fin.read().decode(src_encoding)
        
        return {"content": content}

    try:
        print("🔹 Loading dataset...")
        print("=" * 50)
        dataset = load_dataset(
                config["dataset"]["database"], 
                config["dataset"]["data_dir"],
                split=config["dataset"]["split"],
                streaming=config["dataset"]["streaming"]
                )

        print("Mapping")
        dataset = dataset.map(lambda row: download_contents(row["blob_id"], row["src_encoding"]))
        for row in dataset:
            print(row)
            break

        print("Data Loaded")
        print("=" * 50)
    except RuntimeError as e:
        print("Data Could Not load")
        print("=" * 50)
        print(f"{e}")


    # Define FIM tokens for use in transformation
    fim_tokens = {
        "prefix": "<fim_prefix>",
        "middle": "<fim_middle>",
        "suffix": "<fim_suffix>"
    }

    def filter_fn(sample):
        content = sample.get("content", None)
        if not content:
            return False
        if len(content) < config["data_prep"]["too_short"]:  # code snippet too short
            return False
        if len(content) > config["data_prep"]["too_long"]:  # code snippet  too long
            return False
        if sample.get("is_generated", False):  # remove generated code
            return False
        if sample.get("is_vendor", False):  # remove vendor code
            return False
        return True

    dataset = dataset.filter(filter_fn)


    def fim_transform(text, fim_ratio=0.5):
        """
        Convert code into a FIM sample.
        Otherwise, return original text.
        """
        if random.random() > fim_ratio:
            return text  # keep original

        # Pick a random middle span
        n = len(text)
        if n < 50:
            return text  # too small for FIM

        # pick random split points
        start = random.randint(0, n // 2)
        end = random.randint(start + 1, min(n, start + n // 2))

        prefix = text[:start]
        middle = text[start:end]
        suffix = text[end:]

        # Construct FIM prompt format
        return (
            fim_tokens["prefix"]
            + prefix
            + fim_tokens["suffix"]
            + suffix
            + fim_tokens["middle"]
            + middle
        )

    max_length = config["data_prep"]["max_length"]

    # After tokenization
    def tokenize_fn(sample):
        text = fim_transform(sample["content"], fim_ratio=config["data_prep"]["fim_ratio"])
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        input_ids = tokens["input_ids"]
        vocab_size = len(tokenizer)
        
        # Check for invalid tokens
        max_token_id = max(input_ids) if input_ids else 0
        if max_token_id >= vocab_size:
            print(f"ERROR: Token ID {max_token_id} exceeds vocab size {vocab_size}")
            print(f"Text sample: {text[:100]}...")
            
        return {
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"],
            "labels": input_ids
        }

    dataset = dataset.map(tokenize_fn)
    
    # Remove all original columns, keep only tokenized data
    columns_to_remove = [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
    dataset = dataset.remove_columns(columns_to_remove)
    
    print(f"Final dataset columns: {dataset.column_names}")
    
    # Train/test split
    dataset = dataset.train_test_split(
        test_size=0.15,  # 15% for validation
        seed=42
    )
    
    # Further split test into validation and test
    test_val_split = dataset["test"].train_test_split(
        test_size=0.5,  # Split the 15% into 7.5% val, 7.5% test
        seed=42
    )
    
    return {
        "train": dataset["train"],      # 85%
        "validation": test_val_split["train"],  # 7.5%
        "test": test_val_split["test"]          # 7.5%
    }

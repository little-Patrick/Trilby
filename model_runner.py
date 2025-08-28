import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def load_model_and_tokenizer(adapter_path):
    """Load the base model and trained LoRA adapter"""
    print("Loading model and tokenizer...")
    
    # Load tokenizer (includes FIM tokens)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load base model
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Resize base model embeddings to match tokenizer
    print(f"Original model vocab size: {base_model.config.vocab_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    if len(tokenizer) != base_model.config.vocab_size:
        print(f"Resizing model embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Final model vocab size: {model.config.vocab_size}")
    
    return model, tokenizer

def generate_completion(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    """Generate code completion for the given prompt"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new generated part
    completion = generated_text[len(prompt):]
    
    return completion

def create_fim_prompt(prefix, suffix=""):
    """Create a Fill-in-the-Middle prompt"""
    return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"

def main():
    # Path to your trained adapter
    adapter_path = "./training/TinyLlama/Python/python_trilby"
    
    try:
        model, tokenizer = load_model_and_tokenizer(adapter_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\n" + "="*50)
    print("🐍 Trilby Python Code Completion Runner")
    print("="*50)
    print("Commands:")
    print("  'fim' - Fill-in-the-Middle mode")
    print("  'normal' - Normal completion mode") 
    print("  'quit' - Exit")
    print("="*50)
    
    mode = "normal"
    
    while True:
        try:
            command = input(f"\n[{mode.upper()}] Enter command or code: ").strip()
            
            if command.lower() == 'quit':
                print("Goodbye!")
                break
            elif command.lower() == 'fim':
                mode = "fim"
                print("Switched to Fill-in-the-Middle mode")
                continue
            elif command.lower() == 'normal':
                mode = "normal"
                print("Switched to normal completion mode")
                continue
            
            if not command:
                continue
                
            if mode == "fim":
                print("Enter prefix (code before the gap):")
                prefix = input(">>> ")
                print("Enter suffix (code after the gap, or press Enter for none):")
                suffix = input(">>> ")
                
                prompt = create_fim_prompt(prefix, suffix)
                print(f"\nFIM Prompt: {prompt}")
                
            else:  # normal mode
                prompt = command
            
            print("\nGenerating completion...")
            completion = generate_completion(
                model, tokenizer, prompt, 
                max_new_tokens=100, 
                temperature=0.7
            )
            
            print("\n" + "="*30 + " COMPLETION " + "="*30)
            if mode == "fim":
                print(f"Prefix: {prefix}")
                print(f"Generated Middle: {completion}")
                print(f"Suffix: {suffix}")
            else:
                print(f"Input: {prompt}")
                print(f"Completion: {completion}")
            print("="*73)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
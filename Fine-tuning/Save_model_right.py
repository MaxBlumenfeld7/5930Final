from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
base_model_id = "HuggingFaceTB/SmolLM2-135M"  # Base model
lora_model_path = "5930Final/Fine-tuning/smollm2_finetuned/05"  # Fine-tuned LoRA model
# output_dir = "5930Final/Fine-tuning/smollm2_finetuned/07"  # old path which was actually used
output_dir = 'Big_models/07'## new path which reflects current locations of files

# Step 1: Load the base model
print("Loading the base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_id)

# Step 2: Apply the LoRA weights
print("Loading the LoRA weights...")
lora_model = PeftModel.from_pretrained(model, lora_model_path)

# Step 3: Merge LoRA weights into the base model
print("Merging LoRA weights with the base model...")
model = lora_model.merge_and_unload()  # Merge LoRA weights into the base model

# Step 4: Save the complete model
print(f"Saving the merged model to {output_dir}...")
model.save_pretrained(output_dir, safe_serialization=True)  # Save in `safetensors` format

# Save the tokenizer (reuse the base tokenizer)
print("Saving the tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)  # Use the tokenizer from the base model
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer successfully saved to {output_dir}.")

#This code was used to push the model to hugging face spaces for deployment purposes. 

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model and tokenizer
# model_path = "5930Final/Fine-tuning/smollm2_finetuned/07"
model_path = 'Big_models/07'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Push the model and tokenizer to the Hugging Face Hub. numbers represent attempts at deployment of the same model
model.push_to_hub("MaxBlumenfeld/smollm2-135m-bootleg-instruct04")
tokenizer.push_to_hub("MaxBlumenfeld/smollm2-135m-bootleg-instruct04")

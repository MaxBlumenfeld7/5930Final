#This code was used to push the model to hugging face spaces for deployment purposes. 

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model and tokenizer
# model_path = "5930Final/Fine-tuning/smollm2_finetuned/07" old path which was actually used
model_path = 'Big_models/07' # new path which reflects current locations of things. big_models is outside the repo beacause its files are too big
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Push the model and tokenizer to the Hugging Face Hub. numbers represent attempts at deployment of the same model
model.push_to_hub("MaxBlumenfeld/smollm2-135m-bootleg-instruct04")
tokenizer.push_to_hub("MaxBlumenfeld/smollm2-135m-bootleg-instruct04")

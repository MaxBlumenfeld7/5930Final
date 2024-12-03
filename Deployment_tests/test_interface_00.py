import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

def chat(message):
    inputs = tokenizer(message, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # This is the key fix - return both the user's message and the response
    return [[message, response]]

demo = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="chatbot",
    title="DistilGPT2 Chat"
)

if __name__ == "__main__":
    demo.launch()
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import gradio as gr

# # Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# def chat(message):
#     inputs = tokenizer(message, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs.input_ids,
#             max_length=100,
#             pad_token_id=tokenizer.pad_token_id,
#             do_sample=True,
#             temperature=0.7
#         )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # This is the key fix - return both the user's message and the response
#     return [[message, response]]

# demo = gr.Interface(
#     fn=chat,
#     inputs="text",
#     outputs="chatbot",
#     title="DistilGPT2 Chat"
# )

# if __name__ == "__main__":
#     demo.launch()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# # Define model path
# MODEL_PATH = "./fine_tuned_distilgpt2_02"

# # Load model and tokenizer
# # tokenizer = AutoTokenizer.from_pretrained("distilgpt2")  # Original line
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# tokenizer.pad_token = tokenizer.eos_token

# # model = AutoModelForCausalLM.from_pretrained("distilgpt2")  # Original line
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

def chat(message, temperature, max_length, context_prompt):
    # Prepare input with context
    if context_prompt:
        full_prompt = f"{context_prompt}\n\nHuman: {message}\nAssistant:"
    else:
        full_prompt = f"Human: {message}\nAssistant:"
    
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    response = response.split("Assistant:")[-1].strip()
    
    return [[message, response]]

demo = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="Message"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.1, label="Temperature"),
        gr.Slider(minimum=50, maximum=500, value=50, step=10, label="Max Length"),
        gr.Textbox(
            label="Context Prompt",
            placeholder="Optional: Add context to guide the model's responses",
            value="You are a helpful AI assistant. Keep your responses concise and clear.",
            lines=3
        )
    ],
    outputs="chatbot",
    title="DistilGPT2 Chat",
    description="A basic chatbot using DistilGPT2. Note: This is a very simple model with limited capabilities."
)

if __name__ == "__main__":
    demo.launch()
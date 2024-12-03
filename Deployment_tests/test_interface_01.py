import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load tokenizer and model
model_id = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def chat(message, temperature=0.7, max_length=200, system_prompt=""):
    # Prepare input with optional system prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\nHuman: {message}\nAssistant:"
    else:
        full_prompt = f"Human: {message}\nAssistant:"
    
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        # Try to extract just the assistant's response
        response = response.split("Assistant:")[-1].strip()
    except:
        # Fallback to full response if splitting fails
        pass
        
    return [[message, response]]

# Create Gradio interface
demo = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="Message"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature"),
        gr.Slider(minimum=50, maximum=500, value=200, step=10, label="Max Length"),
        gr.Textbox(
            label="System Prompt (Optional)",
            placeholder="Set context or personality for the model",
            lines=3
        )
    ],
    outputs="chatbot",
    title="SmolLM2-135M Chat",
    description="A chatbot using HuggingFaceTB's SmolLM2-360M model. Keep expectations modest - this is a small 360M parameter model."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
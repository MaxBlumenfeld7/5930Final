import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load all four models and tokenizers
model_ids = {
    "135M": "HuggingFaceTB/SmolLM2-135M",
    "135M-instruct": "HuggingFaceTB/SmolLM2-135M-instruct",
    "360M": "HuggingFaceTB/SmolLM2-360M",
    "360M-instruct": "HuggingFaceTB/SmolLM2-360M-instruct"
}

tokenizers = {name: AutoTokenizer.from_pretrained(model_id) 
             for name, model_id in model_ids.items()}
models = {name: AutoModelForCausalLM.from_pretrained(model_id) 
          for name, model_id in model_ids.items()}

def generate_response(model, tokenizer, message, temperature=0.7, max_length=200, system_prompt="", is_instruct=False):
    if is_instruct:
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nHuman: {message}\nAssistant:"
        else:
            full_prompt = f"Human: {message}\nAssistant:"
    else:
        full_prompt = message
    
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
    
    if is_instruct:
        try:
            response = response.split("Assistant:")[-1].strip()
        except:
            pass
    else:
        response = response[len(message):].strip()
        
    return response

def chat(message, temperature=0.7, max_length=200, system_prompt=""):
    responses = {}
    for name, model in models.items():
        is_instruct = "instruct" in name
        responses[name] = generate_response(
            model, 
            tokenizers[name], 
            message, 
            temperature, 
            max_length, 
            system_prompt,
            is_instruct=is_instruct
        )
    
    return {
        output_boxes[name]: resp for name, resp in responses.items()
    }

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# SmolLM2 Model Comparison Demo")
    gr.Markdown("Compare responses between base and instruction-tuned versions of SmolLM2 135M and 360M models")
    
    with gr.Row():
        with gr.Column():
            message_input = gr.Textbox(label="Input Message")
            system_prompt = gr.Textbox(
                label="System Prompt (Optional)",
                placeholder="Set context or personality for the model",
                lines=3
            )
            
        with gr.Column():
            temperature = gr.Slider(
                minimum=0.1, 
                maximum=2.0, 
                value=0.7, 
                label="Temperature"
            )
            max_length = gr.Slider(
                minimum=50, 
                maximum=500, 
                value=200, 
                step=10, 
                label="Max Length"
            )
    
    output_boxes = {}
    
    # 135M Models
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 135M Base Model")
            output_boxes["135M"] = gr.Textbox(label="SmolLM2-135M", lines=5)
            
        with gr.Column():
            gr.Markdown("### 135M Instruct Model")
            output_boxes["135M-instruct"] = gr.Textbox(label="SmolLM2-135M-instruct", lines=5)
    
    # 360M Models        
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 360M Base Model")
            output_boxes["360M"] = gr.Textbox(label="SmolLM2-360M", lines=5)
            
        with gr.Column():
            gr.Markdown("### 360M Instruct Model")
            output_boxes["360M-instruct"] = gr.Textbox(label="SmolLM2-360M-instruct", lines=5)
    
    submit_btn = gr.Button("Generate Responses")
    submit_btn.click(
        fn=chat,
        inputs=[message_input, temperature, max_length, system_prompt],
        outputs=list(output_boxes.values())
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
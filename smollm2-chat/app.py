import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    logger.info("Starting model load...")
    try:
        model_id = "HuggingFaceTB/SmolLM2-135M"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        logger.info("Model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def chat(message, temperature=0.7, max_length=200, system_prompt=""):
    try:
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
    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        return [[message, f"Error: {str(e)}"]]

if __name__ == "__main__":
    try:
        # Load model at startup
        logger.info("Initializing application...")
        tokenizer, model = load_model()
        
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

        # Get port from environment variable
        port = int(os.environ.get("PORT", 8080))  # Default to 8080 for Cloud Run
        logger.info(f"Starting server on port {port}")
        
        # Launch with Cloud Run compatible settings
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False
        )
        logger.info("Server started successfully")
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise

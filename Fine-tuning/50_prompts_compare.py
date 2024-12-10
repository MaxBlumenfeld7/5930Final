import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Load models
base_model_id = "HuggingFaceTB/SmolLM2-135M"
instruct_model_path = "MaxBlumenfeld/smollm2-135m-bootleg-instruct04"


base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_path, local_files_only=True)

base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_path)

# Evaluation prompts
# Evaluation prompts
prompts = {
    "text_generation": [
        "Write a sentence about a dog.",
        "Complete this sentence: The sky is...",
        "Write a sentence about your favorite food.",
        "Make up a name for a new pet.",
        "Write a short story about a tree.",
        "Create a sentence using the word 'happy'.",
        "Write a thank you message.",
        "Complete: My favorite season is...",
        "Make up a silly superhero name.",
        "Write a sentence about the ocean.",
        "Create a title for a children's book.",
        "Write a short weather report.",
        "Make up a name for a restaurant.",
        "Write a sentence about space.",
        "Create a message for a birthday card.",
        "Write a sentence about a cat.",
        "Complete this sentence: Today I feel...",
        "Write about your dream house.",
        "Make up a name for a new color.",
        "Write a sentence about music.",
        "Create a slogan for ice cream.",
        "Write about your perfect day.",
        "Make up a name for a fictional country.",
        "Write a sentence about rain.",
        "Create a motto for a school."
    ],
    "question_answering": [
        "What color is a banana?",
        "How many legs does a cat have?",
        "What do birds use to fly?",
        "Where do fish live?",
        "What do plants need to grow?",
        "What makes a rainbow appear?",
        "What do people use umbrellas for?",
        "Why do leaves fall from trees?",
        "What do we use to tell time?",
        "How do boats float on water?",
        "What makes ice cream melt?",
        "Why do we need sleep?",
        "What makes the sky look blue?",
        "How do airplanes stay in the air?",
        "What happens to water when it freezes?",
        "Why does the moon shine at night?",
        "What makes a car move forward?",
        "What do cows eat?",
        "How do phones work?",
        "What makes a bike stay up?",
        "Why do we need food?",
        "How do magnets work?",
        "What makes rain fall?",
        "Why do we have seasons?",
        "How do books help us learn?"
    ]
}

def generate_response(model, tokenizer, message, temperature=0.5, max_length=200, system_prompt="", is_instruct=False):
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
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if is_instruct:
        try:
            response = response.split("Assistant:")[-1].strip()
        except:
            pass
    else:
        response = response[len(full_prompt):].strip()
    
    return response

def run_batch_evaluation():
    results = []
    
    # Parameters for generation
    temperature = 0.5
    max_length = 200
    system_prompt = ""
    
    # Process each category
    for category, category_prompts in prompts.items():
        for prompt in tqdm(category_prompts, desc=f"Processing {category}"):
            # Generate responses from both models
            base_response = generate_response(
                base_model,
                base_tokenizer,
                prompt,
                temperature,
                max_length,
                system_prompt,
                is_instruct=False
            )
            
            instruct_response = generate_response(
                instruct_model,
                base_tokenizer,
                prompt,
                temperature,
                max_length,
                system_prompt,
                is_instruct=True
            )
            
            # Store results
            results.append({
                "category": category,
                "prompt": prompt,
                "base_response": base_response,
                "instruct_response": instruct_response,
                "timestamp": datetime.now().isoformat()
            })
    
    return results

def save_results(results):
    # Save as JSON for raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"evaluation_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # # Save as CSV for easy analysis
    # df = pd.DataFrame(results)
    # df.to_csv(f"evaluation_results_{timestamp}.csv", index=False)
    
    # Print summary
    print(f"\nEvaluation complete! Processed {len(results)} prompts")
    print(f"Results saved as evaluation_results_{timestamp}.json and .csv")

if __name__ == "__main__":
    print("Starting batch evaluation...")
    results = run_batch_evaluation()
    save_results(results)
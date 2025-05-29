# GENERATIVE-TEXT-MODEL

 GPT-2 Text Generation with Hugging Face
 
This project uses the GPT-2 language model from Hugging Face Transformers to generate coherent paragraphs based on user prompts. You can customize the prompt, output length, temperature, and number of generated texts.

# Features
 
Uses pre-trained GPT-2 for fast and fluent text generation.

Accepts custom user prompts.

Configurable max_length, temperature, and num_return_sequences.

Easily extendable to web apps or fine-tuned models.

# Code Explanation

from transformers import pipeline, set_seed

pipeline: A high-level Hugging Face function that sets up a text generation model (like GPT-2) with a single line.

set_seed: Used to set a random seed so that the output is reproducible.

generator = pipeline('text-generation', model='gpt2')

This line loads the GPT-2 model and its tokenizer for text generation.

pipeline('text-generation') internally loads the default model (gpt2) unless otherwise specified.

It will download the model on first run and cache it locally.

set_seed(42)

Ensures that every time you run the script, you get the same output (useful for debugging or consistent results).

Function to Generate Text

def generate_paragraph(prompt, max_length=150, temperature=1.0, num_return_sequences=1):
    """
    Generate coherent paragraph using GPT-2.
    """
    output = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        pad_token_id=50256  
    )
    return [o['generated_text'] for o in output]
    
prompt: The initial text or topic you want GPT-2 to expand upon.

max_length: Total number of tokens (words + punctuation) in the output.

temperature: Controls randomness:

Low (e.g., 0.7): more predictable and repetitive.

High (e.g., 1.0+): more creative and diverse.

num_return_sequences: How many different outputs you want for the same prompt.

pad_token_id=50256: The end-of-text token used by GPT-2, to avoid warning messages.

The function returns a list of generated text strings.

# Example Prompt and Output

prompt = "The impact of climate change on future generations"

results = generate_paragraph(prompt, max_length=100, temperature=0.9, num_return_sequences=2)

You input a topic sentence and receive two generated paragraphs of up to 100 tokens each, with moderate creativity.


for i, text in enumerate(results):

    print(f"\n--- Generated Paragraph {i+1} ---\n{text}")

Loops through the generated results and prints each paragraph with a label.

# Dependencies
--- 
transformers
torch
---
List these in a requirements.txt file:

transformers

torch

OUTPUT

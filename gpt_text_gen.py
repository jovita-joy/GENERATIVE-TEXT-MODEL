from transformers import pipeline, set_seed


generator = pipeline('text-generation', model='gpt2')


set_seed(42)

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


prompt = "The impact of climate change on future generations"


results = generate_paragraph(prompt, max_length=100, temperature=0.9, num_return_sequences=2)


for i, text in enumerate(results):
    print(f"\n--- Generated Paragraph {i+1} ---\n{text}")

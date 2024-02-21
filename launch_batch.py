import json
import os
import random
from zero123 import Model  # Assumption: zero123 model can be imported like this

# Initialize the model
model = Model()

# Read the prompts.json file
with open('prompts.json', 'r') as f:
    prompts = json.load(f)

output = []

# Iterate over each prompt
for prompt in prompts:
    seeds = []
    for _ in range(4):
        # Generate a random seed
        seed = random.randint(1, 1000)
        seeds.append(seed)

        # Create the image directory
        image_dir = f"images/{prompt['Prompt'].replace(' ', '_')}/{seed}"
        os.makedirs(image_dir, exist_ok=True)

        # Model inference for each seed
        for view in ['front', 'back', 'left', 'right']:  # Assumption: these views exist
            image_path = f"{image_dir}/{view}.png"

            # TODO : generate image from prompt
            # TODO : remove background
            
            # Assumption: model.inference method takes prompt, seed, view as inputs and generates an image
            model.inference(prompt['Prompt'], seed, view, image_path)

    output.append({
        "Prompt": prompt['Prompt'],
        "Seeds": seeds
    })

# Save the output path to a JSON file
with open('output.json', 'w') as f:
    json.dump(output, f)

import os
import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
# from pptree import print_tree # This is not directly used in main.py, but might be used by utils
# from prettytable import PrettyTable # This is not directly used in main.py, but might be used by utils
import openai
import httpx
print(f"OpenAI library version: {openai.__version__}")
print(f"httpx library version: {httpx.__version__}")
import locale
print(f"Default locale: {locale.getdefaultlocale()}")
print(f"Preferred encoding: {locale.getpreferredencoding()}")


# Assuming your utils.py is in the same directory or accessible via PYTHONPATH
from utils import (
    # Agent, Group, parse_hierarchy, parse_group_info, # Not directly used in main
    setup_model,
    load_data, create_question, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query
)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='medqa', help="Dataset name (e.g., 'medqa')")
parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                    help="Model to use (e.g., 'gemini-2.0-flash', 'gpt-4.1-mini')")
parser.add_argument('--difficulty', type=str, default='adaptive',
                    choices=['adaptive', 'basic', 'intermediate', 'advanced'],
                    help="Difficulty processing mode")
parser.add_argument('--num_samples', type=int, default=1, help="Number of samples to process from the test set")


args = parser.parse_args()

# Configure the model based on API keys.
# setup_model now returns True on success, False on failure for essential configurations.
cprint(f"[INFO] Setting up model: {args.model}", "cyan")
configured_successfully = setup_model(args.model) # Assign the boolean return value
if not configured_successfully: # Check the boolean
    cprint(f"Failed to configure model {args.model}. Please check API keys and settings. Exiting.", "red")
    exit(1)
cprint(f"[INFO] Model {args.model} configured successfully.", "green")


# Load data
cprint(f"[INFO] Loading data for dataset: {args.dataset}", "cyan")
test_qa, examplers = load_data(args.dataset)
if not test_qa:
    cprint(f"No test data loaded for {args.dataset}. Exiting.", "red")
    exit(1)
cprint(f"[INFO] Loaded {len(test_qa)} test samples and {len(examplers)} exemplars.", "green")


# agent_emoji list is defined here but process_intermediate_query in utils.py defines its own.
# This list in main.py is currently not being passed or used by the util functions.
# If it were intended to be used by utils, it should be passed as an argument.
# For now, it's harmless but potentially redundant if utils always uses its internal list.
# agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', # ... rest of emojis ]
# random.shuffle(agent_emoji)

results = []
num_to_process = min(args.num_samples, len(test_qa))
cprint(f"[INFO] Starting processing for {num_to_process} samples...", "cyan")

for no, sample in enumerate(tqdm(test_qa[:num_to_process], desc="Processing Samples")):
    # if no == args.num_samples: # Handled by slicing test_qa[:num_to_process]
    #     break
    
    cprint(f"\n[INFO] Processing sample no: {no+1}/{num_to_process}", "blue")
    # total_api_calls = 0 # Initialized but not used. Can be removed or implemented if needed.

    question, img_path = create_question(sample, args.dataset)
    cprint(f"Question: {question[:100]}...", "white") # Print a snippet of the question

    # Determine difficulty using the utility function
    # The 'args.difficulty' string is passed, and 'determine_difficulty' handles 'adaptive' logic.
    # 'determine_difficulty' uses 'gpt-4.1-mini' internally for adaptive assessment as per utils.py update.
    difficulty_level = determine_difficulty(question, args.difficulty)
    cprint(f"Determined difficulty: {difficulty_level}", "yellow")

    final_decision = None
    if difficulty_level == 'basic':
        final_decision = process_basic_query(question, examplers, args.model, args)
    elif difficulty_level == 'intermediate':
        final_decision = process_intermediate_query(question, examplers, args.model, args)
    elif difficulty_level == 'advanced':
        final_decision = process_advanced_query(question, args.model, args)
    else:
        cprint(f"Warning: Unknown difficulty level '{difficulty_level}' determined for sample {no+1}. Skipping.", "red")
        continue

    if final_decision is None:
        cprint(f"Warning: No final decision received for sample {no+1} (Question: {question[:50]}...).", "red")
        # Optionally, append a placeholder or skip
        results.append({
            'question_id': sample.get('id', f'sample_{no+1}'), # Assuming samples might have an ID
            'question': question,
            'label': sample.get('answer_idx', None) if args.dataset == 'medqa' else sample.get('label', None),
            'answer': sample.get('answer', None),
            'options': sample.get('options', None) if args.dataset == 'medqa' else None,
            'response': "Error: No decision processed",
            'difficulty': difficulty_level
        })
        continue


    cprint(f"Final decision for sample {no+1}: {str(final_decision)[:100]}...", "green")

    if args.dataset == 'medqa':
        results.append({
            'question_id': sample.get('id', f'sample_{no+1}'),
            'question': question,
            'label': sample['answer_idx'],
            'answer': sample['answer'],
            'options': sample['options'],
            'response': final_decision, # This will be a dict like {0.0: "response_str"} or {'majority_vote': "response_str"}
            'difficulty': difficulty_level
        })
    else: # Generic result structure for other datasets
        results.append({
            'question_id': sample.get('id', f'sample_{no+1}'),
            'question': question,
            'label': sample.get('label', None), # Assuming other datasets might have a 'label' field
            'response': final_decision,
            'difficulty': difficulty_level
        })


# Save results
output_dir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_filename = f'{args.model.replace("/", "_")}_{args.dataset}_{args.difficulty}_{num_to_process}samples.json'
output_path = os.path.join(output_dir, output_filename)

try:
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)
    cprint(f"\n[INFO] Results saved to: {output_path}", "green")
except Exception as e:
    cprint(f"\n[ERROR] Failed to save results to {output_path}: {e}", "red")

cprint("[INFO] Processing complete.", "magenta")
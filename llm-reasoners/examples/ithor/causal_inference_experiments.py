import re
import pandas as pd
import json
import torch
from tqdm import tqdm
import os, sys
sys.path.append('../../../')
from experiments.datasets import iTHORDataset
import reasoners.benchmark.ithor_utils as ith_utils
from causal_mappers import CausalMapper, CausalMappers, MLP
from world_model import CausalWorldModel, LMWorldModel, ITHState, ITHAction
import utils
from reasoners.lm import ExLlamaV2Model
import numpy as np
from open_clip import get_tokenizer
lm_model = True


# Paths and configurations
crl_model_path = "models/pretrained_models/ithor_biscuit.ckpt"
autoencoder_path = "models/AE_40l_64hid_3c1b3l.ckpt"
causal_mapper_path = "models/causal_encoders_ithor.pt"
tokenizer_path = 'hf-hub:timm/ViT-B-16-SigLIP'
nl_model_path = None
device = "cuda"
config_file = "val_metadata.json"
config = json.load(open(config_file))

# Load models
crl_model, causal_mapper, nl_model, tokenizer = utils.load_models(
    crl_model_path, autoencoder_path, causal_mapper_path, tokenizer_path, nl_model_path, device=device
)

if lm_model:
    exllamav2_model_dir = '../../Meta-Llama-3-8B-Instruct-special-tokens-exl2-6_5'
    exllamav2_lora_dir = None
    exllamav2_mem_map = None
    batch_size = 1
    base_model = ExLlamaV2Model(exllamav2_model_dir, exllamav2_lora_dir, mem_map=exllamav2_mem_map,
                                max_batch_size=batch_size, max_new_tokens=200, max_seq_length=3072)


def normalize_action(action):
    action_words = action.split()
    
    # Determine the action type
    action_type = None
    if "toggled" in action_words:
        action_type = "ToggleObject"
    elif "adjusted" in action_words:
        action_type = "OpenObject"
    elif "picked up" in action_words:
        action_type = "PickupObject"
    elif "placed" in action_words:
        action_type = "PutObject"
    elif "did nothing" in action_words:
        action_type = "NoOp"
    else:
        action_type = "unknown"
    
    # Determine the object
    object_name = None
    if "stove knob" in action:
        for i, word in enumerate(action_words):
            if word == "stove" and i+1 < len(action_words) and action_words[i+1] == "knob":
                object_name = " ".join(action_words[i:i+3])
                break
    elif "cabinet" in action:
        object_name = "cabinet"
    elif "no particular object" in action:
        object_name = "NoObject1"
    else:
        object_name = "unknown"
    
    return f"{action_type} | {object_name}"

def evaluate_n_step(world_model, dataset):
    sucessful, total = 0, 0
    n_step_results = []
    pbar = tqdm(enumerate(dataset), desc="Evaluating samples")
    for idx, sample in pbar:
        images = sample['images']
        states = sample['states']
        plan = sample['plan'].split('\n')
        plan = [action for action in plan if action and action != '[PLAN END]']
        action_str_ = plan[0].replace('carefully', '').replace('skillfully', '').replace('picked up', 'picked')
        normalized_action = ' | '.join(ith_utils.inverse_action(None, action_str_))
        # plan = [normalize_action(action) for action in plan]
        
        world_model.example = sample
        current_state = ITHState(step_idx=0, image=images[0], description=states[0], latents=None)
        # n_step_success = True
        for step_idx, action in enumerate(plan):
            next_state, percentage = world_model.step(current_state, action)
            do_sample=True,
            percentage = percentage['goal_reached'][1]
            predicted_causals = next_state.description
            actual_causals = states[step_idx + 1]
            current_state = next_state
        
        actual_causals = states[-1]
        predicted_causals = next_state.description
        try:
            actual_causals_dict = ith_utils.convert_state(ith_utils.parse_state(actual_causals))
            predicted_causals_dict = ith_utils.convert_state(ith_utils.parse_state(predicted_causals))
            
            n_step_success = ith_utils.compare_states(actual_causals_dict, predicted_causals_dict)[0]
        except:
            n_step_success = False
            
        total, sucessful = total + 1, sucessful + n_step_success
        pbar_percentage = sucessful / total
        pbar.set_postfix({'n_step_success': n_step_success, 'percentage': pbar_percentage})
        
        n_step_results.append({
            'sample_idx': idx,
            'n_step_success': n_step_success,
            'action': normalized_action,
            'percentage': percentage
        })

    return n_step_results

def evaluate_per_action_and_object(world_model, dataset, keys, tokenizer):
    results = []
    episode_length = 20
    for idx in tqdm(range(len(dataset)), desc="Evaluating samples"):
        if idx % episode_length == 0:
            continue
        img, _, *action, latent = dataset[idx]
        next_img, _, *next_action, next_latent = dataset[idx + 1] if idx + 1 < len(dataset) else (None, None, None, None)
        img = torch.tensor(img).to(device)
        if next_img is None or np.any(action[0].numpy() == -100):
            continue
        next_img = torch.tensor(next_img).to(device)
        action_str = tokenizer.decode(action[0], skip_special_tokens=True)
        action_str_ = action_str.replace('carefully', '').replace('skillfully', '').replace('picked up', 'picked')
        normalized_action = ' | '.join(ith_utils.inverse_action(None, action_str_))
        
        description = ith_utils.describe_latent(latent, keys)
        current_state = ITHState(step_idx=idx, image=img, description=description, latents=None)
        next_state, _ = world_model.step(current_state, action_str)
        
        predicted_causals = next_state.description
        actual_causals = ith_utils.describe_latent(next_latent, keys)
        actual_causals_dict = ith_utils.convert_state(ith_utils.parse_state(actual_causals))
        predicted_causals_dict = ith_utils.convert_state(ith_utils.parse_state(predicted_causals))
        
        success = ith_utils.compare_states(actual_causals_dict, predicted_causals_dict)[0]
        results.append({
            'sample_idx': idx,
            'action': normalized_action,
            'success': success
        })

    return results

# Evaluate n-step accuracy for different N values
N_values = [1, 2, 4]
n_step_accuracies = {}
config_file = "val_metadata.json"
# Load existing results if available
if os.path.exists(f"n_step_accuracies_{'lm' if lm_model else 'cwm'}.json"):
    with open(f"n_step_accuracies_{'lm' if lm_model else 'cwm'}.json", "r") as f:
        n_step_accuracies = json.load(f)

for N in N_values:
    if str(N) in n_step_accuracies:
        print(f"Skipping N={N} as it's already calculated")
        continue
    
    data_path = f'data/step_{N}_CIeval.pth'
    prompt = f'prompts/prompt_{N}step.json'
    with open(prompt) as f:
        prompt = json.load(f)
    depth_limit = N + 2
    if lm_model:
        world_model = LMWorldModel(lm_model=base_model,
                            prompt=prompt,
                            device=device,
                            max_steps=depth_limit,
                            )
    else:
        world_model = CausalWorldModel(crl_model, causal_mapper, nl_model, tokenizer, device, depth_limit, config_file)
    dataset = ith_utils.load_ithor_data(data_path, config_file, return_intermediate=True)
    n_step_results = evaluate_n_step(world_model, dataset)
    n_step_results_df = pd.DataFrame(n_step_results)
    n_step_accuracy = n_step_results_df['n_step_success'].mean()
    n_step_accuracies[str(N)] = n_step_accuracy

    # Save n-step accuracy results after each N
    with open(f"n_step_accuracies_{'lm' if lm_model else 'cwm'}.json", "w") as f:
        json.dump(n_step_accuracies, f)
    
    print(f"Completed and saved results for N={N}")
    n_step_results_df.to_csv(f'{N}_step_results_df_{"lm" if lm_model else "cwm"}.csv')
    
import numpy as np
import json
import os
from concurrent.futures import ThreadPoolExecutor
import tqdm
from transformers import AutoTokenizer
from pcfg_logic import GRAMMAR, get_color_name, generate_description_probabilistic

def load_metadata(path):
    with open(path, 'r') as file:
        metadata = json.load(file)
    return metadata

def translate_description(description, metadata, grammar, use_direction=True):
    object_names = {name: tuple(map(int, name.split('_')[1][1:-1].split(', '))) for name in metadata['object_names']}
    color_names = {name: get_color_name(rgb) for name, rgb in object_names.items()}
    NOOP = 'No action was performed.'
    if description == NOOP:
        return description
    if use_direction:
        actions = ['turned', 'changed the state of', 'moved left', 'moved right', 'moved up', 'moved down']
    else:
        actions = ['move', 'turn', 'changed the state of']
    for action in actions:
        if action in description:
            for name in metadata['object_names']:
                object_type_raw, rgb_str = name.split('_')[0], name.split('_')[1]
                object_type = 'traffic light' if object_type_raw == 'trafficlight' else object_type_raw
                rgb_tuple = tuple(map(int, rgb_str[1:-1].split(', ')))
                color_name = get_color_name(rgb_tuple)
                
                if object_type in description and color_name in description:
                    return generate_description_probabilistic(action, object_type, color_name, grammar, use_direction=use_direction)
    raise ValueError(f"Could not find a matching description for the given input: {description}")

def process_single_episode(file, tokenizer, metadata, grammar, use_direction=True):
    data = dict(np.load(file, allow_pickle=True))
    descriptions = data['action_descriptions'] 
    
    translated_descriptions = [translate_description(description, metadata, grammar, use_direction=use_direction) for description in descriptions]
    tokenized_descriptions = [tokenizer(description, return_token_type_ids=True, padding='max_length', max_length=64) for description in translated_descriptions]  # Adjust context_length if necessary
    input_ids = np.stack([desc['input_ids'] for desc in tokenized_descriptions], axis=0)
    attention_mask = np.stack([desc['attention_mask'] for desc in tokenized_descriptions], axis=0)
    token_type_ids = np.stack([desc['token_type_ids'] for desc in tokenized_descriptions], axis=0)
    
    data['input_ids'] = input_ids
    data['attention_mask'] = attention_mask
    data['token_type_ids'] = token_type_ids
    
    np.savez_compressed(file.strip('.npz'), **data)

def process_dataset(path, split, tokenizer, grammar, use_direction=True):
    metadata = load_metadata(f"{'/'.join(path.split('/')[:-1])}/{split}_metadata.json")
    
    files = [os.path.join(root, filename) 
             for root, dirs, filenames in os.walk(path) 
             for filename in filenames if filename.endswith(".npz")]
                
    with ThreadPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(lambda file: process_single_episode(file, tokenizer, metadata, grammar, use_direction=use_direction), files), total=len(files)))

if __name__ == '__main__':
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    paths = [
        '../data/gridworld_simplified_2c2b2l_noturn_noshufflecars/',
        '../data/gridworld_simplified_2c2b2l_noturn_shufflecars/',
    ]
    for path in paths:
        for split in ['train', 'val', 'test', 'test_indep', 'val_indep']:
            process_dataset(path, split, tokenizer, GRAMMAR, use_direction=False)
    
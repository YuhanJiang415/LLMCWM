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
                if description == "move orange obstacle.":
                    object_type_raw, rgb_str = name.split('_')[0], name.split('_')[1]
                    object_type = 'traffic light' if object_type_raw == 'trafficlight' else object_type_raw
                    rgb_tuple = tuple(map(int, rgb_str[1:-1].split(', ')))
                    color_name = get_color_name(rgb_tuple)
                
                object_type_raw, rgb_str = name.split('_')[0], name.split('_')[1]
                object_type = 'traffic light' if object_type_raw == 'trafficlight' else object_type_raw
                rgb_tuple = tuple(map(int, rgb_str[1:-1].split(', ')))
                color_name = get_color_name(rgb_tuple)
                
                
                
                if object_type in description and color_name in description:
                    return generate_description_probabilistic(action, object_type, color_name, grammar, use_direction=use_direction)
    raise ValueError(f"Could not find a matching description for the given input: {description}, object_type_raw: {object_type_raw}, color_name: {rgb_tuple},{color_name}")

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
                
    # with ThreadPoolExecutor() as executor:
    #     list(tqdm.tqdm(executor.map(lambda file: process_single_episode(file, tokenizer, metadata, grammar, use_direction=use_direction), files), total=len(files)))

    print("Running in serial mode to conserve memory...")
    results = []
    for file in tqdm.tqdm(files, total=len(files)):
        result = process_single_episode(file, tokenizer, metadata, grammar, use_direction=use_direction)
        results.append(result)

if __name__ == '__main__':
    # model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # paths = [
    #     'data/gridworld/',
    #     'data/gridworld/',
    # ]
    # for path in paths:
    #     for split in ['train', 'val', 'test', 'test_indep', 'val_indep']:
    #         process_dataset(path, split, tokenizer, GRAMMAR, use_direction=False)
    
    # 导入os和argparse
    import argparse
    import os

    # 1. 设置参数解析器，使其符合 README 的要求
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the raw dataset (e.g., data/gridworld)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed dataset (can be the same as data_dir)')
    args = parser.parse_args()

    
    # # [调试] 手动设置参数
    # class Args:
    #     data_dir = "data/gridworld_dataset_demo"
    #     output_dir = "data/gridworld_dataset_demo"
    # args = Args()

    # 2. 加载 tokenizer
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. 指定我们之前生成的 splits
    splits_to_process = ['train', 'val', 'test']
    
    print(f"Processing data in base directory: {args.data_dir}")
    print(f"Processing splits: {splits_to_process}")

    # 4. 循环处理我们生成的 splits
    #    (我们假设 'use_direction=False'，这与原始脚本一致)
    for split in splits_to_process:
        # 构造指向特定 split 目录的路径
        # (例如: data/gridworld/train)
        split_data_path = os.path.join(args.data_dir, split)
        
        if not os.path.exists(split_data_path):
            print(f"Warning: Path not found, skipping: {split_data_path}")
            continue

        print(f"\nProcessing split: {split}")
        # process_dataset 函数期望的 'path' 参数是
        # 指向 .npz 文件的目录 (例如 .../train/)
        # 它会自动寻找 .../train_metadata.json
        process_dataset(split_data_path, split, tokenizer, GRAMMAR, use_direction=False)

    print("\nProcessing finished.")
    
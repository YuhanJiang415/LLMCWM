import argparse
import os
import json
import torch
import random
import re
from typing import List
import sys
sys.path.append('../../')

def ithor_describe_latent(latent, keys):
    base_object_translations = {
        "NoObject1": "no particular object",
        "NoObject2": "no particular object",
        "NoObject3": "no particular object",
        "Microwave": "the microwave",
        "Toaster": "the toaster",
        "Cabinet_47fc321b": "a wooden cabinet",
        "StoveKnob_38c1dbc2": "the first stove knob for the front-right burner",
        "StoveKnob_690d0d5d": "the second stove knob for the front-left burner",
        "StoveKnob_c8955f66": "the third stove knob for the back-left burner",
        "StoveKnob_cf670576": "the fourth stove knob for the back-right burner",
        "Plate": "the ceramic plate",
        "Egg": "the spherical, brown, fragile Egg",
        "Pan": "the flat, metal, sturdy Pan",
        "CounterTop_f8092513": "the granite countertop"
    }
    descriptions = []
    for value, key in zip(latent, keys):
        base_entity = "_".join(key.split('_')[:-1])
        entity = base_object_translations.get(base_entity, base_entity.split('_')[0])
        attribute = key.split('_')[-1]
        if "center_x" in key or "center_y" in key or "center_z" in key:
            descriptions.append(f"{entity} {attribute.replace('_', ' ')} is at {value:.2f}")
        elif "open" in key or "on" in key or "cooked" in key or "broken" in key or "pickedup" in key:
            state = (
                "open" if "open" in key and value > 0.5 else
                "closed" if "open" in key else
                "on" if "on" in key and value > 0.5 else
                "off" if "on" in key else
                "cooked" if "cooked" in key and value > 0.5 else
                "uncooked" if "cooked" in key else
                "broken" if "broken" in key and value > 0.5 else
                "unbroken" if "broken" in key else
                "picked up" if "pickedup" in key and value > 0.5 else
                "not picked up" if "pickedup" in key else "unknown state"
            )
            descriptions.append(f"{entity} is {state}")
        else:
            descriptions.append(f"{entity} has an unknown attribute with value {value:.2f}")
    return " | ".join(descriptions)

def gridworld_closest_color(requested_color):
    import webcolors
    min_colours = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def gridworld_describe_latent(latent, keys):
    descriptions = []
    for value, key in zip(latent, keys):
        match = re.search(r'([a-zA-Z]+)_\((\d+,\s*\d+,\s*\d+)\)(.*)', key)
        if match:
            entity = match.group(1)
            color_str = match.group(2)
            attribute = match.group(3).replace('_', ' ').strip()
            color_tuple = tuple(map(int, color_str.split(',')))
            color_name = gridworld_closest_color(color_tuple)
        else:
            entity = "unknown entity"
            color_name = "unknown color"
            attribute = "unknown attribute"
        descriptions.append(f"{color_name} {entity} {attribute} is {value:.2f}")
    return ", ".join(descriptions)

def generate_trajectories(dataset, N, M, tokenizer, include_objects_states=False, dataset_type=None):
    from uuid import uuid4
    trajectories = []
    for _ in range(M):
        uid = uuid4()
        if hasattr(dataset, 'data_files_len'):
            total_episodes = dataset.data_files_len
            episode_length = dataset.episode_length
        else:
            total_episodes = len(dataset.data_files)
            episode_length = dataset.episode_length
        if total_episodes == 0:
            raise ValueError("No episodes available in the dataset.")
        episode_index = random.randint(0, total_episodes - 1)
        episode_start_index = episode_index * episode_length
        max_start_index = episode_start_index + episode_length - N
        if episode_start_index + N > episode_start_index + episode_length:
            continue
        start_index = random.randint(episode_start_index, max_start_index)
        imgs, actions, latents, objects_states = [], [], [], []
        for offset in range(N):
            sample = dataset[start_index + offset]
            if include_objects_states and len(sample) == 6:
                img, _, *action, latent, objects_state = sample
                objects_states.append(objects_state)
            else:
                img, _, *action, latent = sample
            imgs.append(img)
            actions.append(action)
            latents.append(latent)
        imgs = torch.stack(imgs)
        try:
            if dataset_type == 'gridworld':
                # actions[:-1], int64
                actions_dict = {
                    'input_ids': torch.stack([torch.as_tensor(a[0], dtype=torch.int64) for a in actions[:-1]]),
                    'token_type_ids': torch.stack([torch.as_tensor(a[1], dtype=torch.int64) for a in actions[:-1]]),
                    'attention_mask': torch.stack([torch.as_tensor(a[2], dtype=torch.int64) for a in actions[:-1]])
                }
            else:
                # ithor: actions[1:]
                actions_dict = {
                    'input_ids': torch.stack([torch.as_tensor(a[0]) for a in actions[1:]]),
                    'token_type_ids': torch.stack([torch.as_tensor(a[1]) for a in actions[1:]]),
                    'attention_mask': torch.stack([torch.as_tensor(a[2]) for a in actions[1:]])
                }
            actions_decoded = tokenizer.batch_decode(actions_dict['input_ids'], skip_special_tokens=True)
        except Exception as e:
            print(f"Error decoding actions. Sample actions: {[a[0] for a in actions]}\nError: {e}")
            raise
        latents = torch.stack([torch.as_tensor(l) for l in latents])
        if include_objects_states:
            trajectories.append((imgs, actions_decoded, latents, uid, objects_states))
        else:
            trajectories.append((imgs, actions_decoded, latents, uid))
    return trajectories

def ithor_full_coverage_example_pool(trajectories, keys, describe_latent_fn):
    action_types = {
        'toggle_stoveknob_1': ["toggled the first stove knob"],
        'toggle_stoveknob_2': ["toggled the second stove knob"],
        'toggle_stoveknob_3': ["toggled the third stove knob"],
        'toggle_stoveknob_4': ["toggled the fourth stove knob"],
        'toggle_toaster': ["toggled the toaster"],
        'toggle_microwave': ["toggled the microwave"],
        'pickup_plate': ["picked up the ceramic plate"],
        'pickup_egg': ["picked up the spherical brown fragile egg"],
        'put_plate_countertop': ["placed the ceramic plate on the granite countertop"],
        'put_plate_microwave': ["placed the ceramic plate on the microwave"],
        'put_egg_pan': ["placed the spherical brown fragile egg on the flat metal sturdy pan"],
        'move_plate': ["moved the ceramic plate"],
        'move_egg': ["moved the spherical brown fragile egg"],
        'open_microwave': ["adjusted the microwave's door"],
        'open_cabinet': ["adjusted the wooden cabinet's door"],
        'noop': ["did nothing with no particular object"],
    }
    used = set()
    example_pool = []
    for action_key, substrings in action_types.items():
        found = False
        for trajectory in trajectories:
            imgs, actions, latents = trajectory[:3]
            for idx, action in enumerate(actions):
                if all(s.lower() in action.lower() for s in substrings):
                    # Only add one example per action type
                    if (action_key, idx) not in used:
                        init_state = describe_latent_fn(latents[idx], keys)
                        goal_state = describe_latent_fn(latents[idx+1], keys) if idx+1 < len(latents) else describe_latent_fn(latents[-1], keys)
                        plan = "\n".join([f"{a}" for a in actions]) + "\n[PLAN END]\n"
                        scenario = {"init": init_state, "goal": goal_state, "plan": plan, "states": []}
                        example_pool.append(scenario)
                        used.add((action_key, idx))
                        found = True
                        break
            if found:
                break
    return example_pool

def gridworld_full_coverage_example_pool(trajectories, keys, describe_latent_fn):
    action_types = {
        'move_car': ["move", "vehicle"],
        'change_trafficlight': ["change", "traffic light"],
        'move_obstacle': ["move", "obstacle"],
        'no_action': ["no", "action"],
    }
    used = set()
    example_pool = []
    for action_key, substrings in action_types.items():
        found = False
        for trajectory in trajectories:
            imgs, actions, latents = trajectory[:3]
            for idx, action in enumerate(actions):
                if all(s.lower() in action.lower() for s in substrings):
                    if (action_key, idx) not in used:
                        init_state = describe_latent_fn(latents[idx], keys)
                        goal_state = describe_latent_fn(latents[idx+1], keys) if idx+1 < len(latents) else describe_latent_fn(latents[-1], keys)
                        plan = "\n".join([f"{a}" for a in actions]) + "\n[PLAN END]\n"
                        scenario = {"init": init_state, "goal": goal_state, "plan": plan, "states": []}
                        example_pool.append(scenario)
                        used.add((action_key, idx))
                        found = True
                        break
            if found:
                break
    return example_pool

def generate_prompt_json(trajectories, keys, describe_latent_fn, intro, dataset_type=None, full_coverage=False, max_examples=None):
    data = {"intro": intro, "example_pool": []}
    if dataset_type == 'ithor' and full_coverage:
        data["example_pool"] = ithor_full_coverage_example_pool(trajectories, keys, describe_latent_fn)
    else:
        if dataset_type == 'gridworld' and max_examples is not None:
            if len(trajectories) > max_examples:
                trajectories = random.sample(trajectories, max_examples)
        for trajectory in trajectories:
            imgs, actions, latents = trajectory[:3]
            init_state = describe_latent_fn(latents[0], keys)
            goal_state = describe_latent_fn(latents[-1], keys)
            plan = "\n".join([f"{action}" for action in actions]) + "\n[PLAN END]\n"
            scenario = {"init": init_state, "goal": goal_state, "plan": plan}
            if dataset_type == 'gridworld':
                scenario["states"] = []
            data["example_pool"].append(scenario)
    return data

def generate_self_eval_prompts(trajectories, keys, describe_latent_fn, intro):
    self_evals = []
    for trajectory in trajectories:
        actions = trajectory[1]
        latents = trajectory[2]
        for i in range(len(actions)):
            if i < len(latents) - 1:
                positive_action = actions[i]
                positive_init_state = latents[i]
                positive_goal_state = latents[i+1]
                positive_init_desc = describe_latent_fn(positive_init_state, keys)
                positive_goal_desc = describe_latent_fn(positive_goal_state, keys)
                positive_sample = f"[STATEMENT]\nAs initial conditions I have that, {positive_init_desc}.\nMy goal is to have that {positive_goal_desc}.\n[ACTION]\n{positive_action}\n[EVALUATION]\ngood\n\n"
                negative_action = random.choice(actions)
                while negative_action == positive_action:
                    negative_action = random.choice(actions)
                negative_sample = f"[STATEMENT]\nAs initial conditions I have that, {positive_init_desc}.\nMy goal is to have that {positive_goal_desc}.\n[ACTION]\n{negative_action}\n[EVALUATION]\nbad\n\n"
                self_evals.append(positive_sample)
                self_evals.append(negative_sample)
    return intro + "\n\n" + "".join(self_evals) + '[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>\n[ACTION]\n<action>\n[EVALUATION]\n'

def filter_trajectories_for_action(trajectories, action_type):
    filtered = []
    for trajectory in trajectories:
        imgs, actions, latents = trajectory[:3]
        for action in actions:
            if all(word in action for word in action_type):
                filtered.append(trajectory)
                break
    return filtered

def generate_icl_prompt_for_action(filtered_trajectories, action_type, max_scenarios, keys, describe_latent_fn):
    icl_prompts = []
    for idx, trajectory in enumerate(filtered_trajectories[:max_scenarios]):
        imgs, actions, latents = trajectory[:3]
        for action_idx, action in enumerate(actions):
            if all(word in action for word in action_type):
                init_state = describe_latent_fn(latents[action_idx], keys)
                new_state = describe_latent_fn(latents[action_idx + 1], keys)
                prompt = f"[SCENARIO {idx}][STATE 0] {init_state}\n[ACTION] {action}\n[STATE 1] {new_state}\n"
                icl_prompts.append(prompt)
                break
    final_scenario = f"[SCENARIO {max_scenarios + 1}][STATE 0] <state>\n[ACTION] <action>\n[STATE 1]"
    icl_prompts.append(final_scenario)
    return "".join(icl_prompts)

def generate_icl_prompts(trajectories, keys, describe_latent_fn, max_scenarios=2):
    actions_to_track = {
        "world_update_move_car": ["move", "vehicle"],
        "world_update_change_trafficlight": ["change", "traffic light"],
        "world_update_move_obstacle": ["move", "obstacle"],
        "world_update_no_action": ["no", "action"]
    }
    explanation = "I am engaged in experiments within a dynamic gridworld environment. Here are the actions I can perform:\n\n- Move a vehicle according to traffic light status and obstacle presence.\n- Change the state of a traffic light.\n- Relocate an obstacle within the grid.\n\nI have the following restrictions on my actions:\n- A vehicle can only move into an adjacent cell if the traffic light it faces is green and there are no obstacles in the cell.\n- Traffic lights can only switch between green and red states.\n- Obstacles can be moved but not removed from the grid.\n\nHere are some scenarios:\n"
    icl_prompt_structure = {}
    for action_key, action_type in actions_to_track.items():
        filtered_trajectories = filter_trajectories_for_action(trajectories, action_type)
        icl_prompt_structure[action_key] = explanation + generate_icl_prompt_for_action(filtered_trajectories, action_type, max_scenarios, keys, describe_latent_fn)
    return icl_prompt_structure

def generate_ithor_icl_prompts(trajectories, keys, describe_latent_fn, max_scenarios=2):
    # Map action types to world_update_* keys and identifying substrings
    action_types = {
        'world_update_toggle_stoveknob_1': ["toggled the first stove knob"],
        'world_update_toggle_stoveknob_2': ["toggled the second stove knob"],
        'world_update_toggle_stoveknob_3': ["toggled the third stove knob"],
        'world_update_toggle_stoveknob_4': ["toggled the fourth stove knob"],
        'world_update_toggle_toaster': ["toggled the toaster"],
        'world_update_toggle_microwave': ["toggled the microwave"],
        'world_update_pickup_plate': ["picked up the ceramic plate"],
        'world_update_pickup_egg': ["picked up the spherical brown fragile egg"],
        'world_update_put_plate_countertop': ["placed the ceramic plate on the granite countertop"],
        'world_update_put_plate_microwave': ["placed the ceramic plate on the microwave"],
        'world_update_put_egg_pan': ["placed the spherical brown fragile egg on the flat metal sturdy pan"],
        'world_update_move_plate': ["moved the ceramic plate"],
        'world_update_move_egg': ["moved the spherical brown fragile egg"],
        'world_update_open_microwave': ["adjusted the microwave's door"],
        'world_update_open_cabinet': ["adjusted the wooden cabinet's door"],
        'world_update_no_action': ["did nothing with no particular object"],
    }
    explanation = (
        "I am conducting experiments within a dynamic kitchen environment, specifically utilizing the iTHOR dataset based on the FloorPlan10 environment. "
        "This setup features a kitchen floor plan with various interactive objects and a robot positioned in front of the kitchen counter. The robot's position remains fixed throughout the experiments.\n\n"
        "In this environment, I manipulate two movable objects (a plate with a potato and an egg) placed randomly on the counter, and a pan positioned on the stove. At each timestep, a chosen action is performed on one of these objects.\n\n"
        "Here are some scenarios:\n"
    )
    icl_prompt_structure = {}
    for action_key, substrings in action_types.items():
        filtered_trajectories = []
        for trajectory in trajectories:
            imgs, actions, latents = trajectory[:3]
            for idx, action in enumerate(actions):
                if all(s.lower() in action.lower() for s in substrings):
                    filtered_trajectories.append((imgs, actions, latents, idx))
                    break
        icl_prompts = []
        for i, (imgs, actions, latents, idx) in enumerate(filtered_trajectories[:max_scenarios]):
            init_state = describe_latent_fn(latents[idx], keys)
            new_state = describe_latent_fn(latents[idx+1], keys) if idx+1 < len(latents) else describe_latent_fn(latents[-1], keys)
            prompt = f"[SCENARIO {i}][STATE 0] {init_state}\n[ACTION] {actions[idx]}\n[STATE 1] {new_state}\n"
            icl_prompts.append(prompt)
        final_scenario = f"[SCENARIO {len(icl_prompts)}][STATE 0] <state>\n[ACTION] <action>\n[STATE 1]"
        icl_prompts.append(final_scenario)
        icl_prompt_structure[action_key] = explanation + "".join(icl_prompts)
    return icl_prompt_structure

def main():
    parser = argparse.ArgumentParser(description="Generate trajectories and prompts for iTHOR or gridworld.")
    parser.add_argument('--dataset', choices=['ithor', 'gridworld'], required=True)
    parser.add_argument('--N', type=int, required=True, help='Trajectory length')
    parser.add_argument('--M', type=int, required=True, help='Number of trajectories')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--prompt_name', type=str, default=None, help='Prompt file name (optional)')
    parser.add_argument('--max_examples', type=int, default=10, help='Maximum number of examples in prompt (gridworld only)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == 'ithor':
        from experiments.datasets import iTHORDataset
        from open_clip import get_tokenizer
        tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
        dataset = iTHORDataset(data_folder=args.data_folder, split='mcts_eval', return_targets=False, return_latents=True, single_image=True, triplet=False, seq_len=2, cluster=False, return_text=True, return_simulation_object_states=True, subsample_percentage=0.05)
        config = json.load(open(args.config_path))
        keys = list(dataset.get_causal_var_info().keys())
        intro = "I am conducting experiments within a dynamic kitchen environment, specifically utilizing the iTHOR dataset based on the FloorPlan10 environment. This setup features a kitchen floor plan with various interactive objects and a robot positioned in front of the kitchen counter. The robot's position remains fixed throughout the experiments.\n\nIn this environment, I manipulate two movable objects (a plate with a potato and an egg) placed randomly on the counter, and a pan positioned on the stove. At each timestep, a chosen action is performed on one of these objects.\n\nThe actions and objects involved are as follows:\n1. Plate: Can be picked up, moved, or put down.\n2. Egg: Can be picked up, moved, or put into the pan (where it breaks and cannot be picked up or moved again).\n3. Pan: Remains on the stove and interacts with the egg.\n4. Microwave: Can be toggled (if closed) and opened (if turned off).\n5. Stoves: Controlled by knobs to turn on or off.\n6. Cabinet: Can be opened or closed.\n7. Toaster: Can be turned on or off.\n\nThe movement and interaction rules are as follows:\n- Objects can only be opened if they are closed, and vice versa.\n- The microwave can only be opened when it is turned off.\n- Only one of the two movable objects can be picked up at a time.\n- When an object is picked up, other objects can be moved on the counter to new random positions.\n- When an object is put down, it is placed in a randomly chosen available receptacle (e.g., counter, microwave if open for the plate; counter, pan for the egg).\n\nInterventions in the environment involve changing the state of objects or their positions, influencing the subsequent possible actions and states. The action 'You performed no action' represents no interaction and serves as an observational regime.\n\nThe causal variables in this environment include:\n- Binary Variables: Indicating the states of objects (e.g., Cabinet-Open, Egg-Broken, Microwave-Open).\n- Position Variables: Representing the x-y-z coordinates of movable objects (e.g., Egg-Pos-x, Plate-Pos-y).\n- Action-State Variables: Reflecting actions such as whether an object is picked up or active (e.g., Egg-PickedUp, Toaster-Active).\n\nThese variables form a network of interactions that determine the outcomes of various actions within the kitchen environment. \n\nThis model supports queries through a causal world model, which outputs natural language descriptions based on input actions, thereby facilitating an understanding of interaction outcomes. The causal variables in this environment include the positions and states of the objects, and these variables form a network of interactions that determine the results of various actions within the kitchen."
        describe_latent_fn = ithor_describe_latent
        include_objects_states = True
        step_file = os.path.join(args.output_dir, f'step_{args.N}_ws.pth')
        prompt_file = os.path.join(args.output_dir, args.prompt_name or f'prompt_{args.N}step.json')
        trajectories = generate_trajectories(dataset, args.N, args.M, tokenizer, include_objects_states=include_objects_states, dataset_type='ithor')
        prompt_json = generate_prompt_json(trajectories, keys, describe_latent_fn, intro, dataset_type='ithor', full_coverage=True)
        selfeval_text = generate_self_eval_prompts(trajectories, keys, describe_latent_fn, intro)
        prompt_json['self-eval'] = selfeval_text
        icl_prompts = generate_ithor_icl_prompts(trajectories, keys, describe_latent_fn, max_scenarios=2)
        prompt_json.update(icl_prompts)
    else:
        from experiments.datasets import GridworldDataset
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        config = json.load(open(args.config_path))
        dataset = GridworldDataset(data_folder=args.data_folder, split='val', return_targets=False, return_latents=True, single_image=True, triplet=False, seq_len=2, cluster=False, return_text=True, subsample_percentage=0.5)
        keys = config['flattened_causals']
        intro = "I am engaged in experiments within a dynamic gridworld environment, characterized by its causal structure governing entity interactions. The environment is laid out as a grid with dimensions $H \\times H$, where $H \\in \\mathbb{N}$ encompasses both the grid's height and width. The top-left corner of this grid is positioned at coordinate $(0, 0)$. This environment comprises three primary entities: cars, traffic lights, and obstacles. Each entity is visually distinguished by a unique color, which indicates its fixed attributes. Cars are positioned within the grid with normalized coordinates from 0 to 1, with $(0,0)$ marking the top-left corner. Each car possesses an orientation\u2014up, down, left, right\u2014and its movement is dictated by the state of the traffic light it faces, provided there are no obstacles in its intended path. Traffic lights, positioned at fixed points within the grid, exhibit two states: '0' for green and '1' for red, controlling car movements accordingly. Obstacles have initial placements at the start of each episode and can be relocated but not removed during the episode, influencing potential paths for cars. The movement rules for cars specify that a vehicle facing a green traffic light will proceed into an adjacent cell if that cell is free of obstacles, other cars, and traffic lights, and falls within the grid boundaries. The movement direction is dependent on the car's orientation. Interventions in this environment can change the state of traffic lights or relocate obstacles. Such changes directly influence the available actions and subsequent movements of the cars. This model supports queries through a causal world model, which outputs natural language descriptions based on input actions, thereby facilitating an understanding of interaction outcomes. The causal variables in this environment include the positions of cars and obstacles, and the states of traffic lights. These variables form a network of interactions that determine the results of various actions within the gridworld."
        describe_latent_fn = gridworld_describe_latent
        include_objects_states = False
        step_file = os.path.join(args.output_dir, f'step_{args.N}_ws.pth')
        prompt_file = os.path.join(args.output_dir, args.prompt_name or f'prompt_{args.N}step.json')
        trajectories = generate_trajectories(dataset, args.N, args.M, tokenizer, include_objects_states=include_objects_states, dataset_type='gridworld')
        prompt_json = generate_prompt_json(trajectories, keys, describe_latent_fn, intro, dataset_type='gridworld', max_examples=args.max_examples)
        selfeval_text = generate_self_eval_prompts(trajectories, keys, describe_latent_fn, intro)
        prompt_json['self-eval'] = selfeval_text
        icl_prompts = generate_icl_prompts(trajectories, keys, describe_latent_fn, max_scenarios=2)
        prompt_json.update(icl_prompts)

    print(f"Generating {args.M} trajectories of length {args.N} for {args.dataset}...")
    torch.save(trajectories, step_file)
    print(f"Saved trajectories to {step_file}")

    print(f"Generating prompt JSON...")
    with open(prompt_file, 'w') as f:
        json.dump(prompt_json, f, indent=4)
    print(f"Saved prompt JSON to {prompt_file}")

if __name__ == '__main__':
    main() 
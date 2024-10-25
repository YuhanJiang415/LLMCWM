import numpy as np
from ai2thor.controller import Controller
from copy import deepcopy
import json
from collections import defaultdict
import sys
sys.path.append('../../../data_generation/')
from data_generation_ithor import (get_object_id, initialize_environment, 
                                   perform_action, get_environment_state, simplify_latents)
import time
import gc
from subprocess import Popen, PIPE
import data_generation_ithor
from contextlib import contextmanager
import copy

import data_generation_ithor
from contextlib import contextmanager
import copy
from functools import partial
from reasoners.benchmark.ithor_utils import load_ithor_data
from tqdm import tqdm


perform_action = partial(perform_action, break_egg_if_on_pan=True)

@contextmanager
def reset_globals():
    original_state = {
        "RESOLUTION": copy.deepcopy(data_generation_ithor.RESOLUTION),
        "SIMPLE_SET": copy.deepcopy(data_generation_ithor.SIMPLE_SET),
        "OBJECT_NAMES": copy.deepcopy(data_generation_ithor.OBJECT_NAMES),
        "MIN_DIST": copy.deepcopy(data_generation_ithor.MIN_DIST),
        "NOT_MOVABLE": copy.deepcopy(data_generation_ithor.NOT_MOVABLE),
        "FIXED_POSITION_DICT": copy.deepcopy(data_generation_ithor.FIXED_POSITION_DICT),
        "MOVABLE_POSITION_DICT": copy.deepcopy(data_generation_ithor.MOVABLE_POSITION_DICT),
        "COUNTER_POSITIONS": copy.deepcopy(data_generation_ithor.COUNTER_POSITIONS),
        "CATEGORICAL_POSITION_DICT": copy.deepcopy(data_generation_ithor.CATEGORICAL_POSITION_DICT),
        "ACTIONS": copy.deepcopy(data_generation_ithor.ACTIONS),
        "INTERACT_OBJS": copy.deepcopy(data_generation_ithor.INTERACT_OBJS),
        "PICKUP": copy.deepcopy(data_generation_ithor.PICKUP)
    }

    yield

    for var, value in original_state.items():
        setattr(data_generation_ithor, var, value)



COUNTER_POSITIONS = [
    {"x": 0.75, "y": 0.98, "z": -0.35},
    {"x": 1.03, "y": 0.98, "z": -0.35},
    {"x": 0.65, "y": 0.98, "z": -0.55}
]

PAN_POSITION = {"x": 0.85, "y": 0.95, "z": -1.20}

def convert_goal_state(goal_state):
    attribute_mapping = {
        'picked_up': 'pickedup',
        'on': 'on',
        'open': 'open',
        'broken': 'broken',
        'cooked': 'cooked'
    }

    object_mapping = {
        'a wooden cabinet': 'Cabinet_47fc321b',
        'Egg': 'Egg_afaaaca3',
        'Microwave': 'Microwave_d8b935e4',
        'Plate': 'Plate_49b95a7a',
        'the first stove knob for the frontright burner': 'StoveKnob_38c1dbc2',
        'the first stove knob for the front-right burner': 'StoveKnob_38c1dbc2',
        'the second stove knob for the frontleft burner': 'StoveKnob_690d0d5d',
        'the second stove knob for the front-left burner': 'StoveKnob_690d0d5d',
        'the third stove knob for the backleft burner': 'StoveKnob_c8955f66',
        'the third stove knob for the back-left burner': 'StoveKnob_c8955f66',
        'the fourth stove knob for the backright burner': 'StoveKnob_cf670576',
        'the fourth stove knob for the back-right burner': 'StoveKnob_cf670576',
        'Toaster': 'Toaster_194647f5'
    }

    converted_state = {}

    for obj, attributes in goal_state.items():
        mapped_obj = object_mapping.get(obj, obj)
        for attr, value in attributes.items():
            if attr == 'position':
                converted_state[f"{mapped_obj}_center"] = value
            else:
                mapped_attr = attribute_mapping.get(attr, attr)
                if mapped_obj not in converted_state:
                    converted_state[mapped_obj] = {}
                converted_state[mapped_obj][mapped_attr] = 1.0 if value else 0.0

    return converted_state


def compare_positions(pos1, pos2):
    for key in pos1:
        if key not in pos2 or abs(pos1[key] - pos2[key]) > 0.01:
            return False
    return True

def get_general_location(attributes):
    if attributes.get('pickedup', False) or attributes.get('picked_up', False):
        return 'picked up'
    elif 'position' in attributes or 'center_x' in attributes or 'x' in attributes:
        if 'position' in attributes and isinstance(attributes['position'], dict):
            x = attributes['position'].get('x', 0)
            y = attributes['position'].get('y', 0)
            z = attributes['position'].get('z', 0)
        else:
            x = attributes.get('x', 0)
            y = attributes.get('y', 0)
            z = attributes.get('z', 0)
            
    # MERGE CODE FROM CLUSTER FOR THIS LOGIC
    
    return 'other'

def compare_states(goal_state, final_state, coarse_eval=True):
    total_attributes = 0
    matching_attributes = 0
    location_evaluated = {k: False for k in goal_state.keys() if k.endswith('_center')}
    ignore_positions_for = set()

    for obj, attributes in goal_state.items():
        if obj not in final_state:
            continue
        
        for attr in ['broken', 'cooked', 'pickedup']:
            if attr in attributes and final_state[obj].get(attr) == attributes[attr]:
                matching_attributes += 1
                total_attributes += 1
                ignore_positions_for.add(obj.replace('_center', ''))
                break  # Since we only need one attribute match to ignore positions

    for obj, attributes in goal_state.items():
        if obj not in final_state:
            continue
        
        for attr, value in attributes.items():
            if obj.replace('_center', '') in ignore_positions_for and attr in ['x', 'y', 'z']:
                continue  # Skip position attributes if we are ignoring them

            total_attributes += 1

            if attr in final_state[obj] and abs(final_state[obj][attr] - value) <= 0.01:
                matching_attributes += 1
            elif coarse_eval and obj.endswith('_center') and not location_evaluated[obj]:
                goal_location = get_general_location(attributes)
                final_location = get_general_location(final_state.get(obj, {}))
                if goal_location == final_location:
                    matching_attributes += 3
                location_evaluated[obj] = True

    if total_attributes == 0:
        return False, "No attributes to compare"
    
    accuracy = matching_attributes / total_attributes
    if accuracy == 1.0:
        return True, "Goal achieved"
    else:
        return False, f"Goal achieved {accuracy*100:.2f}%"



def goal_check_eval(goal_state, final_state, coarse_eval=False):
    converted_goal_state = convert_goal_state(goal_state)
    
    return compare_states(converted_goal_state, final_state, coarse_eval)

def extract_plan(plan_str):
    actions = []
    for line in plan_str.strip().split('\n'):
        if line == "[PLAN END]":
            break
        actions.append(line.strip())
    return actions

def inverse_action(action):
    action_map = {
        "toggled": "ToggleObject",
        "picked up": "PickupObject",
        "placed": "PutObject",
        "adjusted": "OpenObject",
        "moved": "NoOp",
        "did nothing": "NoOp"
    }

    object_map = {
        "no particular object": "NoObject1",
        "the microwave": "Microwave",
        "the toaster": "Toaster",
        "a wooden cabinet": "Cabinet_47fc321b",
        'the first stove knob for the frontright burner': 'StoveKnob_38c1dbc2',
        'the first stove knob for the front-right burner': 'StoveKnob_38c1dbc2',
        'the second stove knob for the frontleft burner': 'StoveKnob_690d0d5d',
        'the second stove knob for the front-left burner': 'StoveKnob_690d0d5d',
        'the third stove knob for the backleft burner': 'StoveKnob_c8955f66',
        'the third stove knob for the back-left burner': 'StoveKnob_c8955f66',
        'the fourth stove knob for the backright burner': 'StoveKnob_cf670576',
        'the fourth stove knob for the back-right burner': 'StoveKnob_cf670576',
        "the ceramic plate": "Plate",
        "up the ceramic plate": "Plate",
        "the spherical, brown, fragile Egg": "Egg",
        "the spherical brown fragile egg": "Egg",
        "the granite countertop": "CounterTop_f8092513",
        'the flat metal sturdy pan': 'Pan'
    }

    action_words = action.split()
    
    if action_words[1] == "did":
        action_type = "did nothing"
    elif action_words[1] == "picked" and action_words[2] == "up":
        action_type = "picked up"
    else:
        action_type = action_words[1]

    if action_type == "did nothing":
        return action_map[action_type], "NoObject1"

    if action_type == "picked up":
        object_description = " ".join(action_words[3:])
    elif action_type == "placed":
        object_description = ' '.join(action_words[action_words.index('on') + 1:])
    else:
        object_description = " ".join(action_words[2:])

    for description, obj in object_map.items():
        if description in object_description:
            return action_map[action_type], obj
    
    return None, None

def parse_state(state_str):
    state = defaultdict(lambda: defaultdict(dict))
    for item in state_str.split(' | '):
        parts = item.split(' is ')
        if len(parts) == 2:
            obj, state_desc = parts
            if obj.endswith(' x') or obj.endswith(' y') or obj.endswith(' z'):
                obj_name, coord = obj.rsplit(' ', 1)
                state[obj_name]['position'][coord] = float(state_desc.split()[-1])
            else:
                attribute = state_desc.split()[0].lower()
                if attribute in ['broken', 'unbroken']:
                    state[obj]['broken'] = attribute == 'broken'
                elif attribute in ['cooked', 'uncooked']:
                    state[obj]['cooked'] = attribute == 'cooked'
                elif attribute in ['on', 'off']:
                    state[obj]['on'] = attribute == 'on'
                elif attribute in ['open', 'closed']:
                    state[obj]['open'] = attribute == 'open'
                elif 'picked up' in state_desc:
                    state[obj]['picked_up'] = 'not' not in state_desc
                else:
                    state[obj][attribute] = state_desc

    # Convert defaultdict to regular dict
    return {k: dict(v) for k, v in state.items()}

def initialize_simulation(init_state, objects_state):
    attempts = 0
    controller, event = initialize_environment(40)
    
    keywords = ['NoObject1', 'Microwave', 'Toaster', 'Potato', 'Pan', 'Egg_afaaaca3', 'Plate_49b95a7a', 'Egg_afaaaca3']

    filtered_objects = [obj for obj in objects_state if any(keyword in obj['name'] for keyword in keywords)]

    object_poses = []
    for obj in filtered_objects:
        object_pose = {
            "objectName": obj['name'],
            "rotation": {
                "x": obj['rotation']['x'],
                "y": obj['rotation']['y'],
                "z": obj['rotation']['z']
            },
            "position": {
                "x": obj['position']['x'],
                "y": obj['position']['y'],
                "z": obj['position']['z']
            }
        }
        object_poses.append(object_pose)

    # Set object poses
    controller.step(action='SetObjectPoses', objectPoses=object_poses)
    event = controller.step(action='Stand')
    controller.step(action='SetObjectPoses', objectPoses=object_poses)
    event = controller.step(action='Stand')
    
    # Handle egg first
    if 'Egg' in init_state:
        attributes = init_state['Egg']
        broken = attributes.get('broken', False)
        cooked = attributes.get('cooked', False)
        if broken and not cooked:
            event, _ = perform_action(controller, 'PickupObject', 'Egg', event, 0)
            event = controller.step(action='PutObject', objectId=get_object_id(event, 'Pan')[0])
            event = controller.step(action='BreakObject', objectId=get_object_id(event, 'Egg')[0])
            for _ in range(5):
                event = controller.step(action='Stand')
                event = controller.step(action='Stand')
        elif broken and cooked:
            event, _ = perform_action(controller, 'ToggleObject', 'StoveKnob_690d0d5d', event, 0)
            event, _ = perform_action(controller, 'PickupObject', 'Egg', event, 0)
            event = controller.step(action='PutObject', objectId=get_object_id(event, 'Pan')[0])
            event = controller.step(action='BreakObject', objectId=get_object_id(event, 'Egg')[0])
            for _ in range(5):
                event = controller.step(action='Stand')
                event = controller.step(action='Stand')
            event, _ = perform_action(controller, 'ToggleObject', 'StoveKnob_690d0d5d', event, 0)



    for obj, attributes in init_state.items():
        obj_name = get_controller_object_name(obj)
        
        if attributes.get('open', False):
            event, _ = perform_action(controller, 'OpenObject', obj_name, event, 0)
        if attributes.get('on', False):
            event, _ = perform_action(controller, 'ToggleObject', obj_name, event, 0)
        if attributes.get('picked_up', False):
            event, _ = perform_action(controller, 'PickupObject', obj_name, event, 0)

    event = controller.step(action='Stand')
    event = controller.step(action='Stand')

    return controller, event

def get_controller_object_name(parsed_name):
    object_map = {
        "no particular object": "NoObject1",
        "the microwave": "Microwave",
        "the toaster": "Toaster",
        "a wooden cabinet": "Cabinet_47fc321b",
        'the first stove knob for the frontright burner': 'StoveKnob_38c1dbc2',
        'the first stove knob for the front-right burner': 'StoveKnob_38c1dbc2',
        'the second stove knob for the frontleft burner': 'StoveKnob_690d0d5d',
        'the second stove knob for the front-left burner': 'StoveKnob_690d0d5d',
        'the third stove knob for the backleft burner': 'StoveKnob_c8955f66',
        'the third stove knob for the back-left burner': 'StoveKnob_c8955f66',
        'the fourth stove knob for the backright burner': 'StoveKnob_cf670576',
        'the fourth stove knob for the back-right burner': 'StoveKnob_cf670576',
        "Egg": "Egg_afaaaca3",
        "Plate": "Plate_49b95a7a",
        "the ceramic plate": "Plate_49b95a7a",
        "the spherical, brown, fragile Egg": "Egg_afaaaca3",
        "the granite countertop": "CounterTop_f8092513"
    }
    return object_map.get(parsed_name, parsed_name)

def execute_plan(controller, event, action_plan):
    actions = action_plan.split('\n')
    for step_number, action in enumerate(actions):
        action_type, object_name = inverse_action(action)
        event, action_pos = perform_action(controller, action_type, object_name, event, step_number)
        event = controller.step(action='Stand')
        event = controller.step(action='Stand')

    objects_state = controller.last_event.metadata['objects']
    for obj in objects_state:
        if obj['name'] == 'Egg_afaaaca3':
            obj['visible'] = True
    final_state = get_environment_state(event)

            
    simplified_state, simplified_keys = simplify_latents(np.fromiter(final_state.values(), dtype=float)[None], list(final_state.keys()))
    
    structured_final_state = defaultdict(dict)
    for i, key in enumerate(simplified_keys):
        obj, attr = key.rsplit('_', 1)
        structured_final_state[obj][attr] = simplified_state[0][i]
    
    # Don't ask me why this is necessary
    if structured_final_state['Egg_afaaaca3']['broken'] == 2:
        structured_final_state['Egg_afaaaca3']['broken'] = 1
    
    return dict(structured_final_state), objects_state
    
def eval_output(init_state_str, action_plan, goal_state_str, objects_state):
    init_state = parse_state(init_state_str)
    # print("Initial state:", json.dumps(init_state, indent=2))
    goal_state = parse_state(goal_state_str)
    # print("Goal state:", json.dumps(goal_state, indent=2))
    # print("Plan:", action_plan)

    controller, event = initialize_simulation(init_state, objects_state)
    final_state, objects_state = execute_plan(controller, event, action_plan)
    
    # print("Final state:", json.dumps(final_state, indent=2))
    
    for _ in range(15):
        event = controller.step(action='Stand')
        event = controller.step(action='Stand')

    success, message = goal_check_eval(goal_state, final_state, coarse_eval=True)
    # if not success:
    #     print(f"Goal check failed: {message}")
    # else:
    #     print("Goal achieved successfully")
    
    controller.stop()
    del controller, event, 
    gc.collect()
    # pkill -9 -f thor-Linux64-f0
    Popen("pkill -9 -f thor-Linux64-f0", shell=True, stdout=PIPE, stderr=PIPE)
    
    return success, objects_state, final_state

def evaluate_trajectory(trajectory):
    object_states = trajectory['object_states'][0][0]
    init_state_str = trajectory['init']
    action_plan = trajectory['plan'].strip('\n[PLAN END]\n').replace('carefully ', '').replace('skillfully ', '')
    goal_state_str = trajectory['goal']
    result, state, final_state = eval_output(init_state_str, action_plan, goal_state_str, object_states)
    return result, state, final_state

def evaluate_estimated_trajectory(trajectory):
    object_states = trajectory['object_states'][0][0]
    init_state_str = trajectory['init']
    uid = trajectory['uid']
    action_plan = open(f'output/{uid}.txt', 'r').read().strip('\n[PLAN END]\n').replace('carefully ', '').replace('skillfully ', '')
    goal_state_str = trajectory['goal']
    result, state, final_state = eval_output(init_state_str, action_plan, goal_state_str, object_states)
    return result, state, final_state, action_plan

def eval_accuracy(action_plans_dir, data_path, config_path):
    from glob import glob

    action_plan_files = glob(f"{action_plans_dir}/*.txt")
    total_plans = len(action_plan_files)
    successful_plans = 0

    trajectories = load_ithor_data(data_path, config_path, return_intermediate=True, return_object_states=True)


    for action_plan_file in tqdm(action_plan_files, total=total_plans):
        uid = action_plan_file.split('/')[-1].split('.')[0]
        with open(action_plan_file, 'r') as f:
            action_plan = f.read().strip('\n[PLAN END]\n').replace('carefully ', '').replace('skillfully ', '')

        trajectory = get_trajectory(trajectories, uid)
        with reset_globals():
            try:
                result, state, final_state = eval_output(trajectory['init'], action_plan, trajectory['goal'], trajectory['object_states'][0][0])
                if result:
                    successful_plans += 1
            except Exception as e:
                total_plans -= 1
                with open('errors.txt', 'a') as f:
                    f.write(f"Error in trajectory {uid}: {e}\n")

    accuracy = successful_plans / total_plans if total_plans > 0 else 0
    return accuracy

def get_trajectory(trajectories, uid):
    for trajectory in trajectories:
        if str(trajectory['uid']) == uid:
            return trajectory
    raise ValueError(f"Trajectory with UID {uid} not found")

if __name__ == '__main__':
    config_path = 'val_metadata.json'
    data_path = 'data/step_2_ws.pth'
    action_plans_dir = 'logs/ithor_MCTS/07312024-212324/action_plans'
    print(eval_accuracy(action_plans_dir, data_path, config_path))
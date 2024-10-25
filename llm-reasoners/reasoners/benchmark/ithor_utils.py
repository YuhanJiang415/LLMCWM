import torch
import webcolors
import re
import json
import numpy as np
from copy import deepcopy
from collections import defaultdict

def softmax(X, theta = 1.0, axis = None):
    """
    Taken from https://stackoverflow.com/a/42797620/

    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    X = np.array(X)
    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def describe_latent(latent, keys):
    base_object_translations = {
        "NoObject1": "with no particular object",
        "NoObject2": "with no particular object",
        "NoObject3": "with no particular object",
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
        # Extract the base entity and attribute from the key
        base_entity = "_".join(key.split('_')[:-1])
        entity = base_object_translations.get(base_entity, base_entity.split('_')[0])
        
        # Determine the attribute from the key
        attribute = key.split('_')[-1]
        
        # Create a meaningful description based on the attribute type
        if "center_x" in key or "center_y" in key or "center_z" in key:
            descriptions.append(f"{entity} {attribute.replace('_', ' ')} is at {value:.2f}")
        elif "open" in key or "on" in key or "cooked" in key or "broken" in key or "pickedup" in key:
            state = "open" if "open" in key and value > 0.5 else "closed" if "open" in key else \
                    "on" if "on" in key and value > 0.5 else "off" if "on" in key else \
                    "cooked" if "cooked" in key and value > 0.5 else "uncooked" if "cooked" in key else \
                    "broken" if "broken" in key and value > 0.5 else "unbroken" if "broken" in key else \
                    "picked up" if "pickedup" in key and value > 0.5 else "not picked up" if "pickedup" in key else "unknown state"
            if state == "open" or state == "closed":
                descriptions.append(f"{entity}'s door is {state}")
            else:
                descriptions.append(f"{entity} is {state}")
        else:
            descriptions.append(f"{entity} has an unknown attribute with value {value:.2f}")
    
    return " | ".join(descriptions)


def load_ithor_data(file_path, config, return_intermediate=False, return_object_states=False):
    # Load data using PyTorch's deserialization
    config = json.load(open(config, 'r'))
    keys = config['flattened_causals']
    trajectories = torch.load(file_path)

    data = []
    for images, actions, latents, uid, *object_states in trajectories:
        cur_data = {}
        # Generate descriptions for initial and final states
        cur_data['init'] = describe_latent(latents[0], keys)
        cur_data['goal'] = describe_latent(latents[-1], keys)
        cur_data['plan'] = "\n".join(actions) + "\n[PLAN END]\n"
        cur_data['uid'] = uid
        
        if return_intermediate:
            cur_data['states'] = [describe_latent(lat, keys) for lat in latents]
        
        cur_data['question'] = fill_template(cur_data['init'], cur_data['goal'], "")
        cur_data['images'] = images
        if return_object_states:
            cur_data['object_states'] = object_states
        data.append(cur_data)
    
    return data

def extract_init_state(example):
    """Extract the initial state from the example
    
    :param example: example
    """
    # print(example)
    init_statement = example["question"].split("[STATEMENT]\nAs initial conditions I have that, ")[1]\
        .split("My goal")[0].strip()
    return init_statement

def fill_template(INIT, GOAL, PLAN):
    text = ""
    if INIT != "":
        text += "\n[STATEMENT]\n"
        text += f"As initial conditions I have that, {INIT.strip()}."
    if GOAL != "":
        text += f"\nMy goal is to have that {GOAL}."
    text += f"\n\nMy plan is as follows:\n\n[PLAN]{PLAN}"

    # TODO: Add this replacement to the yml file -- Use "Translations" dict in yml
    text = text.replace("-", " ").replace("ontable", "on the table")
    return text

def generate_actions(cur_state):
    acd = {
        'PickupObject': ['Egg', 'Plate'],
        'PutObject': ['Microwave', 'CounterTop_f8092513'],
        'ToggleObject': [
            'Toaster', 'Microwave', 'StoveKnob_38c1dbc2',
            'StoveKnob_690d0d5d', 'StoveKnob_c8955f66', 'StoveKnob_cf670576'
        ],
        'OpenObject': ['Microwave', 'Cabinet_47fc321b'],
        'NoOp': ['NoObject1']
    }
    
    base_object_translations = {
        "NoObject1": "with no particular object",
        "Microwave": "the microwave",
        "Toaster": "the toaster",
        "Cabinet_47fc321b": "a wooden cabinet",
        "StoveKnob_38c1dbc2": "the first stove knob for the front-right burner",
        "StoveKnob_690d0d5d": "the second stove knob for the front-left burner",
        "StoveKnob_c8955f66": "the third stove knob for the back-left burner",
        "StoveKnob_cf670576": "the fourth stove knob for the back-right burner",
        "Plate": "the ceramic plate",
        "Egg": "the spherical, brown, fragile Egg",
        "CounterTop_f8092513": "the granite countertop"
    }
    
    base_action_translations = {
        "PickupObject": "picked up",
        "PutObject": "placed",
        "ToggleObject": "toggled",
        "OpenObject": "adjusted",
        "NoOp": "did nothing"
    }
    
    # Parse current state
    state = dict([item.strip().split(' is ') for item in cur_state.split(' | ')])
    
    # Initialize the states based on current state description
    state['Plate picked up'] = 'Plate is picked up' in cur_state
    state['Egg picked up'] = 'Egg is not picked up' not in cur_state
    state['Microwave is on'] = 'Microwave is on' in cur_state
    state['Microwave is closed'] = 'Microwave is closed' in cur_state
    state['Toaster is on'] = 'Toaster is on' in cur_state
    
    actions = []
    
    possible_actions = deepcopy(acd)
    
    # If holding plate or egg, cannot pick up anything else
    if state['Plate picked up'] or state['Egg picked up']:
        possible_actions['PickupObject'] = []
    
    # Can only put/place things down on the countertop if holding them
    if not (state['Plate picked up'] or state['Egg picked up']):
        possible_actions['PutObject'] = []
    
    # When microwave is open, we cannot toggle it on
    if not state['Microwave is closed']:
        possible_actions['ToggleObject'].remove('Microwave')
    
    # When microwave is toggled on, we cannot open it
    if state['Microwave is on']:
        possible_actions['OpenObject'].remove('Microwave')
    
    # Can put/place things in the microwave only if holding them AND the microwave is open
    if not (state['Plate picked up'] or state['Egg picked up']) or state['Microwave is closed']:
        if 'Microwave' in possible_actions['PutObject']:
            possible_actions['PutObject'].remove('Microwave')
    
    for action, objects in possible_actions.items():
        for obj in objects:
            if action == 'OpenObject' and obj == 'Microwave':
                actions.append(f"You adjusted the microwave's door")
            elif action == 'OpenObject' and obj == 'Cabinet_47fc321b':
                actions.append(f"You adjusted the wooden cabinet's door")
            elif action == 'PutObject':
                if obj == 'Microwave':
                    actions.append(f"You placed the {base_object_translations['Plate' if state['Plate picked up'] else 'Egg']} in the microwave")
                elif obj == 'CounterTop_f8092513':
                    actions.append(f"You placed the {base_object_translations['Plate' if state['Plate picked up'] else 'Egg']} on the granite countertop")
            elif action == 'ToggleObject' and obj == 'Microwave':
                actions.append(f"You toggled the microwave's heating element")
            else:
                actions.append(f"You {base_action_translations[action]} {base_object_translations[obj]}")
    
    return actions

def inverse_action(cur_state, action):
    action_map = {
        "picked up": "PickupObject",
        "picked": "PickupObject",
        "placed": "PutObject",
        "toggled": "ToggleObject",
        "adjusted": "OpenObject",
        "did nothing": "NoOp",
        "moved": "NoOp",
    }
    
    object_map = {
        "no particular object": "NoObject1",
        "the microwave": "Microwave",
        "the toaster": "Toaster",
        "a wooden cabinet": "Cabinet_47fc321b",
        "the first stove knob for the front-right burner": "StoveKnob_38c1dbc2",
        "the first stove knob for the frontright burner": "StoveKnob_38c1dbc2",
        "the second stove knob for the front-left burner": "StoveKnob_690d0d5d",
        "the second stove knob for the frontleft burner": "StoveKnob_690d0d5d",
        "the third stove knob for the back-left burner": "StoveKnob_c8955f66",
        "the third stove knob for the backleft burner": "StoveKnob_c8955f66",
        "the fourth stove knob for the back-right burner": "StoveKnob_cf670576",
        "the fourth stove knob for the backright burner": "StoveKnob_cf670576",
        "the ceramic plate": "Plate",
        "the spherical, brown, fragile Egg": "Egg",
        "the spherical, brown, fragile egg": "Egg",
        "the spherical brown fragile egg": "Egg",
        "the granite countertop": "CounterTop_f8092513"
    }
    
    action_words = action.split()
    action_type = action_words[1]
    if action_words[1] == "did" or action_words[0] == "moved":
        return "NoOp", "NoObject1"
    
    object_description = " ".join(action_words[2:])
    
    for description, obj in object_map.items():
        if description in object_description:
            return action_map[action_type], obj

        
# def extract_goals(example, return_raw=False):
#     """Extract the goals from the example
    
#     :param example: example
#     """
#     goal_statement = example["question"].split("[STATEMENT]")[-1]\
#         .split("My goal is to ")[1].split("My plan is as follows")[0].strip()
#     if return_raw:
#         return goal_statement
#     #TODO regex parse goal statement.
# #     goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
#     return goals

import re

def extract_goals(example, return_raw=False):
    """Extract the goals from the example.
    
    :param example: dict with 'question' key containing goal and plan statements
    :param return_raw: if True, returns the raw goal statement
    :return: Either raw goal string or parsed goals as a dictionary
    """
    goal_statement = example["question"].split("[STATEMENT]")[-1]\
        .split("My goal is to ")[1].split("My plan is as follows")[0].strip()
    
    if return_raw:
        return goal_statement
    
    # Regex pattern to match the identifier, attribute, and value accurately
    goal_statement = goal_statement.lstrip('I have that')
    goals = dict([item.strip().split(' is ') for item in goal_statement.split(' | ')])
    
    # # Initialize the states based on current state description
    # state['Plate picked up'] = 'Plate is picked up' in cur_state
    # state['Egg picked up'] = 'Egg is not picked up' not in cur_state
    # state['Microwave is on'] = 'Microwave is on' in cur_state
    # state['Microwave is closed'] = 'Microwave is closed' in cur_state
    # state['Toaster is on'] = 'Toaster is on' in cur_state
    # goals = {}
    
    # for match in re.finditer(pattern, goal_statement):
    #     key = f"{match.group(1)} {match.group(2)} {match.group(3)}"
    #     value = float(match.group(4).rstrip('.'))
    #     goals[key] = value
    
    return goals

def extract_init_state_dict(example):
    """Extract the initial state from the example
    
    :param example: example
    """
    init_statement = example["question"].split("[STATEMENT]\nAs initial conditions I have that, ")[1]\
        .split("My goal")[0].strip()
    
    # Regex pattern to match the identifier, attribute, and value accurately
    pattern = r"(\w+)\s(\w+)\s(position\s\w+|state)\sis\s([\d\.]+)"
    init_states = {}
    
    for match in re.finditer(pattern, init_statement):
        key = f"{match.group(1)} {match.group(2)} {match.group(3)}"
        value = float(match.group(4).rstrip('.'))
        init_states[key] = value
    
    return init_states

def goal_check(goals, description, epsilon=0.07, ignore_obstacles=False):
    """Check if the description matches the goals with a tolerance and return the percentage of goals met.
    
    :param goals: dictionary of goal states
    :param description: description string containing current states
    :param epsilon: tolerance for numeric comparisons
    :param ignore_obstacles: if True, obstacle positions are ignored in the check
    :return: Tuple (boolean, float) where boolean is True if all goals are met within tolerance, and float is the percentage of goals met
    """
    # Parse the description into a dictionary
    # pattern = r"(\w+)\s(\w+\s\w+)\sis\s([\d\.]+)(?=\,|\s|$)"
    # current_states = {match.group(1) + ' ' + match.group(2): float(match.group(3).rstrip('.')) for match in re.finditer(pattern, description)}
    pattern = r"(\w+)\s(\w+)\s(position\s\w+|state)\sis\s([\d\.]+)"
    current_states = {}
    
    for match in re.finditer(pattern, description):
        key = f"{match.group(1)} {match.group(2)} {match.group(3)}"
        value = float(match.group(4).rstrip('.'))
        current_states[key] = value
    
    total_goals = 0
    met_goals = 0
    for key, goal_value in goals.items():
        if ignore_obstacles and 'obstacle' in key:
            continue
        total_goals += 1
        if key in current_states and (abs(current_states[key] - goal_value) <= epsilon):
            met_goals += 1

    all_goals_met = met_goals == total_goals
    percentage_met = (met_goals / total_goals) * 100 if total_goals > 0 else 0
    return all_goals_met, percentage_met



def goal_check_eval(goals, causal_dict, epsilon=0.07, ignore_obstacles=False):
    """Check if the description matches the goals with a tolerance and return the percentage of goals met.
    
    :param goals: dictionary of goal states
    :param causal_dict: dictionary of current causal variables
    :param epsilon: tolerance for numeric comparisons
    :param ignore_obstacles: if True, obstacle positions are ignored in the check
    :return: Tuple (boolean, float) where boolean is True if all goals are met within tolerance, and float is the percentage of goals met
    """
    total_goals = 0
    met_goals = 0
    for key, goal_value in goals.items():
        if ignore_obstacles and 'obstacle' in key:
            continue
        total_goals += 1
        if key in causal_dict and (abs(causal_dict[key] - goal_value) <= epsilon):
            met_goals += 1

    all_goals_met = met_goals == total_goals
    percentage_met = (met_goals / total_goals) * 100 if total_goals > 0 else 0
    return all_goals_met, percentage_met

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
        
        # These have been absolute hell to get right.
        
        if (-0.97 <= x <= -0.67) and (-0.04 <= y <= 0.26) and (-0.49 <= z <= -0.19):
            return 'on hand'
        
        if (0.49 <= z <= 0.79) and (-0.46 <= x <= -0.06) and (-1.2 <= y <= -0.8):
            return 'on counter'
        
        if (0.27 < x < 0.47) and (-0.93 < y < -0.73) and (-0.93 < z < -0.73):
            return 'in pan'
    
    return 'other'

def compare_states(goal_state, final_state, coarse_eval=True):
    total_attributes = 0
    matching_attributes = 0
    location_evaluated = {k: False for k in goal_state.keys() if k.endswith('_center')}
    ignore_positions_for = set()

    # First pass to identify objects with position attributes to ignore
    for obj, attributes in goal_state.items():
        if obj not in final_state:
            continue
        
        for attr in ['broken', 'cooked', 'pickedup']:
            if attr in attributes and final_state[obj].get(attr) == attributes[attr]:
                matching_attributes += 1
                total_attributes += 1
                ignore_positions_for.add(obj.replace('_center', ''))
                break  # Since we only need one attribute match to ignore positions

    # Second pass to evaluate attributes
    for obj, attributes in goal_state.items():
        if obj not in final_state:
            continue
        
        for attr, value in attributes.items():
            if obj.replace('_center', '') in ignore_positions_for and attr in ['x', 'y', 'z']:
                # continue  # Skip position attributes if we are ignoring them
                if coarse_eval and obj.endswith('_center') and not location_evaluated[obj]:
                                goal_location = get_general_location(attributes)
                                final_location = get_general_location(final_state.get(obj, {}))
                                if goal_location == final_location:
                                    matching_attributes += 3
                                location_evaluated[obj] = True
                total_attributes += 3
                continue
            # total_attributes += 1

            if attr in final_state[obj] and abs(final_state[obj][attr] - value) <= 0.01:
                matching_attributes += 1
            
            # elif coarse_eval and obj.endswith('_center') and not location_evaluated[obj]:
            #     goal_location = get_general_location(attributes)
            #     final_location = get_general_location(final_state.get(obj, {}))
            #     if goal_location == final_location:
            #         matching_attributes += 3
            #     location_evaluated[obj] = True

    if total_attributes == 0:
        return False, "No attributes to compare"
    
    accuracy = matching_attributes / total_attributes
    if accuracy == 1.0:
        return True, "Goal achieved"
    else:
        return False, accuracy
    
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

def convert_state(goal_state):
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


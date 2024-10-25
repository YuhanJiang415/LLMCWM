import numpy as np
import random
import webcolors

GRAMMAR = {
    "ADJECTIVES": {
        "traffic light": [
            ("sturdy, illuminated", 0.2), ("metal, tall", 0.2), ("gleaming, automated", 0.2), ("durable, weather-resistant", 0.2)
        ],
        "vehicle": [
            ("sleek, aerodynamic", 0.2), ("speedy, high-performance", 0.2), ("luxurious, comfortable", 0.2), ("compact, fuel-efficient", 0.2), ("rugged, all-terrain", 0.2)
        ],
        "obstacle": [
            ("solid, rocky", 0.2), ("visible, brightly colored", 0.2), ("large, obstructive", 0.2), ("big, heavy", 0.2),
        ]
    },
    "ACTION_MODIFIER": [
        ("skillfully", 0.2), ("efficiently", 0.2), ("carefully", 0.2), ("precisely", 0.2), ("quickly", 0.2)
    ]
}

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_tuple):
    try:
        color_name = webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        color_name = closest_color(rgb_tuple)
    return color_name

def weighted_choice(choices):
    phrases, weights = zip(*choices)
    total = sum(weights)
    probs = [w / total for w in weights]
    return np.random.choice(phrases, p=probs)

def generate_description_probabilistic(action, object_type, color, grammar, use_direction=True):
    movement_actions = ["moved left", "moved right", "moved up", "moved down", "turned left", "turned right", "turned up", "turned down"]
    is_movement_action = any(movement in action for movement in movement_actions)

    action_modifier = weighted_choice(grammar["ACTION_MODIFIER"]) + ' '
    if random.random() < 0.7:
        action_modifier = ''
    num_adjectives = random.randint(0, 1)
    adjectives_list = grammar["ADJECTIVES"][object_type]
    adjectives = ', '.join(weighted_choice(adjectives_list) for _ in range(num_adjectives))

    if is_movement_action:
        verb, direction = action.split(" ")[0], " ".join(action.split(" ")[1:])
        if use_direction:
            full_description = f"You {action_modifier}{verb} the {adjectives + ', ' if adjectives else ''}{color} {object_type} {direction}."
        else:
            full_description = f"You {action_modifier}{verb} the {adjectives + ', ' if adjectives else ''}{color} {object_type} {direction}."
    else:
        full_description = f"You {action_modifier}{action} the {adjectives + ', ' if adjectives else ''}{color} {object_type}."
    
    return full_description

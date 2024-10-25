import datasets
import json
from tqdm import tqdm
import torch
import os, pickle
from datetime import datetime
import sys
import random
from reasoners import Evaluator
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../data_generation'))
# from gridworld import Gridworld, TrafficLight, GridEntity
import numpy as np
import reasoners.benchmark.ithor_utils as ithor_utils

def rap_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # to make sure the plan is saved before evaluation in multi-process setting
    try:
        if algo_output.trace is None:
            print("No plan found")
            return ""
        else:
            return "\n".join(algo_output.trace[1])
    except Exception as e:
        print("Error in output extraction,", e)
        return ""

def get_icl(init_prompt, examples):
    icl = init_prompt["intro"] + \
        "\n".join([
            "[STATEMENT]\nAs initial conditions I have that, " + \
            example["init"] + \
            ".\nMy goal is to have that " +\
            example["goal"] + \
            ".\n\nMy plan is as follows:\n\n[PLAN]" + \
            example["plan"]
            for example in examples
        ])
    icl += "\n[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>\n\nMy plan is as follows:\n\n[PLAN]<action>"
    return icl

class ITHOREvaluator(Evaluator):
    def __init__(self, 
                 config_file,
                 data_path,
                 init_prompt,
                 disable_log=False,
                 disable_tqdm=False,
                 output_extractor=rap_bw_extractor,
                 answer_extractor=lambda x:x,
                 sample_prompt_type="rap",
                 output_dir='examples/ithor/output/') -> None:

        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x
        self.full_dataset = ithor_utils.load_ithor_data(data_path, config_file, return_intermediate=True) # [{"goal": str, "init": str}]
        self._dataset_name = 'ithor'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type
        self.output_dir = output_dir

        # self.lm_plan_file = "tmp_plan.txt"
        self.config_file = config_file
        self.config = json.load(open(config_file, 'r'))
        self._init_ithor()

    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4):

        sample_prompt_type = self.sample_prompt_type
        if sample_prompt_type == "rap":
            if shuffle_prompt:
                examples = random.sample(self.init_prompt["example_pool"], num_shot)
            else:
                examples = self.init_prompt["example_pool"][:num_shot]

            icl = get_icl(self.init_prompt, examples)
            
            prompt = copy.deepcopy(self.init_prompt)
            prompt["icl"] = icl
            prompt["icl_list"] = [icl]
            examples = copy.deepcopy(examples)
            for i in range(5):
                new_examples = []
                for example in examples:
                    if len(example["states"]) > 1:
                        new_examples.append({
                            "init": example["states"][0],
                            "goal": example["goal"],
                            "plan": "\n" + "\n".join(example["plan"].split("\n")[3:]),
                            "states": example["states"][1:]
                        })
                    else:
                        new_examples.append(example)
                examples = copy.deepcopy(new_examples)
                icl = get_icl(self.init_prompt, examples)
                prompt["icl_list"].append(icl)
        else:
            raise NotImplementedError
        # print("prompt:",  prompt)
        return prompt
    
    def eval_output(self, answer, output, cm_satisfied: bool):
        # validate_plan takes as input the (processed by output_extractor) text output, simulates the actions
        # and checks whether we end up with the right end state
        # print("answer:", answer)
        # goals = ithor_utils.extract_goals(answer)
        # action_plan = ithor_utils.extract_plan(output)
        # init_state = ithor_utils.parse_state(answer['init'])
        # goal_state = ithor_utils.parse_state(answer['goal'])
        
        # controller, event = ithor_utils.initialize_simulation(init_state)
        # final_state = ithor_utils.execute_plan(controller, event, action_plan)
        # return ithor_utils.goal_check_eval(goal_state, final_state, ignore_movement=True)[0]
        goal = answer['goal']
        init_state = answer['init']
        action_plan = output
        uid = answer['uid']
        # Save the plan to a file
        # make sure the output_dir exists, if not create it
        if not os.path.exists(f'{self.log_dir}/action_plans/'):
            os.makedirs(f'{self.log_dir}/action_plans/')
        print(f"Saving action plan to {self.log_dir}/action_plans/{uid}.txt")
        with open(f"{self.log_dir}/action_plans/{uid}.txt", 'w') as f:
            f.write(action_plan)
        return True
        
        # return True
        # init_state_dict = gw_utils.extract_init_state_dict(answer)
        # keys = self.config['flattened_causals']
        # init_state_dict = {k:v for k,v in zip(keys, init_state_dict.values())}
        # goals = {k:v for k,v in zip(keys, goals.values())}
        # # TODO: make this loadaable from the config
        # fixed_car_pos = {
        #     'vehicle_(0, 0, 255)_position_x': 0.42857142857142855,
        #     'vehicle_(192, 192, 192)_position_x': 0.7142857142857143,
        #     'vehicle_(255, 0, 0)_position_x': 0.0,
        # }
        # orientations = {'vehicle_(0, 0, 255)_orientation': 'down',
        #                 'vehicle_(192, 192, 192)_orientation': 'up',
        #                 'vehicle_(255, 0, 0)_orientation': 'up',}
        # self.gridworld.initialize_from_causal_dict(init_state_dict, fixed_car_positions=fixed_car_pos, fixed_car_orientations=orientations)
        # print(output)
        # action_codes = self.gridworld.parse_action_strings(output)
        # self.gridworld.execute_actions(action_codes)
        # new_causals = self.gridworld.get_causal_vector()
        # # TODO: make this loadaable from the config
        # new_causals = np.array(new_causals)[[0, 1, 2, 3, 4, 6, 8, 10]]
        # new_causals = {k:v for k,v in zip(keys, new_causals)}
        # if cm_satisfied and not gw_utils.goal_check_eval(goals, new_causals, ignore_obstacles=True)[0]:
        #     print("Causal model satisfied but goal not achieved")
        #     print("Goals:", goals)
        #     print("New causals:", new_causals)
        # return gw_utils.goal_check_eval(goals, new_causals, ignore_obstacles=True)[0]
        # return True


    def _init_ithor(self):
        pass
        # car_colors, light_colors, boulder_colors = gw_utils.extract_colors(self.config['causal_keys'])
        # colors_dict = {
        #     'cars': car_colors,
        #     'lights': light_colors,
        #     'boulders': boulder_colors
        # }
        # orientations = ['up', 'down', 'left', 'right']
        # # TODO: make this loadaable from the config
        # fixed_light_positions = [(0, 0, 'down'), (3, self.config['grid_y'] - 1, 'up'), (self.config['grid_x'] - 3, 0, 'down')]
        # light_colors_iter = iter(light_colors)
        # GridEntity.preload_sprites(colors_dict, orientations, sprite_path='../data_generation/sprites/', sprite_size=self.config['sprite_size'])
        # self.gridworld = Gridworld(self.config['grid_x'], self.config['grid_y'])
        # for (light_x, light_y, light_orientation), light_color in zip(fixed_light_positions, light_colors_iter):
        #     light = TrafficLight(light_x, light_y, 'red', light_color, light_orientation, frequency=(100, 1))
        #     self.gridworld.add_entity(light)

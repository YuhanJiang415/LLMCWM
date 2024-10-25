import torch
from typing import Union, Tuple, NamedTuple, Callable
import reasoners.benchmark.ithor_utils as utils
from reasoners import WorldModel, LanguageModel
import copy
import json
import os
import random
# Append ../../../data_generation to path
import sys

ITHAction = str

        
class ITHState(NamedTuple):
    """The state of the Blocksworld.
    
    See the docstring of BlocksWorldModel for more details.
    """
    step_idx: int
    image: torch.Tensor
    description: str
    latents: torch.Tensor = None

class CausalWorldModel(WorldModel):
    def __init__(self, crl_model, causal_mapper, nl_model, tokenizer, device, max_steps=6, config_file=None):
        super().__init__()
        # self.autoencoder = autoencoder
        self.crl_model = crl_model.eval()
        self.causal_mapper = causal_mapper
        # self.causal_mapper = CausalMapper(causal_mapper.to(self.crl_model.device), cm_mean, cm_std, target_assignment)
        self.nl_model = nl_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_steps = max_steps
        self.config = json.load(open(config_file, 'r'))
        self.keys = self.config['flattened_causals']

    def init_state(self, initial_image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Initialize the state with an image, encode it to latent, transform it,
        and generate the natural language description of the initial state.
        """
        # initial_image = (initial_image * 2.0) - 1.0
        latents = self.crl_model.autoencoder.encoder(initial_image[None].to(self.device))
        disentangled_latents, _ = self.crl_model.flow.forward(latents)
        causal_variables = self.causal_mapper(disentangled_latents)
        description = self.map_to_language(causal_variables)
        return (disentangled_latents, description)

    @torch.no_grad()
    def step(self, state: ITHState, action: str) -> Tuple[ITHState, dict]:
        """
        Update the state based on the action.
        """
        if state.latents is None:
            # image = (state.image * 2.0) - 1.0
            image = state.image
            current_latents = self.crl_model.autoencoder.encoder(image[None].to(self.device))
            current_latents, _ = self.crl_model.flow.forward(current_latents)
        else:
            current_latents = state.latents
        tokenized_description = self.tokenizer(action, return_token_type_ids=True, padding='max_length', max_length=64)
        input_ids = torch.tensor(tokenized_description['input_ids']).to(self.device)
        token_type_ids = torch.tensor(tokenized_description['token_type_ids']).to(self.device)
        attention_mask = torch.tensor(tokenized_description['attention_mask']).to(self.device)
        tokenized_description = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        new_latents, _ = self.crl_model.prior_t1.sample(current_latents, tokenized_description=tokenized_description, action=torch.empty(1).to(self.device))
        new_latents = new_latents.squeeze(1)
        causal_variables = self.causal_mapper(new_latents)
        if len(causal_variables) == 1 and isinstance(causal_variables, list):
            causal_variables = causal_variables[0]
        new_description = self.map_to_language(causal_variables)
        new_state = ITHState(step_idx=state.step_idx + 1, image=None, description=new_description, latents=new_latents)
        # return new_state, {'goal_reached' : utils.goal_check(utils.extract_goals(self.example), new_description, ignore_obstacles=True)}
        return new_state, {'goal_reached' : utils.compare_states(utils.convert_state(utils.parse_state(self.example['goal'])), utils.convert_state(utils.parse_state(new_description)))}

    def is_terminal(self, state: Tuple[torch.Tensor, str]) -> bool:
        parsed_goal_state = utils.parse_state(self.example['goal'])
        converted_goal_state = utils.convert_state(parsed_goal_state)
        parsed_final_state = utils.parse_state(state.description)
        converted_final_state = utils.convert_state(parsed_final_state)
        res = utils.compare_states(converted_goal_state, converted_final_state)
        # if utils.goal_check(utils.extract_goals(self.example), state.description, ignore_obstacles=True)[0]:
        #     return True
        if res[0]:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False

    def map_to_language(self, causals: torch.Tensor) -> str:
        """
        Map the causal variables to a natural language description (TODO: using the language model.)
        """
        return utils.describe_latent(causals, self.keys)

    def init_state(self) -> ITHState:
        """Initialize the world model.

        :return: the initial state
        """
        return ITHState(step_idx=0, image=self.example['images'][0], description=utils.
                       extract_init_state(self.example))

class LMWorldModel(WorldModel):
    def __init__(self, lm_model: LanguageModel, prompt: dict,
                 device, max_steps=6):
        super().__init__()
        self.lm_model = lm_model
        self.prompt = prompt
        self.device = device
        self.max_steps = max_steps
 
    def init_state(self) -> ITHState:
        return ITHState(
            step_idx=0,
            image=self.example['images'][0],
            description=utils.extract_init_state(self.example)
        )

    @torch.no_grad()
    def step(self, state: ITHState, action: str) -> Tuple[ITHState, dict]:
        """
        Update the state based on the action.
        """
        state = copy.deepcopy(state)
        description = state.description
        step_idx = state.step_idx
        if ("picked up" and "Egg") in action or ("picked up" and "egg") in action:
            key = "world_update_pickup_egg"
        elif "picked up the ceramic plate" in action:
            key = "world_update_pickup_plate"
        elif "placed" in action and "microwave" in action:
            key = "world_update_put_microwave"
        elif "placed" in action and "granite countertop" in action:
            key = "world_update_put_countertop"
        elif "placed" in action and "pan" in action:
            key = "world_update_put_pan"
        elif "toggled the toaster" in action:
            key = "world_update_toggle_toaster"
        elif "toggled the microwaves heating element" in action:
            key = "world_update_toggle_microwave"
        elif "toggled the first stove knob" in action:
            key = "world_update_toggle_stoveknob"
        elif "toggled the second stove knob" in action:
            key = "world_update_toggle_stoveknob"
        elif "toggled the third stove knob" in action:
            key = "world_update_toggle_stoveknob"
        elif "toggled the fourth stove knob" in action:
            key = "world_update_toggle_stoveknob"
        elif "adjusted" in action and "microwave" in action:
            key = "world_update_open_microwave"
        elif "adjusted" in action and "wooden" in action:
            key = "world_update_open_cabinet"
        elif "skillfully moved the ceramic plate" in action:
            key = "world_update_move_plate"
        elif "skillfully moved the spherical brown fragile egg" in action:
            key = "world_update_move_egg"
        elif "did nothing" in action:
            key = "world_update_no_action"
        else:
            raise ValueError(f"Unknown action: {action}")

        world_update_prompt = self.prompt[key].replace("<state>", description).replace("<action>\n", action)
        new_description = self.lm_model.generate(
            [world_update_prompt],
            eos_token_id=self.lm_model.tokenizer.eos_token_id,
            hide_input=True,
            temperature=0.0, # greedy decoding
        ).text[0].strip()
        new_description = new_description.replace(world_update_prompt, '').split('[SCENARIO 3]')[0].lstrip(' ').rstrip('\n')
        new_state = ITHState(step_idx=step_idx+1, image=None, description=new_description)
        try:
            gr = utils.compare_states(utils.convert_state(utils.parse_state(self.example['goal'])), utils.convert_state(utils.parse_state(new_description)))
        except:
            gr = False, 0.0
        return new_state, {'goal_reached' : gr}

    def is_terminal(self, state: Tuple[torch.Tensor, str]) -> bool:
        if utils.goal_check(utils.extract_goals(self.example), state.description, ignore_obstacles=True)[0]:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False
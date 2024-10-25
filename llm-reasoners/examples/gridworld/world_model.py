import torch
from typing import Union, Tuple, NamedTuple, Callable
from reasoners.benchmark import gw_utils as utils
from reasoners import WorldModel, LanguageModel
import copy
import json
import os
# Append ../../../data_generation to path
import sys


GWAction = str

        
class GWState(NamedTuple):
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
        self.crl_model = crl_model.eval()
        self.causal_mapper = causal_mapper
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
        if initial_image.min().item() >= -0.5:
            initial_image = (initial_image * 2.0) - 1.0
        latents = self.crl_model.autoencoder.encoder(initial_image[None].to(self.device))
        disentangled_latents, _ = self.crl_model.flow.forward(latents)
        causal_variables = self.causal_mapper(disentangled_latents)
        description = self.map_to_language(causal_variables)
        return (disentangled_latents, description)

    @torch.no_grad()
    def step(self, state: GWState, action: str) -> Tuple[GWState, dict]:
        """
        Update the state based on the action.
        """
        if state.latents is None:
            if state.image.min().item() >= -0.5:
                image = (state.image * 2.0) - 1.0
            else:
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
        new_state = GWState(step_idx=state.step_idx + 1, image=None, description=new_description, latents=new_latents)
        return new_state, {'goal_reached' : utils.goal_check(utils.extract_goals(self.example), new_description, ignore_obstacles=True)}

    def is_terminal(self, state: Tuple[torch.Tensor, str]) -> bool:
        if utils.goal_check(utils.extract_goals(self.example), state.description, ignore_obstacles=True)[0]:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False

    def map_to_language(self, causals: torch.Tensor) -> str:
        """
        Map the causal variables to a natural language description
        """
        return utils.describe_latent(causals, self.keys)

    def init_state(self) -> GWState:
        """Initialize the world model.

        :return: the initial state
        """
        return GWState(step_idx=0, image=self.example['images'][0], description=utils.extract_init_state(self.example))

class LMWorldModel(WorldModel):
    def __init__(self, lm_model: LanguageModel, prompt: dict, device, max_steps=6):
        super().__init__()
        self.lm_model = lm_model
        self.prompt = prompt
        self.device = device
        self.max_steps = max_steps
 
    def init_state(self) -> GWState:
        return GWState(
            step_idx=0,
            image=self.example['images'][0],
            description=utils.extract_init_state(self.example)
        )

    @torch.no_grad()
    def step(self, state: GWState, action: str) -> Tuple[GWState, dict]:
        """
        Update the state based on the action.
        """
        state = copy.deepcopy(state)
        description = state.description
        step_idx = state.step_idx

        if ("move" in action and "car" in action) or ("move" in action and "vehicle" in action):
            key = "world_update_move_car"
        elif "move" in action and "obstacle" in action:
            key = "world_update_move_obstacle"
        elif "change" in action:
            key = "world_update_change_trafficlight"
        elif "no action" in action:
            key = "world_update_no_action"
        world_update_prompt = self.prompt[key].replace("<state>", description).replace("<action>", action)
        new_description = self.lm_model.generate(
            [world_update_prompt],
            eos_token_id=self.lm_model.tokenizer.eos_token_id,
            hide_input=True,
            temperature=0.0, # greedy decoding
        ).text[0].strip()
        new_description = new_description.replace(world_update_prompt, '').split('[SCENARIO 4]')[0].lstrip(' ').rstrip('\n')
        new_state = GWState(step_idx=step_idx+1, image=None, description=new_description)
        if self.example:
            return new_state, {'goal_reached' : utils.goal_check(utils.extract_goals(self.example), new_description, ignore_obstacles=True)}
        else:
            return new_state, {'goal_reached': False}

    def is_terminal(self, state: Tuple[torch.Tensor, str]) -> bool:
        if utils.goal_check(utils.extract_goals(self.example), state.description, ignore_obstacles=True)[0]:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False
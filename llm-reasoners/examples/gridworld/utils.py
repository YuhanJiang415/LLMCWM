import torch
from transformers import AutoTokenizer
# append the parent dir to path
import sys
sys.path.append('../')
sys.path.append('../../../')
from models.shared import CausalEncoder
from models.biscuit_nf import BISCUITNF
from torch import nn
from causal_mappers import CausalMapper, CausalMappers, MLP

def load_models(crl_model_path, autoencoder_path, causal_mapper_path, tokenizer_path, single_causal_mapper=False, nl_model_path=None, device='cuda'):
    causal_mapper_params = torch.load(causal_mapper_path, map_location=device)
    target_assignment = causal_mapper_params['target_assignment']
    encs_mean = causal_mapper_params['all_encs_mean']
    encs_std = causal_mapper_params['all_encs_std']
    causal_var_info = causal_mapper_params['causal_var_info']
    if single_causal_mapper:
        causal_encoder_sd = causal_mapper_params['model_state_dict']
        causal_encoder = CausalEncoder(c_hid=256, lr=4e-3, causal_var_info=causal_var_info, single_linear=True, c_in=80)
        causal_encoder.load_state_dict(causal_encoder_sd)
        causal_mapper = CausalMapper(causal_encoder, encs_mean, encs_std, target_assignment)
    else:
        causal_encoders = causal_mapper_params['causal_encoders']
        causal_mapper = CausalMappers(causal_encoders, target_assignment, encs_mean, encs_std, device=device)
    crl_model = BISCUITNF.load_from_checkpoint(crl_model_path, autoencoder_path=autoencoder_path, map_location=device)
    if nl_model_path:
        nl_model = torch.load(nl_model_path)
    else:
        nl_model = None
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return crl_model, causal_mapper, nl_model, tokenizer
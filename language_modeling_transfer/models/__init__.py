# Copyright 2023 Matteo Pagliardini, Amirkeivan Mohtashami, Francois Fleuret, Martin Jaggi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#from . import base
from . import denseformer
#from . import connect_to_last
#from . import base_w_gains
#from . import acn
from . import hybrid
#from . import grn_v2
#from . import fixed_hybrid
#from . import input_fixed_hybrid
#from . import mixln_hybrid
#from . import mixln_fixed_hybrid
#from . import learnable_depthLN_hybrid
#from . import mixln_learnable_depthLN_acn
#from . import mixln_learnable_depthLN_hybrid
from . import res_scaled_x
from . import res_scaled_f
from . import denseformer_orig
from . import long_short
from . import gated_acn
from . import resW_scaled_x
from . import hyperconnections
'''
MODELS = {
    "denseformer": denseformer.DenseFormer,
    "residual": base.GPTBase,
    #"connecttolast": connect_to_last.GPTBase,
    #"basewgains": base_w_gains.GPTBase,
    "acn": acn.ACN,
    "hybrid": hybrid.Hybrid,
    "chybrid": hybrid.Hybrid,
    "grn_v2": grn_v2.GRN_v2,
    "fixed_hybrid": fixed_hybrid.Hybrid,
    "input_fixed_hybrid": input_fixed_hybrid.Hybrid,
    "mixln_hybrid": mixln_hybrid.Hybrid,
    "mixln_fixed_hybrid": mixln_fixed_hybrid.Hybrid,
    "learnable_depthLN_hybrid": learnable_depthLN_hybrid.Hybrid,
    "mixln_learnable_depthLN_hybrid": mixln_learnable_depthLN_hybrid.Hybrid,
    "mixln_learnable_depthLN_acn": mixln_learnable_depthLN_acn.ACN,
    "res_scaled_x": res_scaled_x.GPTBase,
    "res_scaled_f": res_scaled_f.GPTBase,
    "denseformer_orig": denseformer_orig.DenseFormer,
    "long_short": long_short.Long_Short
}
'''
MODELS = {
    "denseformer": denseformer.DenseFormer,
    "hybrid": hybrid.Hybrid,
    "res_scaled_x": res_scaled_x.GPTBase,
    "res_scaled_f": res_scaled_f.GPTBase,
    "denseformer_orig": denseformer_orig.DenseFormer,
    "long_short": long_short.Long_Short,
    "gated_acn": gated_acn.gACN,
    "resW_scaled_x": resW_scaled_x.GPTBase,
    "hc": hyperconnections.HC
}


def make_model_from_args(args):
    return MODELS[args.model](args)


def registered_models():
    return MODELS.keys()

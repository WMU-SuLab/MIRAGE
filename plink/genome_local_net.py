import torch
from torch import nn
from copy import copy
from dataclasses import dataclass
from functools import partial
from plink.LCL_model import LCL,LCLResidualBlock
@dataclass
class LCParameterSpec:
    in_features: int
    kernel_width: int
    channel_exp_base: int
    dropout_p: float
    stochastic_depth_p: float
    cutoff: int
    attention_inclusion_cutoff = None
    direction:str = "down"
class DeepcvGRS(nn.Module):
    def __init__(
        self,snp_number=0,patch_size =None,layers =None,kernel_width =16,first_kernel_expansion = -1,channel_exp_base = 2,
            first_channel_expansion = 1, num_lcl_chunks= None,rb_do =0.2,stochastic_depth_p =0.2,
            cutoff = 1024,direction="down",data_dimensions=None,dynamic_cutoff= None):
        super().__init__(),
        self.snp_number = snp_number
        self.patch_size=patch_size
        self.layers=layers
        self.kernel_width=kernel_width
        self.first_kernel_expansion=first_kernel_expansion
        self.channel_exp_base=channel_exp_base
        self.first_channel_expansion=first_channel_expansion
        self.num_lcl_chunks=num_lcl_chunks
        self.rb_do=rb_do
        #self.l1=l1
        self.stochastic_depth_p=stochastic_depth_p
        self.cutoff = cutoff
        self.direction = direction
        self.data_dimensions = data_dimensions
        self.dynamic_cutoff = dynamic_cutoff

        kernel_width = parse_kernel_width(
            kernel_width=self.kernel_width,
            patch_size=self.patch_size,
        )
        print(f'kerel_width',kernel_width)
        fc_0_kernel_size = calc_value_after_expansion(
            base=kernel_width,
            expansion = self.first_kernel_expansion,
        )
        print(f'fc_0_kernel_size',fc_0_kernel_size)
        fc_0_channel_exponent = calc_value_after_expansion(
            base = self.channel_exp_base,
            expansion=self.first_channel_expansion,
        )
        self.fc_0 = LCL(
            in_features=self.snp_number,
            out_feature_sets=2**fc_0_channel_exponent,
            kernel_size=fc_0_kernel_size,
            bias=True,
        )
        cutoff = self.dynamic_cutoff or self.cutoff
        lcl_parameter_spec = LCParameterSpec(
            in_features=self.fc_0.out_features,
            kernel_width=self.kernel_width,
            channel_exp_base=self.channel_exp_base,
            dropout_p=self.rb_do,
            cutoff=cutoff,
            stochastic_depth_p=self.stochastic_depth_p,
            direction=self.direction,
        )
        self.lcl_blocks,self.out_feature = _get_lcl_blocks(
            lcl_spec=lcl_parameter_spec,
            block_layer_spec=self.layers,
        )
        out_feature=self.out_feature
        self.net = nn.Sequential(nn.LayerNorm(out_feature),nn.Linear(out_feature, 1))

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.data_dimensions.num_elements()

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0.weight

    @property
    def num_out_features(self) -> int:
        return self.lcl_blocks[-1].out_features

    def _init_weights(self):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.fc_0(input)
        out = self.lcl_blocks(out)
        out = self.net(out)
        return out
def parse_kernel_width(
    kernel_width,
    patch_size,
) -> int:
    if kernel_width == "patch":
        if patch_size is None:
            raise ValueError(
                "kernel_width set to 'patch', but no patch_size was specified."
            )
        kernel_width = patch_size[0] * patch_size[1] * patch_size[2]
    return kernel_width

def calc_value_after_expansion(base: int, expansion: int, min_value: int = 0) -> int:
    if expansion > 0:
        return base * expansion
    elif expansion < 0:
        abs_expansion = abs(expansion)
        return max(min_value, base // abs_expansion)
    return base

def _get_lcl_blocks(
    lcl_spec,
    block_layer_spec,
) -> nn.Sequential:
    factory = _get_lcl_block_factory(block_layer_spec=block_layer_spec)

    blocks = factory(lcl_spec)

    return blocks

def _get_lcl_block_factory(
    block_layer_spec,
):
    if not block_layer_spec:
        return generate_lcl_residual_blocks_auto

    auto_factory = partial(
        _generate_lcl_blocks_from_spec, block_layer_spec=block_layer_spec
    )

    return auto_factory

def _generate_lcl_blocks_from_spec(
    lcl_parameter_spec: LCParameterSpec,
    block_layer_spec,
) -> nn.Sequential:
    s = lcl_parameter_spec
    block_layer_spec_copy = copy(block_layer_spec)

    first_block = LCLResidualBlock(
        in_features=s.in_features,
        kernel_size=s.kernel_width,
        out_feature_sets=2**s.channel_exp_base,
        dropout_p=s.dropout_p,
        full_preactivation=True,
    )

    block_modules = [first_block]
    block_layer_spec_copy[0] -= 1

    for cur_layer_index, block_dim in enumerate(block_layer_spec_copy):
        for block in range(block_dim):
            cur_out_feature_sets = 2 ** (s.channel_exp_base + cur_layer_index)
            cur_kernel_width = s.kernel_width

            cur_out_feature_sets, cur_kernel_width = _adjust_auto_params(
                cur_out_feature_sets=cur_out_feature_sets,
                cur_kernel_width=cur_kernel_width,
                direction=s.direction,
            )

            cur_size = block_modules[-1].out_features

            cur_block = LCLResidualBlock(
                in_features=cur_size,
                kernel_size=cur_kernel_width,
                out_feature_sets=cur_out_feature_sets,
                dropout_p=s.dropout_p,
                stochastic_depth_p=s.stochastic_depth_p,
            )

            block_modules.append(cur_block)

    return nn.Sequential(*block_modules)
def generate_lcl_residual_blocks_auto(lcl_parameter_spec: LCParameterSpec):
    """
    TODO:   Create some over-engineered abstraction for this and
            ``_generate_lcl_blocks_from_spec`` if feeling bored.
    """

    s = lcl_parameter_spec

    first_block = LCLResidualBlock(
        in_features=s.in_features,
        kernel_size=s.kernel_width*2,
        out_feature_sets=2**s.channel_exp_base,
        dropout_p=s.dropout_p, 
        stochastic_depth_p=s.stochastic_depth_p,                        
        full_preactivation=True,
    )
    block_modules = [first_block]

    while True:
        cur_no_blocks = len(block_modules)
        cur_index = cur_no_blocks // 2

        cur_out_feature_sets = 2 ** (s.channel_exp_base + cur_index)
        cur_kernel_width = s.kernel_width
        cur_out_feature_sets, cur_kernel_width = _adjust_auto_params(
            cur_out_feature_sets=cur_out_feature_sets,
            cur_kernel_width=cur_kernel_width,
            direction=s.direction,
        )

        cur_size = block_modules[-1].out_features

        if _should_break_auto(
            cur_size=cur_size,
            cutoff=s.cutoff,
            direction=s.direction,
        ):
        
            break

        cur_block = LCLResidualBlock(
            in_features=cur_size,
            kernel_size=cur_kernel_width*2,
            out_feature_sets=2**s.channel_exp_base,   #cur_out_feature_sets,
            dropout_p=s.dropout_p,
            stochastic_depth_p=s.stochastic_depth_p,
        )

        block_modules.append(cur_block)
    return nn.Sequential(*block_modules),block_modules[-1].out_features

def _adjust_auto_params(
    cur_out_feature_sets: int, cur_kernel_width: int, direction,
) -> tuple[int, int]:

    if direction == "down":
        while cur_out_feature_sets >= cur_kernel_width:
            cur_kernel_width *= 2
    elif direction == "up":
        while cur_out_feature_sets <= cur_kernel_width:
            cur_out_feature_sets *= 2
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return cur_out_feature_sets, cur_kernel_width

def _should_break_auto(
    cur_size: int, cutoff: int, direction,
) -> bool:
    if direction == "down":
        print(f'============================direction:{direction}=================================')
        print (f'==========================cur_size:{cur_size}==================================')
        return cur_size <= cutoff
    elif direction == "up":
        print (f'==========================cur_size:{cur_size}==================================')
        return cur_size >= cutoff
    else:
        raise ValueError(f"Unknown direction: {direction}")

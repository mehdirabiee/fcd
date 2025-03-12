import os
import torch
from thop import profile, clever_format
from networks2 import MS_DSA_NET

from monai.networks.layers.factories import Norm,Act

def get_model(params):

    if params['patch_size'][0] == params['patch_size'][1] and params['patch_size'][0] == params['patch_size'][2]:
        str_ps = 'ps{}'.format(params['patch_size'][0]) 
    else:
        str_ps = 'ps{}x{}x{}'.format(params['patch_size'][0], params['patch_size'][1], params['patch_size'][2])


    if 'MS_DSA_NET' == params['model_type']: #dual-attention

        model = MS_DSA_NET(
                spatial_dims=3,
                in_channels=params['chans_in'],
                out_channels=params['chans_out'],
                img_size=params['patch_size'],
                feature_size=params['feature_size'],
                pos_embed=True,
                project_size= params['project_size'],
                sa_type=params['sa_type'],
                norm_name= 'instance', #'batch', #
                act_name=  ("leakyrelu", {"inplace": True, "negative_slope": 0.01}), #'relu', 
                res_block=True,
                bias= False, #False,
                dropout_rate=0.1,
            ) 
        
        model_desc_str = '{}_{}_fs{}'.format(model.name, str_ps, params['feature_size'])
        #save_dir = os.path.join(params['base_dir'] , sub_dir)

    params['model_desc_str'] = model_desc_str
    return model, params
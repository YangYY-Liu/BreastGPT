from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from swift.model import (Model, ModelGroup, ModelMeta, MultiModelKeys, register_model, register_model_arch)
from model.architectures.breastgpt import BreastGPT, BreastGPTConfig
from model.architectures.breastgpt2 import BreastGPT2, BreastGPT2Config
from swift.model.models.qwen import Qwen2VLLoader, _compat_qwen3_vl_mixed_data, compat_qwen_vl_utils, require_version


AutoConfig.register('breastgpt', BreastGPTConfig)
AutoConfig.register('breastgpt2', BreastGPT2Config)
AutoModel.register(BreastGPTConfig, BreastGPT)
AutoModelForCausalLM.register(BreastGPTConfig, BreastGPT)

class BreastGPTLoader(Qwen2VLLoader):
    def _check_qwen_vl_utils(self):
        require_version('qwen_vl_utils>=0.0.14')
        compat_qwen_vl_utils(image_patch_size=16)

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        self.auto_model_cls = self.auto_model_cls or BreastGPT
        model = super().get_model(model_dir, config, processor, model_kwargs)
        _compat_qwen3_vl_mixed_data(model.model, processor)
        return model

class BreastGPT2Loader(Qwen2VLLoader):
    def get_model(self, model_dir: str, *args, **kwargs):
        self.auto_model_cls = self.auto_model_cls or BreastGPT2
        return super().get_model(model_dir, *args, **kwargs)
    
register_model_arch(
    MultiModelKeys(
        'breastgpt',
        language_model=['model.language_model', 'lm_head'],
        aligner=['visual_selector', 'deepstack_selector', 'model.visual.merger', 'model.visual.deepstack_merger_list'],
        vision_tower=['slide', 'model.visual']
    ))

register_model_arch(
    MultiModelKeys(
        'breastgpt2',
        language_model=['model.language_model', 'lm_head'],
        aligner=['visual_selector', 'deepstack_selector', 'model.visual.merger'],
        vision_tower=['slide', 'model.visual']
    ))


register_model(
    ModelMeta(
        model_type='breastgpt',
        model_groups=[
            ModelGroup([Model(model_path='/nas/yangye.ly/breastGPT/model/architectures/saved_models/breastgpt_8B'),
                        Model(model_path='/nas/yangye.ly/breastGPT/model/architectures/saved_models/breastgpt_4B')]),
        ],
        template='breastgpt',
        loader=BreastGPTLoader,
        model_arch='breastgpt',
        is_multimodal=True,
        architectures=['BreastGPT'],
    ))

register_model(
    ModelMeta(
        model_type='breastgpt2',
        model_groups=[
            ModelGroup([Model(model_path='/nas/yangye.ly/breastGPT/model/architectures/saved_models/breastgpt2_8B')]),
        ],
        template='breastgpt2',
        loader=BreastGPT2Loader,
        model_arch='breastgpt2',
        is_multimodal=True,
        architectures=['BreastGPT2'],
    ))


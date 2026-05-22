import os
import torch
import numpy as np
import SimpleITK as sitk
import h5py
import torch.nn.functional as F
from PIL import Image
from packaging import version
from functools import partial
from typing import Any, Dict, List, Literal, Optional
from swift.template.templates.qwen import Qwen2VLTemplate, QwenTemplateMeta
from swift.template import StdTemplateInputs, Template, register_template
from swift.template.utils import Context, findall
from swift.template.vision_utils import load_audio, rescale_image, load_batch
from swift.utils import get_env_args, get_logger, get_packed_seq_params

logger = get_logger()
from swift.utils import is_deepspeed_enabled

############################################################## READ NIFTI ########################################################

def get_env_int(key, default=None):
    value = os.environ.get(key)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def preprocess_video(
    data, 
    do = True,
    target_size=None,  # (H, W)
    max_size=(384, 384), # (Max_H, Max_W)
    target_frames=None,
    max_frames=None
):
    """
    重构后的视频预处理函数
    :param data: numpy array, 形状为 (D, H, W)
    """
    data = (data - data.min()) / (data.max() - data.min() + 1e-6) * 255.0
    # 1. 转换为 Tensor 并调整维度为 (B, C, D, H, W) 
    # 这样更符合 3D 视觉任务的标准，且方便统一插值
    video = torch.from_numpy(data).float()
    if video.ndim == 3:
        video = video.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    
    B, C, D0, H0, W0 = video.shape

    # 2. 空间维度缩放 (H, W)
    if target_size:
        new_h, new_w = target_size
    else:
        # 计算等比例缩放因子
        scale = min(max_size[0] / H0, max_size[1] / W0, 1.0)
        new_h, new_w = int(H0 * scale), int(W0 * scale)

    # 3. 时间维度缩放 (D)
    new_d = target_frames if target_frames else (min(D0, max_frames) if max_frames else D0)
    new_d = max(new_d // 3 * 3, 3)
    # 4. 一次性完成 3D 插值 (trilinear)
    # 比起 split 后再 resize，trilinear 能更好地处理帧间平滑
    if do:
        video = F.interpolate(
            video, 
            size=(new_d, new_h, new_w), 
            mode="trilinear", 
            align_corners=False
        ).squeeze()
    video = video.unsqueeze(1).expand(-1, 3, -1, -1)
    # D, H, W = video.shape
    # video = video.reshape(D//3, 3, H, W)
    return video

def fetch_data(ele: Dict[str, Any], image_patch_size: int = 14, return_video_sample_fps: bool = False, return_video_metadata: bool = False, do=True):
    from qwen_vl_utils.vision_process import (fetch_video, logger, smart_resize, MODEL_SEQ_LEN, transforms, InterpolationMode)
    SPATIAL_MERGE_SIZE = get_env_int('SPATIAL_MERGE_SIZE', 2)
    VIDEO_MIN_TOKEN_NUM = get_env_int('VIDEO_MIN_TOKEN_NUM', 128)
    VIDEO_MAX_TOKEN_NUM = get_env_int('VIDEO_MAX_TOKEN_NUM', 768)
    FRAME_FACTOR = get_env_int('FRAME_FACTOR', 2)
    
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    if isinstance(ele["video"], str):
        ele["video"] = ele["video"].strip()
        if ele["video"].lower().endswith(('.nii', '.nii.gz', '.npy')):
            npy_path = ele["video"].replace('.nii.gz', '.npy').replace('.nii', '.npy')
            if os.path.exists(npy_path):
                data = np.load(npy_path) # D, H, W
            else:
                img = sitk.ReadImage(ele["video"])
                data = sitk.GetArrayFromImage(img) # (D, H, W)
            WINDOW_WIDTH, WINDOW_LEVEL = os.environ.get('WINDOW_WIDTH', None), os.environ.get('WINDOW_LEVEL', None)
            if WINDOW_WIDTH and WINDOW_LEVEL:
                min_val = int(WINDOW_LEVEL) - (int(WINDOW_WIDTH) // 2)
                max_val = int(WINDOW_LEVEL) + (int(WINDOW_WIDTH) // 2)
                data = data.clip(min_val, max_val)
            # 目标尺寸 (H, W)
            target_h = get_env_int('H')
            target_w = get_env_int('W')
            target_size = (target_h, target_w) if (target_h and target_w) else None

            # 最大限制
            max_size = (
                get_env_int('MAX_H', 384), 
                get_env_int('MAX_W', 384)
            )

            # 帧数限制 (D)
            target_frames = get_env_int('D')
            max_frames = get_env_int('MAX_D')
            video = preprocess_video(
                data,
                do,
                target_size=target_size,
                max_size=max_size,
                target_frames=target_frames,
                max_frames=max_frames
            )
            
            sample_fps = ele.get("sample_fps", 1.0)
            video_metadata = {"fps": sample_fps, "frames_indices": list(range(len(video))), "total_num_frames": len(video)}
        else:
            # raise TypeError(ele["video"][-10:])
            return fetch_video(ele, image_patch_size=image_patch_size, return_video_sample_fps=return_video_sample_fps, return_video_metadata=return_video_metadata)
    else:
        raise
    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
    max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video

def load_trident_h5(h5_path, device="cpu"):
    """
    Read Trident WSI patch embeddings.

    Returns:
        coords: (N,2) numpy
        feats : (N,D) torch tensor
    """

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

        if "coords" not in keys or "features" not in keys:
            raise ValueError(f"{h5_path} missing coords/features, keys={keys}")

        coords = np.array(f["coords"])
        feats = np.array(f["features"])

    feats = torch.from_numpy(feats).float().to(device)

    return coords, feats

class BreastGPTTemplate(Qwen2VLTemplate):  # 或者根据你实际继承的基类
    image_token_id = 151655
    video_token_id = 151656
    histo_token_id = 151669
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>', '<|histo_pad|>']
    use_model = True
    support_padding_free = True
    version = 'v3'
    default_system: Optional[str] = 'You are BreastGPT, created by Yang Liu. You are a helpful assistant.'

    def _pre_tokenize(self, context_list, loss_scale_list, inputs):
        context_list, loss_scale_list = self._pre_tokenize_images(context_list, loss_scale_list, inputs)
        # 官方只判断 inputs.images，3D 走 videos 路径时 normalize_bbox 不会被调用
        if (inputs.images or inputs.videos) and inputs.objects:
            self.normalize_bbox(inputs)
        # replace tag/object/box
        res, res_loss_scale = [], []
        for k in ['video', 'audio', 'object', 'box']:
            setattr(inputs, f'{k}_idx', 0)
        for context, loss_scale in zip(context_list, loss_scale_list):
            for k in ['video', 'audio']:
                if context == f'<{k}>' and inputs.is_multimodal and getattr(inputs, f'{k}_idx') < len(getattr(inputs, f'{k}s')):
                    c_list = self.replace_tag(k, getattr(inputs, f'{k}_idx'), inputs)
                    setattr(inputs, f'{k}_idx', getattr(inputs, f'{k}_idx') + 1)
                    loss_scale = 0.
                    break
            else:
                ref = inputs.objects.get('ref') or []
                bbox = inputs.objects.get('bbox') or []
                if context == '<ref-object>' and inputs.ref_idx < len(ref):
                    idx = inputs.ref_idx
                    c_list = self.replace_ref(ref[idx], idx, inputs)
                    inputs.ref_idx += 1
                elif context == '<bbox>' and inputs.bbox_idx < len(bbox):
                    idx = inputs.bbox_idx
                    c_list = self.replace_bbox(bbox[idx], idx, inputs)
                    inputs.bbox_idx += 1
                elif context == '<cot-process>' and self.task_type == 'prm':
                    c_list = self.replace_cot_process(inputs)
                else:
                    c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)
        return res, res_loss_scale

    @staticmethod
    def _get_height_width(inputs: StdTemplateInputs) -> None:
        objects = inputs.objects
        width, height, depth = [], [], []

        for image in inputs.images:
            if isinstance(image, Image.Image):
                width.append(image.width)
                height.append(image.height)
                depth.append(1)
            else:
                # .h5 字符串
                width.append(1)
                height.append(1)
                depth.append(1)

        for video_path in inputs.videos:
            npy_path = video_path.replace('.nii.gz', '.npy').replace('.nii', '.npy')
            if os.path.exists(npy_path):
                arr = np.load(npy_path)
                D, H, W = arr.shape[0], arr.shape[1], arr.shape[2]
            elif video_path.lower().endswith(('.nii', '.nii.gz')):
                reader = sitk.ImageFileReader()
                reader.SetFileName(video_path)
                reader.ReadImageInformation()
                W, H, D = reader.GetSize()  # SimpleITK GetSize 返回 (x,y,z) = (W,H,D)
            else:
                W = H = D = 1
            width.append(W)
            height.append(H)
            depth.append(D)

        objects['width'] = width
        objects['height'] = height
        if any(d != 1 for d in depth):
            objects['depth'] = depth

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        self._preprocess_function_call(inputs)
        if self.model_meta.is_multimodal:
            self._replace_image_tags(inputs)
            self._replace_start_image_tags(inputs)

        images = inputs.images
        load_images = self.load_images or self.mode in {'vllm', 'lmdeploy'}
        load_images_origin = load_images
        if self.max_pixels is not None or inputs.objects:
            load_images = True
        if images:
            for i, image in enumerate(images):
                img = images[i]
                img_path = img['path'] if isinstance(img, dict) else img
                if img_path.endswith('.h5'):
                    images[i] = img_path
                else:
                    images[i] = self._load_image(img, load_images)
        if inputs.objects:
            self._get_height_width(inputs)
        if self.max_pixels is not None:
            # Scale the image proportionally without affecting the scaled objects.
            
            images = [img if isinstance(img, str) else rescale_image(img, self.max_pixels) for img in images]
        if images and not load_images_origin:  # fix pt & qwen-vl
            for i, image in enumerate(images):
                if isinstance(image, Image.Image):
                    images[i] = self._save_pil_image(image)
        inputs.images = images

        if self.mode == 'vllm' and inputs.audios:
            sampling_rate = get_env_args('sampling_rate', int, None)
            inputs.audios = load_batch(
                inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate, return_sr=True))
        if inputs.is_multimodal:
            self._add_default_tags(inputs)

    def normalize_bbox(self, inputs: StdTemplateInputs) -> None:
        objects = inputs.objects
        bbox_list = objects['bbox']
        width_list = objects['width']
        height_list = objects['height']
        depth_list = objects.get('depth', [])
        bbox_type = objects.pop('bbox_type', None) or 'real'
        image_id_list = objects.pop('image_id', None) or []
        image_id_list += [0] * (len(bbox_list) - len(image_id_list))

        for bbox, image_id in zip(bbox_list, image_id_list):
            is_3d = len(bbox) % 3 == 0 and len(bbox) >= 6 and depth_list
            # 源坐标系（除数）：bbox 本身的坐标范围
            if bbox_type == 'norm1':
                src_width = src_height = src_depth = 1
            else:  # 'real'
                src_width  = width_list[image_id]  if width_list  else 1
                src_height = height_list[image_id] if height_list else 1
                src_depth  = depth_list[image_id]  if depth_list  else 1
            
            # 目标坐标系（乘数）：归一化后的坐标范围
            if self.norm_bbox == 'norm1000':
                norm_width = norm_height = norm_depth = 1000
            else:  # 'none'：输出真实像素/体素坐标
                norm_width  = width_list[image_id]  if width_list  else 1
                norm_height = height_list[image_id] if height_list else 1
                norm_depth  = depth_list[image_id]  if depth_list  else 1

            if is_3d:
                for i in range(len(bbox) // 3):
                    x, y, z = bbox[3*i], bbox[3*i+1], bbox[3*i+2]
                    bbox[3*i]   = int(round(x / src_width  * norm_width))
                    bbox[3*i+1] = int(round(y / src_height * norm_height))
                    bbox[3*i+2] = int(round(z / src_depth  * norm_depth))
            else:
                for i, (x, y) in enumerate(zip(bbox[::2], bbox[1::2])):
                    bbox[2*i]   = int(round(x / src_width  * norm_width))
                    bbox[2*i+1] = int(round(y / src_height * norm_height))

    @staticmethod
    def _get_bbox_str(bbox: List[int]) -> str:
        if len(bbox) % 3 == 0:
            # 3D: x, y, z
            point = []
            for x, y, z in zip(bbox[::3], bbox[1::3], bbox[2::3]):
                point.append(f'({x},{y},{z})')
        else:
            # 2D: x, y
            point = []
            for x, y in zip(bbox[::2], bbox[1::2]):
                point.append(f'({x},{y})')
        return ','.join(point)
    
    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        """Replace bbox pointing to the objects to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            bbox: [x, y] or [x1, y1, x2, y2]
            index: The index in the `objects` key
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image
        assert media_type in {'image', 'video'}
        kwargs = {'image_patch_size': self.processor.image_processor.patch_size} if self.version == 'v3' else {}
        if media_type == 'image':
            img_path = inputs.images[index]
            
            # ================= 新增：拦截病理特征文件 =================
            if isinstance(img_path, str) and img_path.lower().endswith('.h5'):
                feat = load_trident_h5(img_path)[1] # L,D
                # 直接将 Tensor 存在 images 列表中，供后面的 _encode 识别
                inputs.images[index] = feat 
                if self.mode == 'lmdeploy':
                    # return [[-100] * min(HISTO_TOKEN_NUM, feat.shape[0])]
                    return ['<|vision_start|>', [-100], '<|vision_end|>']
                else:
                    # return ['<|histo_pad|>' * min(HISTO_TOKEN_NUM, feat.shape[0])]
                    return ['<|vision_start|><|histo_pad|><|vision_end|>']
            # ==========================================================
            else:
                # 走原来的普通图像读取逻辑
                inputs.images[index] = fetch_image({'image': img_path}, **kwargs)
                if self.mode == 'lmdeploy':
                    return ['<|vision_start|>', [-100], '<|vision_end|>']
                else:
                    return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            if self.version == 'v3':
                kwargs['return_video_metadata'] = True
            video = inputs.videos[index]
            video_inputs = {'video': video}
            if isinstance(video, list):  # image list
                from qwen_vl_utils import vision_process
                video_inputs['sample_fps'] = vision_process.FPS
            video, video_kwargs = fetch_data(video_inputs, return_video_sample_fps=True, **kwargs)
            if self.version == 'v2_5':
                inputs.mm_processor_kwargs.setdefault('fps', []).append(video_kwargs)
                tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
            elif self.version == 'v3':
                if self.mode == 'vllm':
                    tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
                else:
                    video, video_metadata = video
                    inputs.mm_processor_kwargs.setdefault('video_metadata', []).append(video_metadata)
                    tokens = ['<|video_pad|>']
                inputs.mm_processor_kwargs['do_sample_frames'] = False
            # if isinstance(video, torch.Tensor):
            #     video = video.to(torch.uint8)
            inputs.videos[index] = video
            return tokens
        
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        DO_SELECT_TOKENS = getattr(self.config, 'do_select', None)
        if DO_SELECT_TOKENS:
            if "SELECT_IMAGE_NUM" in os.environ:
                self.config.image_token_num = int(os.environ["SELECT_IMAGE_NUM"])
            if "SELECT_HISTO_NUM" in os.environ:
                self.config.histo_token_num = int(os.environ["SELECT_HISTO_NUM"])
            if "SELECT_VIDEO_NUM" in os.environ:
                self.config.video_token_num = int(os.environ["SELECT_VIDEO_NUM"])
            IMAGE_TOKEN_NUM, VIDEO_TOKEN_NUM, HISTO_TOKEN_NUM = self.config.image_token_num, self.config.video_token_num, self.config.histo_token_num

        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            merge_size = processor.image_processor.merge_size
            merge_length = merge_size ** 2
            if mm_data:
                if media_type == 'images':
                    media_inputs = {}
                    image_grid_thw = histo_grid_thw = None
                    
                    # 1. 独立处理常规图片，获取真实的 grid_thw
                    normal_images = [img for img in mm_data if not isinstance(img, torch.Tensor)]
                    feature_tensors = [img for img in mm_data if isinstance(img, torch.Tensor)]
                    
                    if normal_images:                        
                        normal_inputs = processor.image_processor(images=normal_images, return_tensors='pt', do_resize=False)
                        media_inputs.update(normal_inputs)
                        image_grid_thw = normal_inputs['image_grid_thw']
                        # 4. 扩展图片 Token (只扩展常规图片！)
                        # 注意: replace_tag 中 .h5 已经是 N 个 <|histo_pad|>，无需再次扩展
                        media_token = self.image_token_id
                        idx_list = findall(input_ids, media_token)
                        
                        def _get_new_tokens(i):
                            token_len = image_grid_thw[i].prod().item() // merge_length
                            if DO_SELECT_TOKENS:
                                token_len = min(token_len, IMAGE_TOKEN_NUM)
                            return [media_token] * token_len

                        input_ids, labels, loss_scale = self._extend_tokens(
                            input_ids, labels, loss_scale, idx_list, _get_new_tokens
                        )
                        #  伪造 inputs_ID 实现token selector， 废弃
                        # rope_grid_thw = []
                        # for i in range(len(image_grid_thw)):
                        #     token_len = image_grid_thw[i].prod().item() // merge_length
                        #     k_eff = min(IMAGE_TOKEN_NUM, token_len)
                        #     rope_grid_thw.append([1, merge_size, k_eff * merge_size])
                        # media_inputs['rope_image_grid_thw'] = torch.tensor(rope_grid_thw, dtype=torch.long)
                    
                    if feature_tensors:
                        media_inputs['input_features'] = feature_tensors
                        histo_grid_thw = []
                        for img in feature_tensors:
                            token_len = img.shape[0]
                            histo_grid_thw.append([1, 1, token_len])
                        histo_grid_thw = torch.tensor(histo_grid_thw, dtype=torch.long)
                        media_inputs['histo_grid_thw'] = histo_grid_thw
                        media_token = self.histo_token_id
                        idx_list = findall(input_ids, media_token)
                        def _get_new_tokens(i):
                            token_len = histo_grid_thw[i].prod().item()
                            if DO_SELECT_TOKENS:
                                token_len = min(token_len, HISTO_TOKEN_NUM)
                            return [media_token] * token_len
                        input_ids, labels, loss_scale = self._extend_tokens(
                            input_ids, labels, loss_scale, idx_list, _get_new_tokens
                        )
                    encoded.update(media_inputs)
                else:
                    # Video 逻辑保持不变...
                    if self.version != 'v3':
                        kwargs = {}
                        if hasattr(processor, 'video_processor'):
                            processor_func = processor.video_processor
                        else:
                            processor_func = processor.image_processor
                            kwargs['images'] = None
                        media_inputs = processor_func(videos=mm_data, return_tensors='pt', do_resize=False, **kwargs)
                        media_grid_thw = media_inputs['video_grid_thw']
                        media_token = self.video_token_id
                        if self.version == 'v2_5':
                            fps = inputs.mm_processor_kwargs['fps']
                            media_inputs['second_per_grid_ts'] = [
                                processor.image_processor.temporal_patch_size / tmp for tmp in fps
                            ]
                        fps = inputs.mm_processor_kwargs['fps']
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / tmp for tmp in fps
                        ]
                        def _get_new_tokens(i):
                            token_len = (media_grid_thw[i].prod() // merge_length)
                            return [media_token] * token_len
                    else:
                        split_token = self._tokenize('\n')[0]
                        media_inputs = processor(
                            text=['\n'.join(['<|vision_start|><|video_pad|><|vision_end|>'] * len(mm_data))],
                            videos=mm_data,
                            return_tensors='pt',
                            do_resize=False,
                            **inputs.mm_processor_kwargs)
                        splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
                        media_grid_thw = media_inputs['video_grid_thw']
                        media_inputs.pop('input_ids', None)
                        media_inputs.pop('attention_mask', None)
                        media_token = self.video_token_id
                        # def _get_new_tokens(i):
                        #     return splited_tokens[i]
                        def _get_new_tokens(i):
                            token_len = media_grid_thw[i].prod().item() // merge_length
                            if DO_SELECT_TOKENS:
                                token_len = min(token_len, VIDEO_TOKEN_NUM)
                            return [media_token] * token_len

                    idx_list = findall(input_ids, media_token)
                
                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens)
                    encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded
    
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training or self.version == 'v3':
            return inputs
        else:
            # input_ids = inputs['input_ids']
            # input_features = inputs.get('input_features')
            # base_model = self.get_base_model(model)
            # if hasattr(base_model.model, 'embed_tokens'):
            #     inputs_embeds = base_model.model.embed_tokens(input_ids)
            # else:
            #     inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
            # inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
            # # 含 histo 的混合模态数据场景
            # if input_features is None:
            #     if is_deepspeed_enabled():
            #         # 注意: 由于transformers实现中，经过audio部分模型层的次数与audio数量相关
            #         # 因此zero3在不同进程audios数不同场景下会卡住（需修改transformers代码修复）。此场景请使用zero2。
            #         input_features = input_ids.new_zeros([1, 128, 128], dtype=model.thinker.audio_tower.dtype)
            #         feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
            #         audio_res = model.thinker.get_audio_features(input_features, feature_attention_mask)
            #         # 兼容transformers 5.0
            #         if hasattr(audio_res, 'last_hidden_state'):
            #             audio_embeds = audio_res.last_hidden_state
            #         else:
            #             audio_embeds = audio_res
            #         inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
            # else:
            #     audio_res = model.thinker.get_audio_features(input_features, feature_attention_mask)
            #     # 兼容transformers 5.0
            #     if hasattr(audio_res, 'last_hidden_state'):
            #         audio_embeds = audio_res.last_hidden_state
            #     else:
            #         audio_embeds = audio_res
            #     audio_mask = (input_ids == thinker_config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            #     audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            #     inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
            return super()._post_encode(model, inputs)

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        # 添加 histo_grid
        grid_thw = self.concat_tensor(batch, f'histo_grid_thw', 0)
        if grid_thw is not None:
            res[f'histo_grid_thw'] = grid_thw

        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts

        input_features = []
        for b in batch:
            if 'input_features' in b:
                # b.pop 返回的是当前样本的特征 List，用 extend 铺平
                input_features.extend(b.pop('input_features'))
                
        if input_features:
            # 此时 res['input_features'] 是一个 List，里面装了 N 个形状各异的 Tensor [L, Dim]
            res['input_features'] = input_features
        # =====================================================================
        return res
    
    def _get_position_ids(self, inputs: Dict[str, Any]):
        # fix https://github.com/huggingface/transformers/pull/33487
        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': inputs.get('second_per_grid_ts')}
        base_model = self.get_base_model(self._get_model())
        if hasattr(base_model, 'get_rope_index'):
            get_rope_index = base_model.get_rope_index
        else:
            get_rope_index = base_model.model.get_rope_index
        attention_mask = inputs.get('attention_mask_2d')
        if attention_mask is None:
            attention_mask = inputs.get('attention_mask')
        position_ids, _ = get_rope_index(
            input_ids=inputs['input_ids'],
            image_grid_thw=inputs.get('image_grid_thw'),
            video_grid_thw=inputs.get('video_grid_thw'),
            histo_grid_thw=inputs.get('histo_grid_thw'),
            attention_mask=attention_mask,
            **kwargs)
        return self._concat_text_position_ids(position_ids)

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        Args:
            batch(`List[Dict[str, Any]]`): The input data in batch
            padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
                will be padded to the `longest`
        """
        res = super()._data_collator(batch, padding_to=padding_to)
        if not self.padding_free and self.is_training:
            res['position_ids'] = self._get_position_ids(res)
        if 'position_ids' in res:
            position_ids = res['position_ids']
            res['position_ids'] = position_ids[1:]
            res['text_position_ids'] = text_position_ids = position_ids[0]
            if self.transformers_version >= version.parse('4.53.0.dev') and text_position_ids.shape[0] == 1:
                # https://github.com/huggingface/transformers/pull/40194
                res.update(get_packed_seq_params(text_position_ids))
        return res

register_template(
    QwenTemplateMeta(
        'breastgpt', template_cls=BreastGPTTemplate, default_system=None, thinking_prefix='<think>\n')
)

class BreastGPT2Template(BreastGPTTemplate):
    version = 'v2_5'
    norm_bbox = 'none'

register_template(
    QwenTemplateMeta(
        'breastgpt2', template_cls=BreastGPT2Template)
)


if __name__ == '__main__':
    from swift import get_processor, get_template
    tokenizer = get_processor('Qwen/Qwen3-VL-8B-Instruct')
    template = get_template(tokenizer, template_type='breastgpt')
    template.set_mode('train')
    dummy_img = '/nas/yangye.ly/breastGPT/datasets/RESIZED/Mammography/EMBED/Processed/image/46648133_1972647878893243/15.png'
    dummy_video = '/nas/yangye.ly/breastGPT/datasets/RESIZED/MRI/ZHE2/images_format/MR1588916/T1dyn.nii.gz '
    inputs_norm = {
        "messages": [
            {"role": "system", "content": "As an expert breast pathologist..."},
            {"role": "user",      "content": "<image>\nDescribe <ref-object><bbox>."},
            {"role": "assistant", "content": "There is <ref-object><bbox>."},
        ],
        "images": [dummy_img],
        "objects": {
            "ref":  ["a lesion"],
            "bbox": [[0.114, 0.100, 0.300, 0.213]],
            "bbox_type": "norm1",
            "image_id": [0],
        }
    }

    encoded = template.encode(inputs_norm)
    print(template.safe_decode(encoded['input_ids']))
    print(template.safe_decode([l for l in encoded['labels'] if l != -100]))


    inputs_norm = {
        "messages": [
            {"role": "system", "content": "As an expert breast pathologist..."},
            {"role": "user",      "content": "<video>\nDescribe <ref-object><bbox>."},
            {"role": "assistant", "content": "There is <ref-object><bbox>."},
        ],
        "videos": [dummy_video],
        "objects": {
            "ref":  ["a lesion"],
            "bbox": [[0.114, 0.100, 0.300, 0.114, 0.100, 0.300,]],
            "bbox_type": "norm1",
            "image_id": [0],
        }
    }
    encoded_norm = template.encode(inputs_norm)
    decoded_norm = tokenizer.decode(encoded_norm['input_ids'])
    print("\n=== Test 2: norm1 bbox ===")
    print(decoded_norm)
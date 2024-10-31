import os
import json
import pdb
import math
from itertools import islice
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchmetrics")
warnings.filterwarnings("ignore", message="This property will be removed in 2.0.0. Use `Metric.updated_called` instead.")
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn

import torch.nn.functional as F
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from segmentation_models_pytorch.base import SegmentationHead

import numpy as np
from Long_CLIP.model import longclip
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNBlock, SegmentationBlock, MergeBlock
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder as original_FPNDecoder

from models.segmentation._abstract import SegmentationModel
from ldm.util import instantiate_from_config
from TADP.vpd.models import UNetWrapper, TextAdapter,UNetWrapper2
from torch.optim.lr_scheduler import LambdaLR
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
def apply_cross_attention(self, x, seg_hidden_states):
    if self.use_cross_attention and seg_hidden_states is not None:
        seg_hidden_states = self.mlp(seg_hidden_states)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(2, 0, 1)
        attn_output, _ = self.cross_attention(x_flat, seg_hidden_states.unsqueeze(1), seg_hidden_states.unsqueeze(1))
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        x = x + attn_output
    return x
class FPNDecoder2(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
        embed_dim=256,  # 增加一个embed_dim参数，表示特征图的通道数
        use_cross_attention=True  # 是否启用cross-attention
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.mlp = nn.Sequential(
            nn.Linear(4096, embed_dim),  # 将seg_hidden_states映射到embed_dim大小
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                for n_upsamples in [3, 2, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        # 定义cross-attention模块
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)

    def forward(self, *features,seg_hidden_states):

        c2, c3, c4, c5 = features[-4:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        # 如果启用了cross-attention且seg_hidden_states不为空
        if self.use_cross_attention and seg_hidden_states is not None:
            # 1. 将seg_hidden_states通过MLP变换
            seg_hidden_states = self.mlp(seg_hidden_states)  # 将seg_hidden_states映射到和特征图相同的维度

            # 2. 将特征图x展开为2D（flatten）以进行cross-attention
            B, C, H, W = x.shape  # 假设 x 形状为 [B, C, H, W]  [4, 128, 64, 64]
            x_flat = x.view(B, C, -1).permute(2, 0, 1)  # 转换为 [H*W, B, C]  #4096*4*128

            # 3. 执行cross-attention，使用seg_hidden_states作为查询向量
            attn_output, _ = self.cross_attention(x_flat, seg_hidden_states.unsqueeze(1),
                                                  seg_hidden_states.unsqueeze(1))

            # 4. 将cross-attention输出重新变换回原始特征图的形状
            attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)

            # 5. 将cross-attention结果和原始特征图进行融合（例如直接相加）
            x = x + attn_output

        return x
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchmetrics")
warnings.filterwarnings("ignore", message="This property will be removed in 2.0.0. Use `Metric.updated_called` instead.")
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
def adjust_and_cat_tensors(tensors, target_size):
    adjusted = [F.interpolate(t, size=(target_size, target_size), mode='bilinear', align_corners=False) for t in tensors]
    return torch.cat(adjusted, dim=1)

def group_and_process_tensors(decoder_outs):
    groups = {
        64: [],
        128: [],
        256: [],
        512: []
    }

    for tensor in decoder_outs:
        size = tensor.shape[2]  # 假设空间维度是方形的
        if size in groups:
            groups[size].append(tensor)

    return groups
def log_and_calculate_lr(x, cfg):
    #logger.info(f"x: {x}, max_epochs: {cfg['max_epochs']}, dataset_len: {cfg['dataset_len']}")
    lr_value = max((1 - x / (cfg["dataset_len"] * cfg["max_epochs"])) ** 0.9, 1e-6)
    return lr_value

#### adapted from
# https://github.com/wl-zhao/VPD
class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        # encoder_channels = encoder_channels[::-1]
        self.encoder_channels = encoder_channels[: encoder_depth + 1]

        # do automaticallyy instead
        for i in range(1, len(self.encoder_channels)):
            setattr(self, f"p{i + 1}", FPNBlock(pyramid_channels, self.encoder_channels[i - 1]))
        setattr(self, f"p{len(self.encoder_channels) + 1}",
                nn.Conv2d(self.encoder_channels[-1], pyramid_channels, kernel_size=1))

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                #for n_upsamples in [3, 2, 1, 0]
                for n_upsamples in [0, 1, 2, 3]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):

        # replace throghu loop
        ps = []
        k = -1
        for i in reversed(range(2, len(features) + 2)):
            if i == len(features) + 1:
                p = getattr(self, f"p{i}")(features[k])
            else:
                p = getattr(self, f"p{i}")(p, features[k])
            k -= 1
            ps.append(p)

        # feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, ps)]
        x = self.merge(feature_pyramid)

        # 如果开启了 cross-attention 且 seg_hidden_states 不为空，进行 cross-attention
        if self.use_cross_attention and seg_hidden_states is not None:
            # 将特征展开为二维，以便与 seg_hidden_states 做 attention
            x_flat = x.flatten(2).permute(2, 0, 1)  # 转换为 (seq_len, batch_size, feature_dim)
            seg_hidden_states = seg_hidden_states.unsqueeze(1)  # 增加 batch 维度
            attn_output, _ = self.cross_attention(x_flat, seg_hidden_states, seg_hidden_states)

            # 将 cross-attention 的输出 reshape 回原来的形状，并与原特征融合
            attn_output = attn_output.permute(1, 2, 0).view_as(x)
            x = x + attn_output  # 将 cross-attention 的结果与原始特征相加

        x = self.dropout(x)

        return x
def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]
def forward_model(question, pixel_values,
                  tokenizer, model, llm,
                  projector_text2vision,
                  gen_config, stop_criteria):
    # pixel_values = projector(
    #     visual_outputs.hidden_states[args.visual_select_layer][:, 1:])

    inputs = question
    # print("Question: ", inputs)
    chunk_encode = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    assert len(chunk_encode) == 2
    ids = []
    hooks = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    ids = torch.tensor(ids).cuda().unsqueeze(0)
    mm_inputs = prepare_inputs_labels_for_multimodal(
        llm=llm, input_ids=ids, pixel_values=pixel_values)
    # mm_inputs['inputs_embeds'] = mm_inputs['inputs_embeds'].to(torch.float16)
    generate_output = llm.generate(
        **mm_inputs,
        generation_config=gen_config,
        streamer=None,
        bos_token_id=tokenizer.bos_token_id,
        stopping_criteria=stop_criteria,
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    predict = tokenizer.decode(
        # generate_output.sequences[0], skip_special_tokens=True).strip()
        generate_output.sequences[0]).strip()
    print("Answer:", predict)

    hidden_states = generate_output.hidden_states  #47*33*1   但这里的1其实是个元组是个4096的元组，
    #也就是说for item in hidden_states就取出了每一个token的最后一层的输出


    # # 获取倒数第三层的特征
    # third_last_hidden_states = [item[-3][0] for item in hidden_states]
    # third_last_hidden_states = torch.cat(third_last_hidden_states, dim=0)
    # # 获取倒数第二层的特征
    # second_last_hidden_states = [item[-2][0] for item in hidden_states]
    # second_last_hidden_states = torch.cat(second_last_hidden_states, dim=0)
    # 获取倒数第一层的特征
    last_hidden_states = [item[-1][0] for item in hidden_states] #这里的第一个有307个，然后后面的46个有1个，一共是353个
    last_hidden_states = torch.cat(last_hidden_states, dim=0)  #353*4096 47个输出token对应的Embedding


    seg_hidden_states = get_seg_hidden_states(
        last_hidden_states, generate_output.sequences[0][:-1],
        seg_id=model.seg_token_idx
    )
    # # 处理倒数第二层的特征
    # seg_hidden_states_second_last = get_seg_hidden_states(
    #     second_last_hidden_states, generate_output.sequences[0][:-1],
    #     seg_id=model.seg_token_idx
    # )
    # seg_hidden_third_states = get_seg_hidden_states(
    #    third_last_hidden_states, generate_output.sequences[0][:-1],
    #     seg_id=model.seg_token_idx
    # )
    # seg_hidden_states = seg_hidden_states.to(torch.float32)
    # print("Mask num: ", len(seg_hidden_states))

    # seg_hidden_states = projector_text2vision(seg_hidden_states)
    return predict, seg_hidden_states


class TADPSeg(SegmentationModel):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 decode_head=None,
                 sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
                 unet_config=dict(),
                 class_embedding_path='ade_class_embeddings.pth',
                 gamma_init_value=1e-4,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 use_decoder=False,
                 cfg=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        # get config from *args and **kwargs
        self.args= args
        self.cfg = cfg
        print("dataset_len",self.cfg["dataset_len"])
        self.cfg['original_tc_str'] = self.cfg['text_conditioning']  #class_emb

        self.texts = []
        self.batch_count = 0
        self.gradient_threshold = 70.0  # 设置一个梯度范数的阈值
        self.top_n = 5  # 打印前5个最大梯度的层
        # turn text conditioning into list
        if '+' in self.cfg['text_conditioning']:
            self.cfg['text_conditioning'] = self.cfg['text_conditioning'].split('+')
        else:
            self.cfg['text_conditioning'] = [self.cfg['text_conditioning']]

        ### check if model is there if not DL
        self.text2imgModel = None
        ckpt = "v1-5-pruned-emaonly.ckpt"    #这个是sd的模型
        repo = "runwayml/stable-diffusion-v1-5"
        out_dir = "checkpoints"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(os.path.join(out_dir, ckpt)):
            hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename=ckpt, local_dir=out_dir)

        config = OmegaConf.load('stable_diffusion/configs/stable-diffusion/v1-inference.yaml')
        config.model.params.ckpt_path = f'./{sd_path}'

        if 'paligemma' in self.cfg["original_tc_str"]:
            #omg_config
            # load config
            omgcfg = Config.fromfile(cfg['config'])
            model_name = omgcfg.model.type if isinstance(omgcfg.model.type,
                                                      str) else omgcfg.model.type.__name__
            if 'LLaVAModel' in model_name:
                omgcfg.model.pretrained_pth = None
            omgmodel = BUILDER.build(omgcfg.model)
            # 打印词汇表中的倒数 20 个词
            # print("Last 20 tokens in the vocabulary:")
            # print(
            #     omgmodel.tokenizer.convert_ids_to_tokens(range(len(omgmodel.tokenizer) - 20, len(omgmodel.tokenizer))))

            backend = get_file_backend(cfg['pth_model'])
            if isinstance(backend, PetrelBackend):
                from xtuner.utils.fileio import patch_fileio
                with patch_fileio():
                    state_dict = guess_load_checkpoint(cfg['pth_model'])
            else:
                state_dict = guess_load_checkpoint(cfg['pth_model'])
            omgmodel.load_state_dict(state_dict, strict=False)
            path=cfg['pth_model']
            print(f'omg model Load PTH model from {path}')
            # image_processor_cfg = copy.deepcopy(cfg.image_processor)
            image_processor = omgcfg.image_processor
            image_processor_type = image_processor['type']
            del image_processor['type']
            image_processor = image_processor_type(**image_processor)

            llm = omgmodel.llm
            tokenizer = omgmodel.tokenizer
            print(f"Initial tokenizer vocab size: {len(tokenizer)}")
            self.tokenizer=tokenizer
            self.omgmodel=omgmodel

            self.omgmodel.eval()

            self.llm=llm
            self.llm.eval()
            visual_encoder = omgmodel.visual_encoder
            projector = omgmodel.projector
            projector_text2vision = omgmodel.projector_text2vision
            self.projector_text2vision=projector_text2vision.cuda()


            self.projector=projector
            self.projector.cuda()
            self.projector.eval()

            self.visual_encoder=visual_encoder
            self.visual_encoder.cuda()
            self.visual_encoder.eval()
            stop_words = cfg['stop_words']
            print("stop_words", stop_words)
            if cfg['prompt_template']:
                template = PROMPT_TEMPLATE[cfg['prompt_template']]
                stop_words += template.get('STOP_WORDS', [])
            print("stop_words", stop_words)
            stop_criteria = get_stop_criteria(
                tokenizer=tokenizer, stop_words=stop_words)
            self.stop_criteria=stop_criteria
            gen_config = GenerationConfig(
                max_new_tokens=cfg['max_new_tokens'],
                do_sample=True,  # 改为True以增加多样性
                temperature=0.7,  # 添加温度参数
                top_p=0.9,  # 添加top_p参数
                min_length=40,  # 添加这一行
                min_new_tokens=100,
                # do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            self.gen_config=gen_config
            print(f"Final tokenizer vocab size: {len(tokenizer)}")

            self.GCG_QUESTIONS = [
        'Could you please give me a detailed description of the image at least 40 words? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
        'Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
        'Please describe in detail the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
        'Could you give a comprehensive explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
        'Could you give me an elaborate explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
        'Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
    ]




        # For present_class_embeds_only 设置是否只使用存在的类别嵌入，如果是，则获取空字符串的CLIP输出作为空类别嵌入。
        self.present_class_embeds_only = cfg['present_class_embeds_only']  #默认是false

        if self.present_class_embeds_only:
            temp_model = instantiate_from_config(config.model).to('cuda')
            empty_str_clip_output = temp_model.get_learned_conditioning([""])
            self.empty_class_embed = empty_str_clip_output[0, -1]

        #如果使用类别嵌入，则修改条件阶段配置。
        if self.cfg['original_tc_str'] == 'class_embs': #对会进入
            config.model.params.cond_stage_config.target = 'stable_diffusion.ldm.modules.encoders.modules.AbstractEncoder'

        # prepare the unet
        # sys.path.append('./stable_diffusion/ldm')
        # sd_model = instantiate_from_config(config.model)
        #实例化Stable Diffusion模型。
        sd_model = instantiate_from_config(config.model)

        # handle logic for using scaled encoder 设置编码器，根据配置决定是否使用缩放编码和解码器。
        #缩放编码（scaled encode）通常指的是在编码过程中对特征进行尺度调整的技术。这可能包括对输入数据进行归一化或标准化，
        # 或者在网络的不同层之间调整特征的尺度，以便更好地适应模型的后续处理。
        print("use_scaled_encode",self.cfg['use_scaled_encode'])
        if not self.cfg.get('use_scaled_encode', False):  #默认是false
            self.encoder_vq = sd_model.first_stage_model
            sd_model.first_stage_model = None
            if self.cfg['use_decoder_features']: #这里是false
                use_decoder = True
            if not use_decoder:
                del self.encoder_vq.decoder
            ### set grads to zero to be safe
            for param in self.encoder_vq.parameters():
                param.requires_grad = False
        else:
            if not use_decoder:
                del sd_model.first_stage_model.decoder

        if 'chatgpt' in self.cfg["original_tc_str"]:
            if self.cfg["long_clip"]==True:
                clip_model, _ = longclip.load("/home/yufei/TADP/Long_CLIP/checkpoints/longclip-L.pt")
                self.clip_model = clip_model.to('cuda')
                # 冻结 CLIP 模型的所有参数
                for param in self.clip_model.parameters():
                    param.requires_grad = False
            # 初始化 caption embedding
            caption_embeddings = []

            # 获取每个描述嵌入，并扩展到批次大小
            for caption_text in ["The overview describe", "more detail describe", "the most detail describe"]:
                if self.cfg['long_clip']== True:
                    batch_size = 8
                    caption_text_batch = [caption_text] * batch_size
                    tokens = longclip.tokenize(caption_text_batch).to('cuda')
                    embedding ,_= self.clip_model.encode_text(tokens)  # 获取嵌入，形状为 [batch_size, 248, 768]

                    # 扩展为批次大小的嵌入，最终形状为 [batch_size, 77, 768]

                    #batch_embedding = embedding.expand(batch_size, -1, -1)  # [batch_size, 248, 768]

                    # 添加到 caption_embeddings 列表
                    caption_embeddings.append(embedding)
                else:
                    embedding = sd_model.to('cuda').get_learned_conditioning([caption_text])  # 获取嵌入，形状为 [1, 248, 768]

                    # 扩展为批次大小的嵌入，最终形状为 [batch_size, 77, 768]
                    batch_size = 8
                    batch_embedding = embedding.expand(batch_size, -1, -1)  # [batch_size, 77, 768]

                    # 添加到 caption_embeddings 列表
                    caption_embeddings.append(batch_embedding)

            # 将列表中的嵌入拼接起来，形成 [batch_size, 3, 77, 768] 的张量
            caption_embeddings = torch.stack(caption_embeddings, dim=1)  # [batch_size, 3, 77, 768]

            self.model = UNetWrapper2(sd_model.model,caption_embeddings, cfg['long_clip'],**unet_config)  # 创建UNetWrapper实例，包装Stable Diffusion的UNet模型。
        else:
            self.model = UNetWrapper(sd_model.model, **unet_config)  # 创建UNetWrapper实例，包装Stable Diffusion的UNet模型。
        sd_model.model = None
        keep_cond = False

        #如果配置中包含 'blip'，则加载 BLIP 生成的图像描述，这个比classname和classemb要好，用这个最后是
        if 'blip' in self.cfg["original_tc_str"]:
            with open(self.cfg['blip_caption_path'], 'r') as f:
                self.blip_captions = json.load(f)
                # get max length
                self.blip_max_length = max([len(caption) for caption in self.blip_captions])#计算描述的最大长度
            keep_cond = True#设置 keep_cond 为 True，表示需要保留条件模型
        if 'omg' in self.cfg["original_tc_str"]:
            with open(self.cfg['blip_caption_path'], 'r') as f:
                self.blip_captions = json.load(f)
                # get max length
                self.blip_max_length = max([len(caption) for caption in self.blip_captions])#计算描述的最大长度
            keep_cond = True#设置 keep_cond 为 True，表示需要保留条件模型
        if 'paligemma' in self.cfg["original_tc_str"]:
            with open(self.cfg['blip_caption_path'], 'r') as f:
                self.blip_captions = json.load(f)
                # get max length
                self.blip_max_length = max([len(caption) for caption in self.blip_captions])#计算描述的最大长度
            keep_cond = True#设置 keep_cond 为 True，表示需要保留条件模型
        if 'chatgpt' in self.cfg["original_tc_str"]:
            with open(self.cfg['blip_caption_path'], 'r') as f:
                self.blip_captions = json.load(f)
                # get max length
                #print(f)
                self.blip_max_length = max([len(caption) for caption in self.blip_captions])#计算描述的最大长度
                #print("the max length",self.blip_max_length)
            keep_cond = True#设置 keep_cond 为 True，表示需要保留条件模型

        #如果配置指定只使用存在的类别嵌入，则加载存在的类别信息，默认为false
        if self.cfg['present_class_embeds_only']:
            with open(self.cfg['present_classes_path'], 'r') as f:
                self.present_classes = json.load(f)

        if 'class_names' in self.cfg['text_conditioning']:  #这种是class_names的那种，把class名字embedding，后连起来
            #如果使用类别名称作为文本条件，则为每个类别名称生成嵌入
#使用 Stable Diffusion 的条件模型生成这些嵌入
#去除每个嵌入的 EOS（结束）标记，并将所有嵌入连接在一起
            with torch.no_grad():
                sd_model.cond_stage_model.to('cuda')
                class_emb_stack = []
                all_pos = 0
                eos_token = 49407
                for i, class_name in enumerate(self.class_names):
                    _emb, tokens = sd_model.get_learned_conditioning(class_name, return_tokens=True)
                    if len(class_emb_stack) == 0:
                        eos_pos = torch.where(tokens == eos_token)[1][0].item()
                        all_pos = eos_pos
                        class_emb_stack.append(_emb[:, :eos_pos])
                    else:
                        eos_pos = torch.where(tokens == eos_token)[1][0].item()
                        all_pos += (eos_pos - 1)
                        class_emb_stack.append(_emb[:, 1:eos_pos])

                self.class_names_embs = torch.cat(class_emb_stack, dim=1)
#如果不需要保留条件模型，则删除它
#否则，根据配置决定是否训练条件模型
        if not keep_cond:
            del sd_model.cond_stage_model
        else:
            if self.cfg['cond_stage_trainable']: #默认false
                for param in sd_model.cond_stage_model.parameters():
                    param.requires_grad = True
            else:
                for param in sd_model.cond_stage_model.parameters():
                    param.requires_grad = False

        self.use_decoder = use_decoder
        self.sd_model = sd_model

        # check if class_embedding_path exists
        if not os.path.exists(class_embedding_path):
            print('No class embeddings provided!, please run create_class_embeddings.py --dataset pascal')

        class_embeddings = torch.load(class_embedding_path)#加载类别嵌入
        self.register_buffer('class_embeddings', class_embeddings)#注册类别嵌入为缓冲区
        text_dim = class_embeddings.size(-1)
        self.gamma = nn.Parameter(torch.ones(text_dim) * gamma_init_value)#初始化 gamma 参数
        self.text_adapter = TextAdapter(text_dim=text_dim)#创建文本适配器实例

        # check if using the correct class embeddings
        assert class_embeddings.size(0) == self.n_classes

        self.with_neck = True
        self.decode_head = nn.Module()#创建解码头模块

        enc_mid_channels, enc_end_channels = self.compute_decoder_head_shapes()#计算解码头的通道数
        #pdb.set_trace()
        if self.cfg["decode_head"] == 'FPN':#根据配置初始化 FPN 或 DeepLabV3+ 解码头,这里默认选 FPN original_FPNDecoder这个
            if self.cfg['use_decoder_features']:

                self.decode_head.decoder = FPNDecoder(
                    encoder_channels=(384, 512, 512, 1856, enc_mid_channels, enc_end_channels, 1280),
                   # encoder_channels=(5120, 512, 512, 1856, enc_mid_channels, enc_end_channels, 1280),
                    encoder_depth=7,
                    pyramid_channels=256,
                    segmentation_channels=128,
                    dropout=0.2,
                    merge_policy="add",
                )
            else:
                self.decode_head.decoder = original_FPNDecoder(
                    encoder_channels=(320, enc_mid_channels, enc_end_channels, 1280),
                    encoder_depth=4,
                    pyramid_channels=256,
                    segmentation_channels=128,
                    dropout=0.2,
                    merge_policy="add",
                )
                # self.decode_head.decoder = FPNDecoder2(
                #     encoder_channels=(320, enc_mid_channels, enc_end_channels, 1280),
                #     encoder_depth=4,
                #     pyramid_channels=256,
                #     segmentation_channels=128,
                #     dropout=0.2,
                #     merge_policy="add",
                # )
        elif self.cfg["decode_head"] == 'deeplabv3plus':
            self.decoder = DeepLabV3PlusDecoder(
                encoder_channels=(320, enc_mid_channels, enc_end_channels, 1280),
                out_channels=256,
                atrous_rates=(12, 24, 36),
                output_stride=16,
            )
        else:
            raise NotImplementedError

        self.decode_head.segmentation_head = SegmentationHead(
            in_channels=self.decode_head.decoder.out_channels,
            out_channels=self.n_classes,
            activation=None,
            kernel_size=1,
            upsampling=8,
        )
        self.decode_head.num_classes = self.n_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.cfg["freeze_text_adapter"]:
            for param in self.text_adapter.parameters():
                param.requires_grad = False

        if self.cfg["use_token_embeds"]:
            self.reduce_embeds = nn.Linear(768, 8)

    def initialize_model(self):
        pass

    # def load_state_dict(self, state_dict, strict=True):
    #     # 如果需要，在这里进行任何必要的状态字典转换
    #     return super().load_state_dict(state_dict, strict=strict)
    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def compute_decoder_head_shapes(self):

        text_cond = self.cfg['text_conditioning']  #这里我们应该是用blip，效果最好
        enc_mid_channels = 640
        enc_end_channels = 1280
        if self.cfg['append_self_attention']:#默认false
            enc_mid_channels += 1024
            enc_end_channels += 256

        if self.cfg.get('use_attn', True):#默认为true
            if 'blip' in text_cond:
                enc_mid_channels += 77
                enc_end_channels += 77
            if 'omg_llava' in text_cond:
                enc_mid_channels += 77
                enc_end_channels += 77
            if 'paligemma' in text_cond:
                enc_mid_channels += 77
                enc_end_channels += 77
            if 'chatgpt' in text_cond:
                if self.cfg.get('long_clip', True):
                    enc_mid_channels += 248
                    enc_end_channels += 248
                else:
                    enc_mid_channels += 77
                    enc_end_channels += 77
            if 'class_names' in text_cond:
                enc_mid_channels += self.class_names_embs.shape[1]
                enc_end_channels += self.class_names_embs.shape[1]

            if 'class_emb' in text_cond:
                enc_mid_channels += self.n_classes
                enc_end_channels += self.n_classes

        return enc_mid_channels, enc_end_channels

    def create_text_embeddings(self, img_metas, latents):
        bsz = len(latents)
        text_cond = self.cfg['text_conditioning']
        conds = []
        texts = None

        if 'blip' in text_cond:
            texts = []
            _cs = []
            for img_id in img_metas['img_id']:

                text = self.blip_captions[img_id]['captions']  #他好像是提前用blip生成对应的caption然后直接读取了
                c = self.sd_model.get_learned_conditioning(text)  #从提示的文本中去拿到embedding
                texts.append(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)
        if 'omg_llava' in text_cond:
            texts = []
            _cs = []
            for img_id in img_metas['img_id']:
                text = self.blip_captions[img_id]['captions']  # 他好像是提前用blip生成对应的caption然后直接读取了
                c = self.sd_model.get_learned_conditioning(text)  # 从提示的文本中去拿到embedding
                texts.append(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)
        if 'paligemma' in text_cond:
            texts = []
            _cs = []
            for img_id in img_metas['img_id']:
                text = self.blip_captions[img_id]['captions']  # 他好像是提前用blip生成对应的caption然后直接读取了
                c = self.sd_model.get_learned_conditioning(text)  # 从提示的文本中去拿到embedding
                texts.append(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)
        if 'chatgpt' in text_cond:
            #还得改
            if self.cfg['long_clip']== True:

                texts = []
                for img_id in img_metas['img_id']:
                    full_text = self.blip_captions[img_id]['captions']  # 他好像是提前用blip生成对应的caption然后直接读取了

                    full_text = " ".join(full_text)
                    text_parts = full_text.split("\n\n")
                    if len(text_parts) == 3:
                        text1, text2, text3 = text_parts
                    else:
                        raise ValueError(f"Expected 3 text parts, but got {len(text_parts)} for img_id {img_id}")
                    texts.append([text1, text2, text3])
                # 将 texts 转换为扁平化的结构，即 batch_size*3 的形式
               # pdb.set_trace()
                flattened_texts = [text for sample_texts in texts for text in sample_texts]
                tokens = longclip.tokenize(flattened_texts).to('cuda')  #现在相当于是先第一个sample的3个，接着第二个sample的3个
                embedding, _ = self.clip_model.encode_text(tokens)  # 得到嵌入，形状为 [batch_size * 3, 248, 768]
                embedding = embedding.view(len(texts), 3, 248, 768)
                    # # Get the embeddings for each of the three text parts
                    # c1, _ = self.clip_model.encode_text.to('cuda')(longclip.tokenize(text1))
                    # c2, _ = self.clip_model.encode_text.to('cuda')(longclip.tokenize(text2))
                    # c3, _ = self.clip_model.encode_text.to('cuda')(longclip.tokenize(text3))
                    #   # [1, 248, 768]
                    #
                    # # c = self.sd_model.get_learned_conditioning(text)  # 从提示的文本中去拿到embedding
                    # # Store the individual texts
                    # texts.extend([text1, text2, text3])
                    # combined_embedding = torch.cat([c1, c2, c3], dim=0)  # [3, 77, 768]
                    # _cs.append(combined_embedding.unsqueeze(0))
            else:
                texts = []
                _cs = []
                #pdb.set_trace()
                for img_id in img_metas['img_id']:
                    full_text = self.blip_captions[img_id]['captions']  # 他好像是提前用blip生成对应的caption然后直接读取了

                    full_text = " ".join(full_text)
                    text_parts = full_text.split("\n\n")
                    if len(text_parts) == 3:
                        text1, text2, text3 = text_parts
                    else:
                        raise ValueError(f"Expected 3 text parts, but got {len(text_parts)} for img_id {img_id}")
                    # Get the embeddings for each of the three text parts
                    c1 = self.sd_model.get_learned_conditioning(text1)#[1, 77, 768]
                    c2 = self.sd_model.get_learned_conditioning(text2)
                    c3 = self.sd_model.get_learned_conditioning(text3)
                    #c = self.sd_model.get_learned_conditioning(text)  # 从提示的文本中去拿到embedding
                    # Store the individual texts
                    texts.extend([text1, text2, text3])
                    combined_embedding = torch.cat([c1, c2, c3], dim=0)#[3, 77, 768]
                    _cs.append(combined_embedding.unsqueeze(0))
                #texts.append(text)
                #_cs.append(c)
       # pdb.set_trace()
        if self.cfg['long_clip']== False:
            c = torch.cat(_cs, dim=0)
            conds.append(c)
        #pdb.set_trace()
        if 'blank_str' in text_cond:
            texts = []
            _cs = []
            for img_id in img_metas['img_id']:
                text = ['']
                c = self.sd_model.get_learned_conditioning(text)
                texts.append(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)

        if 'class_names' in text_cond:
            _cs = []
            for img_id in img_metas['img_id']:
                _cs.append(self.class_names_embs)
            c = torch.cat(_cs, dim=0)
            conds.append(c)

        if 'class_emb' in text_cond:
            c_crossattn = self.class_embeddings.repeat(bsz, 1, 1).clone()

            if self.present_class_embeds_only:
                for img_idx, img_id in enumerate(img_metas['img_id']):
                    present_classes = self.present_classes[img_id]['captions'][0]
                    # print(img_idx, img_id, present_classes)
                    for class_idx in range(c_crossattn.shape[1]):
                        if class_idx not in present_classes:
                            c_crossattn[img_idx, class_idx, :] = self.empty_class_embed
            conds.append(c_crossattn)
        if self.cfg['long_clip']== True:
            c_crossattn =embedding
        else:
            c_crossattn = torch.cat(conds, dim=1) #这里相当于把一堆blip产生的caption全部都cat到一起，这里因为会有一个batch的

        #c_crossattn [16, 77, 768]
        if self.cfg['use_text_adapter']:#这里应该用text_inversion的话效果会好
            c_crossattn = self.text_adapter(c_crossattn, self.gamma)#相当于用text_inversion去调整

        if texts is not None:
            self.texts = texts
        #pdb.set_trace()
        return c_crossattn#[24, 77, 768]

    def extract_feat(self, img, img_metas):
        """Extract features from images."""
        if self.cfg.get('use_scaled_encode', False): #默认是false,但我这里用的true
            with torch.no_grad():
                latents = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(img))
        else:
            with torch.no_grad():
                latents = self.encoder_vq.encode(img)
                #使用 VQ（Vector Quantized）编码器编码图像，并获取量化后的潜在表示。
            latents = latents.mode().detach()

        #latents shape [16, 4, 64, 64]
        c_crossattn = self.create_text_embeddings(img_metas, latents)  #latents是img的embedding，但这里的c-crossattn应该只是text的embedding
        #创建文本嵌入，可能用于交叉注意力机制。
        #pdb.set_trace()
        t = torch.from_numpy(np.array([1])).to(img.device)
        #创建时间步 tensor（这里固定为1），并将潜在表示、时间步和交叉注意力嵌入传入模型，得到输出
        #pdb.set_trace()
        if self.cfg['long_clip']==True:
             c_crossattn = c_crossattn.to(torch.float32)
        #pdb.set_trace()
        outs = self.model(latents, t, c_crossattn=[c_crossattn])

        if self.cfg['use_decoder_features']:# 如果使用解码器特征,这里默认不使用
            hooks = []
            decoder_outs = []
            # 为解码器的每个主要层注册hook
            def hook_fn(module, input, output):
                decoder_outs.append(output)
            for name, module in self.encoder_vq.named_modules():
                if 'decoder' in name and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    hooks.append(module.register_forward_hook(hook_fn))

            # 运行解码过程
            #with torch.no_grad():
            _ = self.encoder_vq.decode(latents)

            # 移除所有hooks
            for hook in hooks:
                hook.remove()

            # grouped_tensors = group_and_process_tensors(decoder_outs)
            #
            # # 处理64x64的张量
            # chan_64 = torch.cat(grouped_tensors[64][:3], dim=1)
            # outs[0] = torch.cat([outs[0], chan_64], dim=1)
            #
            # # 插入调整后的128x128张量
            # if grouped_tensors[128]:
            #     outs.insert(0, adjust_and_cat_tensors(grouped_tensors[128], 64))
            #
            # # 插入调整后的256x256张量
            # if grouped_tensors[256]:
            #     outs.insert(0, adjust_and_cat_tensors(grouped_tensors[256], 64))
            #
            # # 插入调整后的512x512张量
            # if grouped_tensors[512]:
            #     outs.insert(0, adjust_and_cat_tensors(grouped_tensors[512], 64))

            # 确保我们至少有6个输出
            if len(decoder_outs) >= 6:
                chan_64 = torch.cat(decoder_outs[:3], dim=1)    #39*4*512*64*64
                outs[0] = torch.cat([outs[0], chan_64], dim=1)
                outs.insert(0, decoder_outs[3])
                outs.insert(0, decoder_outs[4])
                outs.insert(0, torch.cat(decoder_outs[5:15], dim=1))
            else:
                print(f"Warning: Not enough decoder outputs. Got {len(decoder_outs)}, expected at least 6.")


            #decoded, decoder_outs = self.encoder_vq.decode_blocks(latents)#使用 VQ 编码器的解码块处理潜在表示

            # stack decoder outs[:3] along first dim
            # chan_64 = torch.cat(decoder_outs[:3], dim=1)# 将解码器的前三个输出在通道维度上拼接
            # outs[0] = torch.cat([outs[0], chan_64], dim=1)#将拼接结果与模型输出的第一个元素拼接
            # outs.insert(0, decoder_outs[3])#插入解码器的第4和第5个输出到 outs 的开头
            # outs.insert(0, decoder_outs[4])
            # outs.insert(0, torch.cat(decoder_outs[5:], dim=1))#将剩余的解码器输出拼接并插入到 outs 的开头





        return outs  #4,16,320,64

    def forward_train(self, img, img_metas, gt_semantic_seg): #这个相当于就返回了loss
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img, img_metas)

        if self.with_neck: #默认会进来

            x = self.neck(x)

        return x

    def forward(self, x, img_metas):

        x = x.permute(0, 3, 1, 2).float()
        if self.normalize_images:
            # x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
            x = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x)

        #x [16, 3, 512, 512]
        x = self.extract_feat(x, img_metas)
        # x #4,16,320,64,64

        #pdb.set_trace()
        #print("到了decoder了")
        #7,4,5120,64,64
        # 4, 5120, 64, 64]   512
        # 4, 512, 64, 64    256
        # [4, 512, 64, 64])128
        # [4, 1856, 64, 64])
        # [4, 717, 32, 32])
        # [4, 1357, 16, 16])
        # ([4, 1280, 8, 8])


        if self.with_neck:
            # TODO: needs to be adjusted for deeplabv3+ architecture  #现在变成了7,4,5120,64,64
            #x = self.decode_head.decoder(*x,seg_hidden_states=seg_hidden_states)#16, 128, 64, 64
            x = self.decode_head.decoder(*x)  # 16, 128, 64, 64
            x = self.decode_head.segmentation_head(x)

        return x

    def initialize_loss(self):
        loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=self.ignore_index)
        return loss

    def configure_optimizers(self):
        lesslr_no_decay = list()
        lesslr_decay = list()
        no_lr = list()
        no_decay = list()
        decay = list()
        for name, m in self.named_parameters():
            if 'unet' in name and 'norm' in name:
                lesslr_no_decay.append(m)
            elif 'unet' in name:
                lesslr_decay.append(m)
            elif 'encoder_vq' in name:
                no_lr.append(m)
            elif 'norm' in name:
                no_decay.append(m)
            elif 'embedding_manager' in name:
                pass
            else:
                decay.append(m)

        params_to_optimize = [
            {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale': 0.01}, #UNet中的归一化层，使用较低的学习率且不使用权重衰减
            {'params': lesslr_decay, 'lr_scale': 0.01}, #UNet中的其他层，使用较低的学习率
            {'params': no_lr, 'lr_scale': 0.0},         #编码器VQ层，不进行优化（学习率为0）
            {'params': no_decay, 'weight_decay': 0.0}, #其他归一化层，不使用权重衰减
            {'params': decay},  #其他所有参数
        ]
        base_lr = 0.00001  # 设置基础学习率
        warm_up_steps = 10000
        constant_steps = 4 * self.cfg["dataset_len"]  # 4 epochs of constant learning rate
        total_steps = self.cfg["dataset_len"] * self.cfg["max_epochs"]

        optimizer = torch.optim.AdamW(params_to_optimize,
                                      lr=base_lr*math.sqrt(self.cfg['num_gpus']*self.cfg['batch_size']/8),
                                      weight_decay=1e-2,
                                      amsgrad=False
                                      )
        #logger.info(f"max_epochs: {self.cfg['max_epochs']}, dataset_len: {self.cfg['dataset_len']}")
        def lr_lambda(current_step: int):
            if current_step < warm_up_steps:
                return float(current_step) / float(max(1, warm_up_steps))
            elif current_step < warm_up_steps + constant_steps:
                return 1.0
            else:
                progress = (current_step - warm_up_steps - constant_steps) / (
                            total_steps - warm_up_steps - constant_steps)
                return max((1 - progress) ** 0.9, 1e-6)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            #lr_lambda=lambda x: max((1 - x / (self.cfg["dataset_len"] * self.cfg["max_epochs"])) ** 0.9, 1e-6)
            lr_lambda = lambda x: log_and_calculate_lr(x, self.cfg)
        )
        if self.cfg['use_warmup_scheduler']:
            lr_scheduler = LambdaLR(optimizer, lr_lambda)


        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}]

    # def on_after_backward(self):
    #     self.batch_count += 1
    #     if self.batch_count >= 14:  # 只在第14个batch开始执行
    #         gradients = []
    #         total_norm = 0
    #         for name, param in self.named_parameters():
    #             if param.grad is not None:
    #                 grad_norm = param.grad.data.norm(2).item()
    #                 total_norm += grad_norm ** 2
    #                 gradients.append((name, grad_norm))
    #
    #         total_norm = total_norm ** 0.5
    #         print(f"Batch {self.batch_count}, Total Gradient Norm: {total_norm:.4f}")
    #
    #         # 排序并打印前N个最大梯度的层
    #         gradients.sort(key=lambda x: x[1], reverse=True)
    #         print(f"Top {self.top_n} layers with largest gradients:")
    #         for name, norm in gradients[:self.top_n]:
    #             print(f"  Layer: {name}, Gradient Norm: {norm:.4f}")
    #
    #         if total_norm > 50:  # 降低警告阈值
    #             print("Warning: Total gradient norm is large and may cause instability!")
    #

    def training_step(self, batch, batch_idx):
        images, masks, img_metas,question,image_1024 = batch
        image_1024 = image_1024[0]
        #image_1024 = image_1024.cuda().unsqueeze(0).to(self.visual_encoder.dtype)
        #image_1024 = image_1024.cuda().to(self.visual_encoder.dtype)
        #visual_outputs = self.visual_encoder(image_1024, output_hidden_states=True)
        # if isinstance(visual_outputs, list) or isinstance(visual_outputs, tuple) \
        #         or isinstance(visual_outputs, torch.Tensor):
        #     pixel_values = self.projector(visual_outputs)
        # else:
        #     pixel_values = self.projector(
        #         visual_outputs.hidden_states[args.visual_select_layer][:, 1:])

        # questions="Could you please give me a detailed description of the image at least 40 words? Please respond with interleaved segmentation masks for the corresponding parts of the answer."
        # texts = DEFAULT_IMAGE_TOKEN + '\n' + questions
        # if self.cfg['prompt_template']:
        #     prompt_text = ''
        #     template = PROMPT_TEMPLATE[self.cfg['prompt_template']]
        #     prompt_text += template['INSTRUCTION'].format(
        #         input=texts, round=1, bot_name=cfg['bot_name'])
        # else:
        #     prompt_text = texts
        # batch_inputs = prompt_text
        # _, seg_hidden_states = forward_model(
        #     batch_inputs, pixel_values,
        #     self.tokenizer, self.omgmodel, self.llm,
        #     self.projector_text2vision,
        #     self.gen_config, self.stop_criteria)
        #
        # pdb.set_trace()
        seg_hidden_states=[]


       # loss = self._step(images, masks, img_metas, "train",seg_hidden_states)

        loss = self._step(images, masks, img_metas, "train")
        sch = self.lr_schedulers()
        sch.step()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in loss: {loss}")
        return loss
# def validation_step(self, batch, batch_idx):#, dataloader_idx=0):
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Skip the first 90 batches
        # if batch_idx < 180:
        #     return
        images, masks, img_metas,question,image_1024 = batch
        image_1024=image_1024[0]
        #image_1024 = image_1024.cuda().to(self.visual_encoder.dtype)
        #visual_outputs = self.visual_encoder(image_1024, output_hidden_states=True)
        #if isinstance(visual_outputs, list) or isinstance(visual_outputs, tuple) \
        #        or isinstance(visual_outputs, torch.Tensor):
        #    pixel_values = self.projector(visual_outputs)
        #else:
       #     pixel_values = self.projector(
       #         visual_outputs.hidden_states[args.visual_select_layer][:, 1:])

       # questions = "Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer."
        #texts = DEFAULT_IMAGE_TOKEN + '\n' + questions
       # if self.cfg['prompt_template']:
       #     prompt_text = ''
       #     template = PROMPT_TEMPLATE[self.cfg['prompt_template']]
       #     prompt_text += template['INSTRUCTION'].format(
       #         input=texts, round=1, bot_name=self.cfg['bot_name'])
       # else:
       #     prompt_text = texts
        #batch_inputs = prompt_text
        # _, seg_hidden_states = forward_model(
        #     batch_inputs, pixel_values,
        #     self.tokenizer, self.omgmodel, self.llm,
        #     self.projector_text2vision,
        #     self.gen_config, self.stop_criteria)
        #pdb.set_trace()
        seg_hidden_states=[]
        #print(f"Validating image: {img_metas['img_id']}")
        #self._step(images, masks, img_metas, f"val_{dataloader_idx}",seg_hidden_states)
        loss =self._step(images, masks, img_metas, f"val_{dataloader_idx}")
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in loss: {loss}")
    



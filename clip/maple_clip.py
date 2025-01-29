import pdb

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *


import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clipm import clip
from clipm.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT = '~/.cache/clip'

def load_clip_to_cpu(backbone, n_ctx=2):
    url = clip._MODELS[backbone]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": n_ctx}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        # x = prompts + self.positional_embedding.type(self.dtype)
        pdb.set_trace()
        if len(prompts.size()) > 3:
            tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(prompts.size(0), 1, 1)
            tokenized_prompts = tokenized_prompts.view(-1, tokenized_prompts.size()[-1])
        x = prompts.view(-1, prompts.size()[-2], prompts.size()[-1]) + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, batch_size=None, n_ctx=2, ctx_init='a photo of a', ctx_position='end',
                 num_p=1):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = n_ctx
        ctx_init = ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)

        # initialize mixture online prompts
        learn_vectors = torch.zeros(embedding[0, 1 + (n_ctx0 - n_ctx): 1 + n_ctx0, :].size(), dtype=dtype)
        if num_p > 1:
            learn_vectors = learn_vectors.unsqueeze(0).repeat(num_p, 1, 1)  # (N, L, D)
        self.octx_init_state = learn_vectors.detach().clone()
        self.octx = nn.Parameter(learn_vectors)  # to be optimized
        self.ctx_order = list(range(num_p))
        self.ctx_use = [0] * num_p


        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def reset(self):
        # pdb.set_trace()
        learn_vectors = self.octx_init_state
        self.octx.copy_(learn_vectors) # to be optimized

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        pdb.set_trace()
        # ctx = self.ctx
        ctx = self.ctx + self.init_ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.batch_size is not None:
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        ################ mixture of prompts ################
        elif self.num_p > 1:
            prefix = prefix.unsqueeze(0).repeat(self.num_p, 1, 1, 1)
            suffix = suffix.unsqueeze(0).repeat(self.num_p, 1, 1, 1)


        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        pdb.set_trace()
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                 n_ctx=2, ctx_init='a photo of a', num_p=1,
                 num_select_p=1):
        super(ClipTestTimeTuning, self).__init__()
        # clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        clip = load_clip_to_cpu('ViT-B/16', n_ctx)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.num_p = num_p
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip, batch_size, n_ctx, ctx_init, ctx_position='end', num_p=num_p)
        self.criterion = criterion

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self, deep_compound_prompts_text, p_s=None):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        pdb.set_trace()
        if p_s is not None:
            t_features = self.text_encoder(prompts[p_s - 1], tokenized_prompts, deep_compound_prompts_text)
        else:
            t_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)

        # pdb.set_trace()
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image, p_s=None):
        # with torch.no_grad():
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

        if self.num_p > 1 and p_s is not None:
            text_features = self.get_text_features(deep_compound_prompts_text, p_s)
        else:
            text_features = self.get_text_features(deep_compound_prompts_text)

        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        self.image_features = image_features.mean(0)

        # if "ensemp" in args.log_date:
        #     text_features = self.get_text_features(p_s)
        # if self.num_p > 1 and p_s:


        # if self.num_p > 1:
        # pdb.set_trace()
        # allimgfeats = image_features.mean(dim=0)
        # text_features = self.get_text_features()
        # else:
        # text_features = self.get_text_features()

        logit_scale = self.logit_scale.exp()
        # pdb.set_trace()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input, p_s=None):
        # pdb.set_trace()
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input, p_s)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            pdb.set_trace()

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False, num_p=1, featmemo=None, num_select_p=1):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                               n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, num_p=num_p, featmemo=featmemo,
                               num_select_p=num_select_p)

    return model


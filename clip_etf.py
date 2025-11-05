import os.path as osp
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

    
    def forward(self, x):
        return self.trans(x)



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x



class PromptLearner_client(nn.Module):
    def __init__(self, n_ctx_num, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = n_ctx_num
        ctx_init = ''
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        CSC = True
        if CSC:
            print("Initializing class-specific contexts")
            global_ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            local_ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            global_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            local_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(global_ctx_vectors, std=0.02)
        nn.init.normal_(local_ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * 16 * 2)
        

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx * 2}")

        self.ctx_global = nn.Parameter(global_ctx_vectors)  # to be optimized
        self.ctx_local = nn.Parameter(local_ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda(1)).type(dtype)

        # print(embedding.shape)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + 16 * 2:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

        
    def forward(self, img_ave):
        # print('---ctx:',ctx.shape)
        ctx_global = self.ctx_global
        ctx_local = self.ctx_local
        if ctx_global.dim() == 2 and ctx_local.dim() ==2:
            ctx_global = ctx_global.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_local = ctx_local.unsqueeze(0).expand(self.n_cls, -1, -1)
        

        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx_local = ctx_local * img_ave

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx_global,
                ctx_local,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts
    



class CustomCLIP_client(nn.Module):
    def __init__(self, classnames, clip_model,n_ctx_num=16, domain_number=6):
        super().__init__()
        self.prompt_learner = PromptLearner_client(n_ctx_num, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model) 
        self.sem_trans = Transform()
        self.dom_trans = Transform()
        self.sem_trans.half()
        self.dom_trans.half()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        
        logit_scale = self.logit_scale.exp()

        ave_image = torch.mean(image_features, 0).unsqueeze(0).unsqueeze(1)
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner(ave_image)

        text_features = self.text_encoder(prompts, tokenized_prompts)


        image_sem_logits = self.sem_trans(image_features)
        image_dom_logits = self.dom_trans(image_features)


        with torch.no_grad():
            text_sem_logits = self.sem_trans(text_features)
            text_dom_logits = self.dom_trans(text_features)


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()
        
        return logits, image_sem_logits, image_dom_logits, text_sem_logits, text_dom_logits


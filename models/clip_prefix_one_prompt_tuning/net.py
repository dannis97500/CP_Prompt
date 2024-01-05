import torch
import torch.nn as nn
import copy

from .prompt import PrefixKeqV,PrefixKneqV

from .prompt_learner import cfgc, load_clip_to_cpu, TextEncoder, PromptLearner
from utils.class_names import core50_classnames, domainnet_classnames, cddb_classnames


class PrefixOnePromptNet(nn.Module):

    def __init__(self, args):
        super(PrefixOnePromptNet, self).__init__()
        self.cfg = cfgc()
        clip_model = load_clip_to_cpu(self.cfg)
        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.is_fix_share_prompt=args["is_fix_share_prompt"]
        self.class_num = 1
        if args["dataset"] == "cddb":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(cddb_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 2
        elif args["dataset"] == "domainnet":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(domainnet_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 345
        elif args["dataset"] == "core50":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 50
        else:
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))

        self.fix_prompt_weights=[]
        self.share_prompt=nn.Linear(args["embd_dim"], args["share_prompt_length"], bias=False)
        if args["prefix_tuning"]=="keqv":
            self.prefix_prompt = PrefixKeqV(args["embd_dim"], args["total_sessions"], args["prefix_prompt_length"],args["total_sessions"],args["prefix_prompt_layers"]) 
        elif args["prefix_tuning"]=="kneqv":
            self.prefix_prompt = PrefixKneqV(args["embd_dim"], args["total_sessions"], args["prefix_prompt_length"],args["total_sessions"],args["prefix_prompt_layers"]) 
        else:
            return ValueError('Unknown prefix_tuning: {}.'.format(args["prefix_tuning"]))
        
        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.output_dim

    def extract_vector(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features
    
    def extract_share_prompt_vector(self, image):
        image_features = self.image_encoder(image.type(self.dtype),instance_tokens=self.share_prompt.weight)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features


    def forward(self, image):
        logits = []
        image_features = self.image_encoder(image.type(self.dtype),instance_tokens=self.share_prompt.weight, prefix_prompt=self.prefix_prompt, task_id=self.numtask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.classifier_pool[self.numtask]

        tokenized_prompts = prompts.tokenized_prompts
        text_features = self.text_encoder(prompts(), tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits.append(logit_scale * image_features @ text_features.t())
        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }

    def interface(self, image, selection):
        if self.is_fix_share_prompt:
            instance_batch = torch.stack(self.fix_prompt_weights, 0)[selection, :, :]
        else:
            instance_batch=self.share_prompt.weight
        image_features = self.image_encoder(image.type(self.dtype),instance_tokens=instance_batch,prefix_prompt=self.prefix_prompt,task_id=selection)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = []
        for prompt in self.classifier_pool:
            tokenized_prompts = prompt.tokenized_prompts
            text_features = self.text_encoder(prompt(), tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits.append(logit_scale * image_features @ text_features.t())
        logits = torch.cat(logits,1)
        selectedlogit = []
        for idx, ii in enumerate(selection):
            selectedlogit.append(logits[idx][self.class_num*ii:self.class_num*ii+self.class_num])
        selectedlogit = torch.stack(selectedlogit)
        return selectedlogit


    def update_fc(self):
        self.numtask +=1
        self.fix_prompt_weights.append(copy.deepcopy(self.share_prompt.weight))

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

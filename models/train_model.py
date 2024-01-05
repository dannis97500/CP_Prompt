import torch
import torch.nn as nn
import copy

from models.clip_train.prompt_learner import cfgc, load_clip_to_cpu, TextEncoder, PromptLearner
from utils.class_names import core50_classnames, domainnet_classnames, cddb_classnames


class TrainModel(nn.Module):

    def __init__(self, args):
        super(TrainModel, self).__init__()
        self.class_num = 1
        self.class_prompt_length=args["class_prompt_length"]
        self.session_num=args["total_sessions"]
        self.non_layers=args["non_layers"]
        if args["dataset"] == "cddb":
            self.class_num = 2
        elif args["dataset"] == "domainnet":
            self.class_num = 345
        elif args["dataset"] == "core50":
            self.class_num = 50
        else:
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))
        self.cfg = cfgc()
        clip_model = load_clip_to_cpu(self.cfg,self.non_layers,self.class_num,self.session_num,self.class_prompt_length)
        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        
        self.embd_dim=args["embd_dim"]
        self.domain_prompt_length=args["domain_prompt_length"]
        
        if args["dataset"] == "cddb":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(cddb_classnames.values()), self.clip_model)
                for i in range(self.session_num)
            ])
            
        elif args["dataset"] == "domainnet":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(domainnet_classnames.values()), self.clip_model)
                for i in range(self.session_num)
            ])
            
        elif args["dataset"] == "core50":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model)
                for i in range(self.session_num)
            ])
            
        else:
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))

        self.prompt_pool = nn.ModuleList([
            nn.Linear(self.embd_dim,self.domain_prompt_length, bias=False)
            for i in range(self.session_num)
        ])

        self.numtask = -1
        

    @property
    def feature_dim(self):
        return self.image_encoder.output_dim
    
    def setCenter(self,all_class_keys):
        self.clip_model.initCenter(all_class_keys)
    def setClassPromptsParams(self,prompts_params):
        self.clip_model.setClassPromptsParams(prompts_params)
    
    def extract_vector(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def forward(self, image):
        logits = []
        image_features = self.image_encoder(image.type(self.dtype), self.prompt_pool[self.numtask-1].weight)
        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.classifier_pool[self.numtask-1]
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
        instance_batch = torch.stack([i.weight for i in self.prompt_pool], 0)[selection, :, :]
        image_features = self.image_encoder(image.type(self.dtype), instance_batch)
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


    def update_fc(self, nb_classes):
        self.numtask +=1
        self.image_encoder.transformer.task_num=self.numtask

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

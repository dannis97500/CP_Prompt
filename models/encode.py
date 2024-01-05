import torch
import torch.nn as nn
import copy

from models.clip.prompt_learner_shared import cfgc, load_clip_to_cpu, TextEncoder, PromptLearnerShared
from utils.class_names import core50_classnames, domainnet_classnames, cddb_classnames
from models.clip import clip
class Encode():

    def __init__(self, args):
        #super(Encode, self).__init__()

        self.cfg = cfgc()
        clip_model = load_clip_to_cpu(self.cfg)
        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.class_num = 1
        if args["dataset"] == "cddb":
            self.class_num = 2
        elif args["dataset"] == "domainnet":
            self.class_num = 345
        elif args["dataset"] == "core50":
            self.class_num = 50

        else:
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))

    
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features
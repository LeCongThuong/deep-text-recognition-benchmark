"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
import os
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BiLSTM
from modules.prediction import Attention, CTC_Prediction
import torch
from utils import calculate_model_params


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Transformation': opt.Transformation, 'FeatureExtraction': opt.FeatureExtraction,
                       'SequenceModeling': opt.SequenceModeling, 'Prediction': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(opt.ft_config['trans'],
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW),
                I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.ft_config['feat'], opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = BiLSTM(opt.ft_config['seq'], self.FeatureExtraction_output, opt.hidden_size)
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = CTC_Prediction(opt.ft_config['pred'], self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(opt.ft_config['pred'], self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')
        self.set_parameter_requires_grad()
        self.optimizers = self.configure_optimizers()
        self.make_statistics_params()

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Transformation'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['SequenceModeling'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Prediction'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

    def set_parameter_requires_grad(self):
        for key, value in self.stages.items():
            if value is not None:
                net = getattr(self, key)
                net.set_parameter_requires_grad()

    def load_pretrained_networks(self):
        checkpoint = torch.load(self.opt.saved_model)
        state_dict = self.state_dict()
        checkpoint = {'.'.join(k.split('.')[1:]): v for k, v in checkpoint.items()}
        checkpoint = {mapping_pretrained_key_to_current_model_key(k): v for k, v in checkpoint.items()}
        checkpoint = {k: v for k, v in checkpoint.items() if checkpoint[k].shape == state_dict[k].shape}
        self.load_state_dict(checkpoint, strict=False)

    def load_checkpoint(self, name):
        model_path = os.path.join(f'./saved_models/{self.opt.exp_name}', name)
        checkpoint = torch.load(model_path)
        for key, value in self.stages.items():
            if value is not None and self.optimizers[key] is not None:
                net = getattr(self, key)
                net.load_state_dict(checkpoint[key])
                optimizer_name = key + '_optimizer'
                self.optimizers[key].load_state_dict(checkpoint[optimizer_name])

    def save_checkpoints(self, iteration, name):
        state_dict = {}
        for key, value in self.stages.items():
            if value is not None and self.optimizers[key] is not None:
                state_dict[key] = getattr(self, key).state_dict()
                optimizer_name = key + '_optimizer'
                state_dict[optimizer_name] = self.optimizers[key].state_dict()
        state_dict['iteration'] = iteration
        save_path = os.path.join(f'./saved_models/{self.opt.exp_name}', name)
        torch.save(state_dict, save_path)

    def configure_optimizers(self):
        optimizers = {}
        for key, value in self.stages.items():
            if value is not None:
                print("In if branch")
                net = getattr(self, key)
                print("Class: ", net.__class__)
                optimizers[key] = net.configure_optimizers()
            else:
                optimizers[key] = None
        print("All optimizers: ", optimizers)
        return optimizers

    def optimize_parameters(self):
        for key, value in self.optimizers.items():
            if value is not None:
                value.step()

    def make_statistics_params(self):
        for key, value in self.stages.items():
            if value is not None:
                net = getattr(self, key)
                total_num, true_grad_num, false_grad_num = calculate_model_params(net)
                print("Net: ", net.__class__)
                print("Total parameters: ", total_num)
                print("Number of parameters requires grad: ", true_grad_num)
                print("Number of parameters do not require grad: ", false_grad_num)


def mapping_pretrained_key_to_current_model_key(key):
    if 'SequenceModeling.0' in key:
        key = key.replace('SequenceModeling.0', 'SequenceModeling.SequenceModeling.0')
    if 'SequenceModeling.1' in key:
        key = key.replace('SequenceModeling.1', 'SequenceModeling.SequenceModeling.1')
    return key

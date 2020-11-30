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

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
import re


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
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
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

    def _set_parameter_requires_grad_trans(self, ft_trans_config):
        mode = ft_trans_config['mode']
        if mode == 0:
            for param in self.Transformation:
                param.requires_grad = False
        else:
            for param in self.Transformation:
                param.requires_grad = True

    @staticmethod
    def _check_whether_update_resnet_grad_feat(self, name_parameter, layer_postion):
        str = "ConvNet.layer3.3.bn2.weight"
        layers = name_parameter.split('.')[1]
        layer = int(re.findall('\d+', layers)[0])
        if layer <= layer_postion:
            return False

    def _set_parameter_requires_grad_feat(self, ft_feat_config):
        mode = ft_feat_config['mode']
        if mode == 1:
            for param in self.FeatureExtraction.parameters():
                param.requires_grad = True
        elif mode == 2:
            layer_position = ft_feat_config['layer_position']
            for name, param in self.FeatureExtraction.named_parameters():
                can_update = self._check_whether_update_resnet_grad_feat(name, layer_position)
                if can_update:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in self.FeatureExtraction:
                param.requires_grad = False

    def _set_parameter_requires_grad_seq(self, ft_seq_config):
        mode = ft_seq_config['mode']
        if mode == 0:
            for param in self.SequenceModeling:
                param.requires_grad = False
        else:
            for param in self.SequenceModeling:
                param.requires_grad = True

    def _set_parameter_requires_grad_pred(self, ft_pred_config):
        mode = ft_pred_config['mode']
        if mode == 0:
            for param in self.Prediction:
                param.requires_grad = False
        else:
            for param in self.Prediction:
                param.requires_grad = True

    def set_parameter_requires_grad_model(self, ft_config):
        if self.stages['Trans'] is not None:
            self._set_parameter_requires_grad_trans(ft_config["trans"])
        self._set_parameter_requires_grad_feat(ft_config["feat"])
        if self.stages['Seq'] is not None:
            self._set_parameter_requires_grad_seq(ft_config["seq"])
        self._set_parameter_requires_grad_pred(ft_config["pred"])





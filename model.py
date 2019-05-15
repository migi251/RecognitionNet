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

import torch
import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.Transformer import Transformer


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(
                opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(
                opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(
                opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        # int(imgH/16-1) * 512
        self.FeatureExtraction_output = opt.output_channel
        if 'Transformer' not in opt.Prediction:  # Transformer use all pixel
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
                (None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output,
                                  opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(
                self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(
                self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        elif opt.Prediction == 'Transformer':
            len_feature = (opt.imgH//16-1)*(opt.imgW//4+1)
            self.src_pos = torch.arange(1, len_feature+1).expand(opt.batch_size, -1).long().cuda()
            self.Prediction = Transformer(
                n_src_vocab=self.SequenceModeling_output,
                n_tgt_vocab=opt.num_class,
                len_max_seq_enc=len_feature,
                len_max_seq_dec=opt.batch_max_length +2 , # +2 for <s> and </s>
                tgt_emb_prj_weight_sharing=opt.proj_share_weight,
                emb_src_tgt_weight_sharing=opt.embs_share_weight,
                d_k=opt.d_k,
                d_v=opt.d_v,
                d_model=self.SequenceModeling_output,
                d_word_vec=self.SequenceModeling_output,
                d_inner=opt.d_inner_hid,
                n_layers_enc=opt.n_layers_enc,
                n_layers_dec=opt.n_layers_dec,
                n_head=opt.n_head,
                dropout=opt.dropout
            )
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input_img, text, is_train=True, tgt_pos=None):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input_img = self.Transformation(input_img)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input_img)
        # print('visual_feature', visual_feature.size(), input_img.size(),text.size(),self.opt.batch_max_length)
        if 'Transformer' not in self.opt.Prediction:
            visual_feature = self.AdaptiveAvgPool(
                visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            # print('visual_feature AdaptiveAvgPool',visual_feature.size())
            visual_feature = visual_feature.squeeze(3)
            # print('squeeze',visual_feature.size())
        else:
            batch_size = visual_feature.size(0)
            visual_feature = visual_feature.permute(0, 2, 3, 1).contiguous() # [b, c, h, w] -> [b, h, w, c]
            visual_feature = visual_feature.view(batch_size, -1, self.FeatureExtraction_output)
            # print('visual_feature Transformer',visual_feature.size(),(self.opt.imgH//16-1)*(self.opt.imgW//4+1))
        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            # for convenience. this is NOT contextually modeled by BiLSTM
            contextual_feature = visual_feature

        # print('forward size',contextual_feature.size(),text.size(),self.opt.batch_max_length)
        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        elif self.stages['Pred'] == 'Transformer':
            src_pos = self.src_pos[:batch_size]
            prediction = self.Prediction(contextual_feature.contiguous(
            ), src_pos, text, tgt_pos, self.opt.batch_max_length, is_train)
        else:
            prediction = self.Prediction(contextual_feature.contiguous(
            ), text, is_train, batch_max_length=self.opt.batch_max_length)
        return prediction

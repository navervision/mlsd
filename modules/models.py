'''
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0
'''
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, ReLU, LeakyReLU
import itertools
import numpy as np

def _regularizer(weights_decay):
    """l2 regularizer"""
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer"""
    return tf.keras.initializers.he_normal()


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                 scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)

        return super().call(x, training)


def Backbone(backbone_type='ResNet50', use_pretrain=True, post_name='_extractor'):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'
    
    def backbone(x):
        if backbone_type.lower() == 'MLSD'.lower():
            extractor = MobileNetV2(
                input_shape=x.shape[1:], include_top=False, weights=weights)
            pick_layers = [27, 54, 90]
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        elif backbone_type.lower() == 'MLSD_large'.lower():
            extractor = MobileNetV2(
                input_shape=x.shape[1:], include_top=False, weights=weights)
            pick_layers = [9, 27, 54, 90, 116]
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        else:
            raise NotImplementedError(
                'Backbone type {} is not supported.'.format(backbone_type))
        
        if int(tf.__version__.split('.')[1]) >= 4:
            output_layers = [extractor.layers[pick_layer - 1].output for pick_layer in pick_layers]
        else:
            output_layers = [extractor.layers[pick_layer].output for pick_layer in pick_layers]
        return Model(extractor.input,
                     output_layers,
                     name=backbone_type + post_name)(preprocess(x))

    return backbone


class Conv_BN_Act(tf.keras.layers.Layer):
    """Conv + BN + Act"""
    def __init__(self, out_ch, k_size, strides, dilate=1, wd=0.0001, act=None):
        super(Conv_BN_Act, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=out_ch, kernel_size=k_size, strides=strides, dilation_rate=dilate, kernel_regularizer=_regularizer(wd),
                                           padding='same', use_bias=False, name='conv')
        self.bn = BatchNormalization(name='bn')
        
        if act is None:
            self.act_fn = tf.identity
        else: # act == 'relu'
            self.act_fn = ReLU()

    def call(self, x):
        return self.act_fn(self.bn(self.conv(x)))


class Upblock(tf.keras.layers.Layer):
    '''
    ref1: https://github.com/Siyuada7/TP-LSD/blob/master/modeling/TP_Net.py#L94-L109
    ref2: https://github.com/Siyuada7/TP-LSD/blob/master/modeling/TP_Net.py#L134-L142
    '''
    def __init__(self, out_ch, squeeze_rate, cfg, last=False):
        super(Upblock, self).__init__()
        self.conv_block1 = Conv_BN_Act(out_ch // squeeze_rate, k_size=cfg.type_a_ksize, strides=1, wd=cfg.wd, act='relu')
        self.conv_block2 = Conv_BN_Act(out_ch // squeeze_rate, k_size=cfg.type_a_ksize, strides=1, wd=cfg.wd, act='relu')
        self.last = last

        if squeeze_rate == 1:
            squeeze_rate = 2
        elif squeeze_rate == 2:
            squeeze_rate = 1

        self.residual_type = cfg.residual_type

        if not self.last:
            if self.residual_type == 0:
                self.conv_block3 = Conv_BN_Act(out_ch * squeeze_rate, k_size=3, strides=1, wd=cfg.wd, act='relu')
                self.conv_block4 = Conv_BN_Act(out_ch, k_size=3, strides=1, wd=cfg.wd, act=None)
            elif self.residual_type == 1:
                self.conv_block3 = Conv_BN_Act(out_ch * squeeze_rate, k_size=3, strides=1, wd=cfg.wd, act=None)
                self.conv_block4 = Conv_BN_Act(out_ch, k_size=3, strides=1, wd=cfg.wd, act=None)
            elif self.residual_type == 2:
                self.conv_block3 = Conv_BN_Act(out_ch, k_size=3, strides=1, wd=cfg.wd, act='relu')
                self.conv_block4 = Conv_BN_Act(out_ch * squeeze_rate, k_size=3, strides=1, wd=cfg.wd, act=None)
            elif self.residual_type == 3:
                self.conv_block3 = Conv_BN_Act(out_ch * squeeze_rate, k_size=3, strides=1, wd=cfg.wd, act='relu')
                self.conv_block4 = Conv_BN_Act(out_ch * squeeze_rate, k_size=3, strides=1, wd=cfg.wd, act=None)
            elif self.residual_type == 4:
                self.conv_block3 = Conv_BN_Act(out_ch * squeeze_rate, k_size=3, strides=1, wd=cfg.wd, act='relu')
                self.conv_block4 = Conv_BN_Act(out_ch * squeeze_rate, k_size=3, strides=1, wd=cfg.wd, act=None)
            elif self.residual_type == 5:
                self.conv_block3 = Conv_BN_Act(out_ch * squeeze_rate, k_size=3, strides=1, wd=cfg.wd, act=None)
                self.conv_block4 = Conv_BN_Act(out_ch * squeeze_rate, k_size=3, strides=1, wd=cfg.wd, act=None)
        self.final_act = ReLU()


    def call(self, x_list, act=True):
        x_large, x_small = x_list

        _, h, w, _ = x_large.shape
        _, h_small, w_small, _ = x_small.shape

        if h_small != h:
            x_small = tf.image.resize(x_small, [h, w], method='bilinear')
        
        x_large = self.conv_block1(x_large)
        x_small = self.conv_block2(x_small)
        
        x = tf.concat([x_large, x_small], axis=-1)

        if not self.last:
            residual = x
            if self.residual_type == 0:
                x = self.conv_block3(x)
                x += residual
                x = self.conv_block4(x)
            elif self.residual_type == 1:
                x = self.conv_block3(x)
                x += residual
                x = self.final_act(x)
                x = self.conv_block4(x)
            elif self.residual_type == 2:
                x = self.conv_block3(x)
                x = self.conv_block4(x)
                x += residual
            elif self.residual_type == 3:
                x = self.conv_block3(x)
                x = self.conv_block4(x)
                x += residual
            elif self.residual_type == 4:
                x = self.conv_block3(x)
                x += residual
                x = self.conv_block4(x)
            elif self.residual_type == 4:
                x = self.conv_block3(x)
                x += residual
                x = self.final_act(x)
                x = self.conv_block4(x)

        if act:
            out = self.final_act(x)
        else:
            out = x

        return out


class Decoder_FPN(tf.keras.layers.Layer):
    def __init__(self, x_list, cfg):
        '''
        x_list = [x0, x1, x2]
        x0: extractor.layers[pick_layer0] [batch, 128, 128, 24]
        x1: extractor.layers[pick_layer1] [batch, 64, 64, 32]
        x2: extractor.layers[pick_layer2] [batch, 32, 32, 64]

        2 > 1 > 0
        '''
        super(Decoder_FPN, self).__init__()
        
        out_ch = 64

        # check squeeze rate
        if cfg.backbone_type.lower() == 'MLSD'.lower():
            squeeze_rates = [1, 2]
        elif cfg.backbone_type.lower() == 'MLSD_large'.lower():
            squeeze_rates = [1, 1, 1, 1]

        self.up_blocks = []
        if cfg.backbone_type.lower() == 'MLSD'.lower():
            # for backward compatibility
            self.up1 = Upblock(out_ch, 1, cfg)
            self.up2 = Upblock(out_ch, 2, cfg)
           
            self.up_blocks.append(self.up1)
            self.up_blocks.append(self.up2)
        else:
            for squeeze_rate in squeeze_rates:
                self.up_blocks.append(Upblock(out_ch, squeeze_rate, cfg))
        
        self.final_act = cfg.final_act

    def call(self, x_list):
        x_list_inv = x_list[::-1]

        x = x_list_inv[0]
        for idx in range(len(x_list) - 1):
            x = self.up_blocks[idx]([x_list_inv[idx+1], x])

        return x
        

class Decoder(tf.keras.layers.Layer):
    """Decoder"""
    def __init__(self, in_ch, out_ch, map_size=112, topk=200, dilate=1, cfg=None, name='Decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.map_size = map_size
        self.decoder0 = Conv_BN_Act(in_ch, k_size=3, strides=1, dilate=dilate, wd=cfg.wd, act='relu')
        self.decoder1 = Conv_BN_Act(in_ch, k_size=3, strides=1, dilate=1, wd=cfg.wd, act='relu')

        if cfg.final_padding_same:
            self.decoder2 = Conv2D(filters=out_ch, kernel_size=1, strides=1, padding='same')
        else:
            self.decoder2 = Conv2D(filters=out_ch, kernel_size=1, strides=1)

        self.topk = topk
        self.batch_size = cfg.batch_size
        self.center_thr = cfg.center_thr

    def get_pts_scores(self, raw_map):
        raw_map_act = tf.math.sigmoid(raw_map)
        max_raw_map_act = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(raw_map_act) # NMS
        raw_map_act = raw_map_act * tf.cast((tf.math.equal(raw_map_act, max_raw_map_act)), tf.float32)
        
        # topk centers
        topk = self.topk
        batch_size, size, _, _, = raw_map_act.shape
        flatten_raw_map_act = tf.reshape(raw_map_act, [-1, size * size])
        topk_scores, topk_indices = tf.math.top_k(flatten_raw_map_act, k=topk)
        y = tf.expand_dims(topk_indices // size, axis=-1)
        x = tf.expand_dims(topk_indices % size, axis=-1)
        topk_pts = tf.concat([y, x], axis=-1)
        
        return topk_pts, topk_scores

    def get_pts_scores_fast(self, raw_map):
        raw_map_act = tf.math.sigmoid(raw_map)
        max_raw_map_act = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(raw_map_act) # NMS
        raw_map_act = raw_map_act * tf.cast((tf.math.equal(raw_map_act, max_raw_map_act)), tf.float32)
        
        raw_map_act = raw_map_act[:,:,:,0]
        indices = tf.where(raw_map_act > self.center_thr)

        scores = tf.gather_nd(raw_map_act, indices)
        center_pts = indices[:,1:] # [y, x]
        
        scores = tf.expand_dims(scores, axis=0)
        center_pts = tf.expand_dims(center_pts, axis=0)

        return center_pts, scores

    def call(self, x):
        x = self.decoder0(x)
        x = self.decoder1(x)
        out_map = self.decoder2(x)
        if out_map.shape[1] != self.map_size:
            out_map = tf.image.resize(out_map, [self.map_size, self.map_size], method='bilinear')
        
        center_map = out_map[:,:,:,0:1]
        line_map = out_map[:,:,:,1:2]
        corner_map = out_map[:,:,:,2:3]
        disp_map = out_map[:,:,:,3:7]
        org_center_map = out_map[:,:,:,7:8]
        org_disp_map = out_map[:,:,:,8:12]
        org_dist_map = out_map[:,:,:,12:13]
        org_deg_map = out_map[:,:,:,13:14]
        split_dist_map = out_map[:,:,:,14:15]
        split_deg_map = out_map[:,:,:,15:16]

        org_dist_map = tf.math.sigmoid(org_dist_map)
        org_deg_map = tf.math.sigmoid(org_deg_map)
        split_dist_map = tf.math.sigmoid(split_dist_map)
        split_deg_map = tf.math.sigmoid(split_deg_map)
        
        disp_map_act = disp_map      

        if self.batch_size > 1 or (self.batch_size == 1 and self.topk is not None):
            org_center_pts, org_center_scores = self.get_pts_scores(org_center_map)
        else:
            org_center_pts, org_center_scores = self.get_pts_scores_fast(org_center_map)

        if self.batch_size > 1 or (self.batch_size == 1 and self.topk is not None):
            center_pts, center_scores = self.get_pts_scores(center_map)
        else:
            center_pts, center_scores = self.get_pts_scores_fast(center_map)
        
        if self.batch_size > 1 or (self.batch_size == 1 and self.topk is not None):
            corner_pts, corner_scores = self.get_pts_scores(corner_map)
        else:
            corner_pts, corner_scores = self.get_pts_scores_fast(corner_map)

        return center_map, disp_map, center_pts, center_scores, disp_map_act, line_map, corner_map, corner_pts, corner_scores, org_center_map, org_disp_map, org_center_pts, org_center_scores, org_dist_map, org_deg_map, split_dist_map, split_deg_map


def WireFrameModel(cfg, training=False, name='WireFrameModel'):
    '''Wireframe Model
    '''
    input_size = cfg.input_size #if training else None
    backbone_type = cfg.backbone_type

    # define model
    # input_size sholud be 512 ?!
    x = inputs = Input([input_size, input_size, 3], name='input_image')
    x = Backbone(backbone_type=backbone_type, post_name=cfg.post_name)(x) # default: use imagenet pretrained
    x = Decoder_FPN(x, cfg)(x) # sigmoid > 1 / softmax > 2

    # out_ch: 16
    center_map, disp_map, center_pts, center_scores, disp_map_act, line_map, corner_map, corner_pts, corner_scores, org_center_map, org_disp_map, org_center_pts, org_center_scores, org_dist_map, org_deg_map, split_dist_map, split_deg_map = Decoder(in_ch=x.shape[-1], out_ch=1 + 1 + 1 + 4 + 1 + 4 + 1 + 1 + 1 + 1, map_size=cfg.map_size, topk=cfg.topk, dilate=cfg.dilate, cfg=cfg)(x)
    
    out = (center_map, disp_map, center_pts, center_scores, disp_map_act, line_map, corner_map, corner_pts, corner_scores, org_center_map, org_disp_map, org_center_pts, org_center_scores, org_dist_map, org_deg_map, split_dist_map, split_deg_map)
    
    return Model(inputs, out, name=name)




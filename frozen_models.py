'''
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0
'''
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import numpy as np
import tensorflow as tf
import time
import tqdm

from modules.models import WireFrameModel

# basic flags 
flags.DEFINE_string('model_path', './pretrained_models/test', 'path to save folder')
flags.DEFINE_string('model_tflite_path', './pretrained_models/conveted_model.tflite', 'path_to_save_tflite_model')
flags.DEFINE_boolean('with_alpha', True, 'whether support RGBA image')
flags.DEFINE_boolean('fp16', False, '')

# input images
flags.DEFINE_integer('batch_size', 128, 'size of input batch')
flags.DEFINE_integer('input_size', 512, 'size of input image')
flags.DEFINE_integer('map_size', 256, 'size of lmap, jmap, and joff')

# encoder
flags.DEFINE_string('backbone_type', 'MLSD_large', 'MLSD | MLSD_large')
flags.DEFINE_boolean('pretrain', True, 'whether use imagenet pretrained weights')
flags.DEFINE_integer('out_channel', 256, 'n_channels of output encoded spatial features')
flags.DEFINE_integer('dilate', 5, 'dilation rate')
flags.DEFINE_boolean('final_last', False, '')
flags.DEFINE_boolean('final_act', True, '')
flags.DEFINE_boolean('final_res1', False, '')
flags.DEFINE_boolean('final_res2', False, '')
flags.DEFINE_integer('residual_type', 0, '')
flags.DEFINE_string('post_name', '_extractor', '_extractor | _extrator')
flags.DEFINE_integer('type_a_ksize', 1, 'type_a_ksize')

# decoder
flags.DEFINE_integer('topk', 200, 'topk')
flags.DEFINE_boolean('final_padding_same', True, '')
flags.DEFINE_float('center_thr', 0.001, 'weight for loss_center_map')

flags.DEFINE_float('wd', 0.0001, 'weight decay value')

def main(_):
    # initialize systems
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    

    cfg = FLAGS # I love FLAGS!!!
    
    # define network
    model = WireFrameModel(cfg, training=False)
    model.summary(line_length=80)

    # load checkpoint
    checkpoint_dir = cfg.model_path
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        print("[*] training from imagenet pretrained weights.")
    

    offset = 0
    if int(tf.__version__.split('.')[1]) >= 4:
        offset = 1

    if cfg.with_alpha:
        '''
        input: RGBA image
        output: [center points, center scores, displacement vector map]
        '''
        new_input_layer = tf.keras.layers.Input([cfg.input_size, cfg.input_size, 4], batch_size=1, name='input_image_with_alpha')
        # preprocess protocol
        output_tensor = new_input_layer
        
        for top_idx, top_layer in enumerate(model.layers):
            print(top_idx, top_layer)
        print('\n\n\n')
        backbone_outputs = []
        for top_idx, top_layer in enumerate(model.layers):
            #print('test', output_tensor)
            print(top_idx, top_layer.name)
            '''
            0 input_image
            1 tf_op_layer_RealDiv
            2 tf_op_layer_Sub
            3 MobileNetV2_extrator
            4 Decoder_FPN
            5 Decoder
            '''
            if top_idx in [0, 1]:
                continue
            if top_idx == 2:
                output_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(output_tensor)
            elif top_idx not in [3, 4]:
                output_tensor = top_layer(output_tensor)
            elif top_idx == 4:
                output_tensor = top_layer(backbone_outputs)
            elif top_idx == 3: # MobileNetV2 backbone
                extractor = top_layer

                # define new_stem_layer
                stem_layer = extractor.layers[2 - offset]
                stem_weights = stem_layer.get_weights()[0]
                zero_weights = tf.zeros([stem_weights.shape[0], stem_weights.shape[1], 1, stem_weights.shape[3]])
                new_weights = tf.concat([stem_weights, zero_weights], axis=2)

                _, input_tensor_size, _, _ = extractor.layers[1 - offset].output.shape
                new_stem_layer = tf.keras.layers.Conv2D(new_weights.shape[-1], new_weights.shape[0], input_shape=(input_tensor_size, input_tensor_size, 4),
                                                        strides=(2, 2),
                                                        padding='same' if int(tf.__version__.split('.')[1]) >= 4 else 'valid',
                                                        kernel_initializer=tf.keras.initializers.Constant(new_weights),
                                                        use_bias=False,
                                                        name='Conv1_with_alpha')

                front_list = []
                block_dict = {}
                add_block_list = []
                block_list = []
                end_list = []
                print('*** MobileNetV2 backbone ***')
                for idx, layer in enumerate(extractor.layers):
                    if idx == 0:
                        print('no need anymore', layer)
                        continue
                    layer_name = layer.name
                    if 'block' in layer_name:
                        block_name = '%s_%s' % (layer_name.split('_')[0], layer_name.split('_')[1])
                        if block_name in block_dict:
                            block_dict[block_name].append(layer)
                        else:
                            block_list.append(block_name)
                            block_dict[block_name] = [layer]
                        if 'add' in layer.name:
                            add_block_list.append(block_name)
                    elif idx < 10 - offset:
                        if idx == 2 - offset:
                            front_list.append(new_stem_layer)
                        else:
                            front_list.append(layer)
                    
                    else:
                        end_list.append(layer)
                
                # output_tensor = preprocessed RGBA input images
                for layer in front_list:
                    output_tensor = layer(output_tensor)

                if 'large' in cfg.backbone_type:
                    backbone_outputs.append(output_tensor)
                
                block_output_tensor = block_input_tensor = output_tensor
                for block_idx, block_name in enumerate(block_list):
                    layer_list = block_dict[block_name]
                    block_input_tensor = block_output_tensor # for add layer
                    for layer in layer_list[:-1]:
                        block_output_tensor = layer(block_output_tensor)

                    if block_name in add_block_list:
                        block_output_tensor = layer_list[-1]([block_input_tensor, block_output_tensor])
                    else:
                        block_output_tensor = layer_list[-1](block_output_tensor)

                    if 'large' in cfg.backbone_type:
                        pooling_block_list = [1, 4, 8, 11]
                    else:
                        pooling_block_list = [1, 4, 8]
                    if block_idx in pooling_block_list:
                        backbone_outputs.append(block_output_tensor)

        new_output_tensor = [output_tensor[-6], output_tensor[-5], output_tensor[-7]]
        new_model = tf.keras.Model(new_input_layer, new_output_tensor, name='WireFrameModel_with_alpha')
        
        # test sample here
        input1 = tf.constant(np.random.rand(3,cfg.input_size,cfg.input_size,3), dtype=tf.float32)
        input2 = tf.concat([input1, tf.ones([3,cfg.input_size,cfg.input_size,1])], axis=-1)
        org_times = []
        for _ in tqdm.tqdm(range(5), desc='Testing'):
            t_start = time.time()
            output1 = model(input1)
            org_times.append(time.time() - t_start)
        alpha_times = []
        for _ in tqdm.tqdm(range(5), desc='Testing'):
            t_start = time.time()
            output2 = new_model(input2)
            alpha_times.append(time.time() - t_start)
        print(1/np.mean(org_times), 1/np.mean(alpha_times))
        output1 = [output1[-6], output1[-5], output1[-7]]
        for val1, val2 in zip(output1, output2):
            print((val1.numpy() == val2.numpy())) # True

        print('\nall values should be True\n')
        print(new_model.input)
        print(new_model.output)
        
        model = new_model
    else:
        '''
        input: RGB image
        output: [center points, center scores, displacement vector map]
        '''
        model = tf.keras.Model(model.input, [model.output[-6], model.output[-5], model.output[-7]], name='WireFrameModel')
        input1 = tf.constant(np.random.rand(3,cfg.input_size,cfg.input_size,3), dtype=tf.float32)
        output1 = model(input1)
    # convert model here
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if cfg.fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(cfg.model_tflite_path, 'wb') as f:
        f.write(tflite_model)
    print('done!')
    #############################################################

if __name__ == '__main__':
    app.run(main)

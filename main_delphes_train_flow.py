import delphes_flows.flows.flow_catalogue as fc
import os
import numpy as np
import tensorflow as tf
import pofah.path_constants.sample_dict_file_parts_reco as sdr
import pofah.util.experiment as expe
import pofah.util.sample_factory as safa
import sarewt.data_reader as dare
import pofah.phase_space.cut_constants as cuts
import delphes_flows.utils.data_handlers as dha
from collections import namedtuple

train_sample='qcdSigMCOrigReco'
valid_sample='qcdSigMCOrigValidReco'
seed=50005

Parameters = namedtuple('Parameters', 'run_n input_shape epochs train_total_n gen_part_n valid_total_n batch_n initializer learning_rate max_lr_decay lambda_reg')
params = Parameters(run_n=seed, 
                    input_shape=(100,3),
                    epochs=100,
                    train_total_n=int(4e6),
                    valid_total_n=int(1e3), 
                    gen_part_n=int(1e6),
                    batch_n=4096,
                    initializer='he_uniform',
                    learning_rate=0.001,
                    max_lr_decay=8, 
                    lambda_reg=0.01,
                    ) # 'L1L2'
experiment = expe.Experiment(run_n=params.run_n).setup(flow_dir=True)
paths = safa.SamplePathDirFactory(sdr.path_dict).update_base_path({'$run$': experiment.run_dir})
print(paths.sample_dir_path(train_sample))
data_train_generator = dha.CMSDataGenerator(path=paths.sample_dir_path(train_sample), sample_part_n=params.gen_part_n, sample_max_n=params.train_total_n) # generate 10 M jet samples
train_ds = tf.data.Dataset.from_generator(data_train_generator, output_signature=(
  tf.TensorSpec(shape=(None, 100,3), dtype=tf.float32, name='orig'),
  tf.TensorSpec(shape=(None, 100,3), dtype=tf.float32, name='reco'))).batch(params.batch_n, drop_remainder=True) # already shuffled

#validation (full tensor, 1M events -> 2M samples)                                                                          
print('>>> Preparing validation dataset')
const_orig_valid, const_reco_valid = dha.CMSDataHandler(path=paths.sample_dir_path(valid_sample)).read_events_from_dir(read_n=params.valid_total_n)
data_valid = dha.events_to_input_samples(const_orig_valid) # We need to use only the "original" jets as validation
import pdb;pdb.set_trace()

valid_ds = tf.data.Dataset.from_tensor_slices(data_valid).batch(params.batch_n, drop_remainder=True)

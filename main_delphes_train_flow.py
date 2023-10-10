import delphes_flows.flows.flow_catalogue as fc
import os,sys
import numpy as np
import tensorflow as tf
import pofah.path_constants.sample_dict_file_parts_reco as sdr
import pofah.util.experiment as expe
import pofah.util.sample_factory as safa
import sarewt.data_reader as dare
import pofah.phase_space.cut_constants as cuts
import delphes_flows.utils.data_handlers as dha
import flows.flow_catalogue as fc
import utils.trainer as tra
import get_free_gpu_id as gpu
from collections import namedtuple

if gpu.fetch_id()<0:
    print("No GPU devices with sufficient memory. Try again when its free")
    sys.exit(0)
free_gpu=gpu.fetch_id()
os.environ["CUDA_VISIBLE_DEVICES"]=str(free_gpu)


print(f'Running on GPU ID: {free_gpu}')
train_sample='qcdSigMCOrigReco'
valid_sample='qcdSigMCOrigValidReco'
seed=50005

Parameters = namedtuple('Parameters', 'run_n n_bijectors hidden_units input_shape epochs train_total_n gen_part_n valid_total_n batch_n learning_rate min_delta max_lr_decay reg_factor weight_decay clip')
params = Parameters(run_n=seed, 
                    input_shape=(100,3),
                    epochs=100,
                    n_bijectors=3,
                    hidden_units=[32,16,8],
                    train_total_n=int(5e5),
                    valid_total_n=int(1e4), 
                    gen_part_n=int(5e5),
                    batch_n=1000,
                    learning_rate=0.0005,  
                    min_delta=0.03,
                    max_lr_decay=4, 
                    reg_factor=1.0e-4,
                    weight_decay=0.005,
                    clip=0.75
                    ) # 'L1L2'
experiment = expe.Experiment(run_n=params.run_n).setup(flow_dir=True)
paths = safa.SamplePathDirFactory(sdr.path_dict).update_base_path({'$run$': experiment.run_dir})
print(paths.sample_dir_path(train_sample))
data_train_generator = dha.CMSDataGenerator(path=paths.sample_dir_path(train_sample), sample_part_n=params.train_total_n, sample_max_n=params.train_total_n) # generate 10 M jet samples
train_ds = tf.data.Dataset.from_generator(data_train_generator, output_signature=(
  tf.TensorSpec(shape=(100,3), dtype=tf.float32, name='orig'),
  tf.TensorSpec(shape=(100,3), dtype=tf.float32, name='reco'))).batch(params.batch_n, drop_remainder=True) # already shuffled

#validation (full tensor, 1M events -> 2M samples)                                                                          
print('>>> Preparing validation dataset')
const_orig_valid, const_reco_valid = dha.CMSDataHandler(path=paths.sample_dir_path(valid_sample)).read_events_from_dir(read_n=params.valid_total_n)
data_orig_valid,data_reco_valid = dha.events_to_orig_reco_samples(const_orig_valid,const_reco_valid) # We need to use only the "original" jets as validation
valid_ds = tf.data.Dataset.from_tensor_slices((data_orig_valid,data_reco_valid)).batch(params.valid_total_n//10, drop_remainder=True)

optimizer1 = tf.keras.optimizers.AdamW(learning_rate=params.learning_rate,weight_decay=params.weight_decay)
optimizer2 = tf.keras.optimizers.AdamW(learning_rate=params.learning_rate,weight_decay=params.weight_decay)

jet_flow_orig=fc.JetMAF(hidden_units=params.hidden_units,n_bijectors=params.n_bijectors)
jet_flow_reco=fc.JetMAF(hidden_units=params.hidden_units,n_bijectors=params.n_bijectors)
#et_flow_orig.load_from_checkpoint(os.path.join(experiment.flow_dir,'orig'),optimizer1)
#jet_flow_reco.load_from_checkpoint(os.path.join(experiment.flow_dir,'reco'),optimizer2)
jet_flow_orig.build((None,300))
jet_flow_reco.build((None,300))

jtra=tra.JointTrainer(optimizer1,optimizer2,patience=3, min_delta=params.min_delta, max_lr_decay=params.max_lr_decay,datalength=params.train_total_n, batchsize=params.batch_n, lr_decay_factor=0.3)
losses_train,losses_valid=jtra.train(experiment.flow_dir,jet_flow_orig,jet_flow_reco,train_ds,valid_ds,params.epochs,params.reg_factor)



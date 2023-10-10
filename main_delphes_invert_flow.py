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
from utils.plotters import plot
import get_free_gpu_id as gpu
import flows.flow_manager as fm
from collections import namedtuple
import matplotlib.pyplot as plt


gpu.set_gpu()

train_sample='qcdSigMCOrigReco'
valid_sample='qcdSigMCOrigValidReco'
seed=50005

Parameters = namedtuple('Parameters', 'run_n n_bijectors hidden_units input_shape valid_total_n batch_n learning_rate min_delta max_lr_decay reg_factor weight_decay clip')
params = Parameters(run_n=seed, 
                    input_shape=(100,3),
                    n_bijectors=3,
                    hidden_units=[32,16,8],
                    valid_total_n=int(1e6), 
                    batch_n=1000,
                    learning_rate=0.001,  
                    min_delta=0.03,
                    max_lr_decay=4, 
                    reg_factor=1.0e-5,
                    weight_decay=0.005,
                    clip=0.75
                    ) # 'L1L2'
experiment = expe.Experiment(run_n=params.run_n).setup(flow_dir=True)
paths = safa.SamplePathDirFactory(sdr.path_dict).update_base_path({'$run$': experiment.run_dir})
print(paths.sample_dir_path(train_sample))
 
#validation (full tensor, 1M events -> 2M samples)                                                                          
print('>>> Preparing validation dataset')
const_orig_valid, const_reco_valid = dha.CMSDataHandler(path=paths.sample_dir_path(valid_sample)).read_events_from_dir(read_n=params.valid_total_n)
orig_jets,reco_jets=dha.events_to_orig_reco_samples(const_orig_valid,const_reco_valid)
#orig_jets=dha.convert_ptetaphi_to_pxpypz(orig_jets)
#reco_jets=dha.convert_ptetaphi_to_pxpypz(reco_jets)

optimizer1 = tf.keras.optimizers.AdamW(learning_rate=params.learning_rate,weight_decay=params.weight_decay)
optimizer2 = tf.keras.optimizers.AdamW(learning_rate=params.learning_rate,weight_decay=params.weight_decay)

predict_ds=tf.data.Dataset.from_tensor_slices((orig_jets,reco_jets)).batch(params.batch_n)
fManager=fm.FlowManager(os.path.join(experiment.flow_dir,'orig'),os.path.join(experiment.flow_dir,'reco'),optimizer1,optimizer2,params)
orig_jet_array,reco_jet_array,flow_jet_array=fManager.reco_to_orig(predict_ds)

flow_jet_samples=dha.samples_to_events(flow_jet_array)
orig_jet_samples=dha.samples_to_events(orig_jet_array)
reco_jet_samples=dha.samples_to_events(reco_jet_array)

import pdb;pdb.set_trace()

plot(orig_jet_samples,flow_jet_samples,variableid=0)
plot(orig_jet_samples,flow_jet_samples,variableid=1)
plot(orig_jet_samples,flow_jet_samples,variableid=2)
import tensorflow as tf
import tensorflow_probability as tfp
import delphes_flows.flows.flow_catalogue as fc
import numpy as np
class FlowManager():
    # Class to manage two flows at the same time. Implements the orig to reco, and reco to orig methods for converting between the two
    def __init__(self,path_to_orig,path_to_reco,optimizer_orig,optimizer_reco,params):
        self.model_orig=fc.JetMAF(hidden_units=params.hidden_units,n_bijectors=params.n_bijectors)
        self.model_reco=fc.JetMAF(hidden_units=params.hidden_units,n_bijectors=params.n_bijectors)
        self.optimizer_orig=optimizer_orig
        self.optimizer_reco=optimizer_reco
        self.model_orig.load_from_checkpoint(path_to_orig,self.optimizer_orig,'best_so_far_orig')
        self.model_reco.load_from_checkpoint(path_to_reco,self.optimizer_reco,'best_so_far_reco')
        #self.model_orig.build((None,300))
        #self.model_reco.build((None,300))

    def reco_to_orig(self,predict_ds):
        orig=[]
        reco=[]
        flow=[]
        for step,(orig_jets,reco_jets) in enumerate(predict_ds):
            #import pdb;pdb.set_trace()
            latent_reco=self.model_reco.invert(tf.reshape(reco_jets,[-1,300]))
            flow_jets=tf.reshape(self.model_orig(latent_reco),[-1,100,3])
            orig.append(orig_jets)
            reco.append(reco_jets)
            flow.append(flow_jets)
            print(f"At step {step:03d}",end='\r',flush=True)
        print(f'Inverted {step} batches for dataset')
        return (np.concatenate(orig_jets,axis=0)),(np.concatenate(reco_jets,axis=0)),(np.concatenate(flow_jets,axis=0))
    
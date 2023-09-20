import tensorflow as tf
import delphes_flows.flows.flow_base as fb
from typing import Sequence
from delphes_flows.flows.flow_layers import MADE
import tensorflow_probability as tfp
from scipy.stats import wasserstein_distance

tfk=tf.keras
tfb=tfp.bijectors
tfd=tfp.distributions

class MAF(fb.NormalizingFlow):
    def __init__(self,output_dim,n_bijectors,base_distribution,flow_layers:Sequence,**kwargs):
        self.n_bijectors=n_bijectors
        self.output_dim=output_dim
        bijectors=[]
        for i in range(self.n_bijectors):
            made_layer=MADE(event_shape=self.output_dim)
            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made_layer))
            bijectors.append(tfb.Permute(permutation=[1,0]))
        bijector = tfb.Chain(list(reversed(bijectors)))
        super(MAF,self).__init__(base_distribution=base_distribution,flow_layers=bijector,**kwargs)
        # Calling the base class will initialize the normalizing flow which can be accessed as self.flow
        # __call__ is implemented in the base class as well

class JetMAF(fb.NormalizingFlow):
    pass
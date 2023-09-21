import tensorflow as tf
import delphes_flows.flows.flow_base as fb
from typing import Sequence
from delphes_flows.flows.flow_layers import MADE
import tensorflow_probability as tfp
import numpy as np
tfk=tf.keras
tfb=tfp.bijectors
tfd=tfp.distributions

class MAF(fb.NormalizingFlow):
    def __init__(self,event_dim,n_bijectors,base_distribution,flow_layers:Sequence,**kwargs):
        self.n_bijectors=n_bijectors
        self.event_dim=event_dim
        bijectors=[]
        for i in range(self.n_bijectors):
            made_layer=MADE(event_shape=self.event_dim)
            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made_layer))
            bijectors.append(tfb.Permute(permutation=range(self.event_dim)))
        bijector = tfb.Chain(list(reversed(bijectors)))
        super(MAF,self).__init__(base_distribution=base_distribution,flow_layers=bijector,**kwargs)
        # Calling the base class will initialize the normalizing flow which can be accessed as self.flow
        # __call__ is implemented in the base class as well

class JetMAF(fb.JetFlow):
    def __init__(self,event_dim,n_bijectors,base_distribution,target_distribution,flow_layers:Sequence,**kwargs):
        self.n_bijectors=n_bijectors
        self.event_dim=tf.convert_to_tensor(event_dim)
        self.flattened_dim=tf.convert_to_tensor(np.prod(event_dim))
        bijectors=[]
        made_layer=MADE(event_shape=self.flattened_dim)
        bijectors.append(tfb.Reshape(self.event_dim,self.flattened_dim)) # Reshape from 300 to 100x3
        for i in range(self.n_bijectors):
            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made_layer))
            bijectors.append(tfb.Permute(permutation=range(np.prod(event_dim))))
        bijectors.append(tfb.Reshape(self.flattened_dim,self.event_dim)) # Reshape from 100x3 to 300
        # Order of the reshape bijectors appears to be reversed here because the list of bijectors is reversed in the base class when chaining together
        super(MAF,self).__init__(base_distribution=base_distribution,target_distribution=target_distribution,flow_layers=bijectors,**kwargs)
        # Calling the base class will initialize the normalizing flow which can be accessed as self.flow
        # __call__ is implemented in the base class as well
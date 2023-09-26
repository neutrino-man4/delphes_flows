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
    def __init__(self,event_dim=[100,3],n_bijectors=3,base_distribution=None,flow_layers:Sequence=None,**kwargs):
        self.n_bijectors=n_bijectors
        self.event_dim=event_dim
        if base_distribution is None:
            self.base_distribution=tfd.MultivariateNormalDiag(loc=tf.zeroes(np.prod(event_dim)))
        else:
            self.base_distribution=base_distribution
        bijectors=[]
        for i in range(self.n_bijectors):
            made_layer=MADE(event_shape=self.event_dim)
            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made_layer))
            bijectors.append(tfb.Permute(permutation=range(self.event_dim)))
        bijector = tfb.Chain(list(reversed(bijectors)))
        super(MAF,self).__init__(base_distribution=self.base_distribution,flow_layers=bijector,**kwargs)
        # Calling the base class will initialize the normalizing flow which can be accessed as self.flow
        # __call__ is implemented in the base class as well

class JetMAF(fb.JetFlow):
    def __init__(self,event_dim=[100,3],n_bijectors=3,base_distribution=None,**kwargs):
        super().__init__(**kwargs)
        
        self.flattened_dim=np.prod(event_dim)
        self.n_bijectors=n_bijectors
        self.bijector_fns=[]
        if base_distribution is None:
            self.base_distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(self.flattened_dim))
        else:
            self.base_distribution=base_distribution        
        bijectors=[]
        perm=np.arange(self.flattened_dim)
        np.random.seed(1);np.random.shuffle(perm)
        #bijectors.append(tfb.Reshape(self.event_dim,self.flattened_dim)) # Reshape from 300 to 100x3
        for i in range(self.n_bijectors):
            made_layer=tfb.AutoregressiveNetwork(params=2, event_shape=[300], hidden_units=[128,64,32],
                                                 activation='relu', use_bias=True, kernel_initializer='he_uniform')
            self.bijector_fns.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made_layer))
            
            bijectors.append(self.bijector_fns[-1])
            bijectors.append(tfb.Permute(permutation=perm))
        bijectors=bijectors[:-1]+[tfb.BatchNormalization()]
        #bijectors.append(tfb.Reshape(self.flattened_dim,self.event_dim)) # Reshape from 100x3 to 300
        # Order of the reshape bijectors appears to be reversed here because the list of bijectors is reversed in the base class when chaining together
        
        self.bijector=tfb.Chain(list(reversed(bijectors)))
        self.flow=tfd.TransformedDistribution(distribution=self.base_distribution,bijector=self.bijector)
        
        # Calling the base class will initialize the normalizing flow which can be accessed as self.flow
        # __call__ is implemented in the base class as well
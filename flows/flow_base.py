import tensorflow as tf
import tensorflow_probability as tfp
from typing import Sequence
import os
tfd=tfp.distributions
tfb=tfp.bijectors

class NormalizingFlow(tf.keras.models.Model):

    def __init__(self,base_distribution,flow_layers:Sequence,**kwargs) -> None:
        self.base_distribution=base_distribution
        self.flow_layers=flow_layers
        bijector=tfb.Chain(list(reversed(self.flow_layers)))
        self.flow=tfd.TransformedDistribution(distribution=self.base_distribution,bijector=bijector)
        super(NormalizingFlow,self).__init__(**kwargs)

    def __call__(self,input):
        return self.flow.bijector.forward(input)
    
    @tf.function
    def forward_step(self,X,optimizer,training=True):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.flow.log_prob(X, training=True))
            if training:
                gradients = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
class JetFlow(tf.keras.models.Model):
    '''
    Flow: transforms reco jet into original jet
    Base distribution: kinematics of reco jet
    target distribution: kinematics of orig. jet
    '''
    def __init__(self,base_distribution,flow_layers:Sequence,**kwargs) -> None:
        self.base_distribution=base_distribution
        self.flow_layers=flow_layers
        bijector=tfb.Chain(list(reversed(self.flow_layers)))
        self.flow=tfd.TransformedDistribution(distribution=self.base_distribution,bijector=bijector)
        super(JetFlow,self).__init__(**kwargs)

    def __call__(self,input):
        return self.flow.bijector.forward(input)

    @tf.function
    def forward_step(self, X_batch,Y_batch,optimizer,loss_fn,training=True):
        '''
        For training,
        X_batch: reco jets
        Y_batch: orig jets
        orig_flow: "original" jet as reconstructed by the flow
        loss function - threeD loss (as used by VAE) computed between orig_flow and Y_batch (actual original jets)
        '''
        with tf.GradientTape() as tape:
            tape.watch(self.flow.trainable_variables)
            orig_flow=self.flow.sample(X_batch) # Shape: batch size x 100 x 3
            loss=loss_fn(Y_batch,orig_flow) # Shape: batch size x 100 x 3 to be compatible with threeD loss
            if training:
                gradients = tape.gradient(loss, self.flow.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
        return loss
    
    def save_model(self,path,fname='jetMAF.tf'):
        model_name=os.path.join(path,fname)
        self.save(model_name,save_format='tf')
    
    def load_model(self,path,custom_objects={},fname='jetMAF.tf'):
        model_name=os.path.join(path,fname)
        return tf.keras.saving.load_model(model_name,custom_objects=custom_objects)


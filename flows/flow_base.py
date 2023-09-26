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
        super().__init__(**kwargs)

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
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        self.flow=None
        
    def __call__(self,*input):
        return self.flow.bijector.forward(*input)

    @tf.function
    def forward_step(self, X_batch,optimizer,training=True):
        '''
        For training,
        X_batch: reco/orig jets
        Ignore: This is wrong implementation
        '''
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss=-tf.reduce_mean(self.flow.log_prob(X_batch,training=training))
        if training:
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
    def nll_loss(self,X_batch,training=False):
        return tf.math.abs(-tf.reduce_mean(self.flow.log_prob(X_batch,training=training)))
    
    def invert(self,X_batch):
        return self.flow.bijector.inverse(X_batch)
    
    def save_model(self,path,fname='jetMAF.tf'):
        model_name=os.path.join(path,fname)
        self.save(model_name,save_format='tf')
    
    def load_model(self,path,custom_objects={},fname='jetMAF.tf'):
        model_name=os.path.join(path,fname)
        return tf.keras.saving.load_model(model_name,custom_objects=custom_objects)


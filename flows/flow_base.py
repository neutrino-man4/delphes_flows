import tensorflow as tf
import tensorflow_probability as tfp
from typing import Sequence
import os
import pathlib

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
    
class JetFlow(tf.keras.Model):
    '''
    Flow: transforms reco jet into original jet
    Base distribution: kinematics of reco jet
    target distribution: kinematics of orig. jet
    '''
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        self.flow=None
        
        
    def call(self,x):
        return self.flow.bijector.forward(x)

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
        log_prob=self.flow.log_prob(X_batch,training=training)
        log_prob=tf.where(tf.math.is_finite(log_prob),log_prob,tf.zeros_like(log_prob))
        log_prob=tfp.math.clip_by_value_preserve_gradient(log_prob,-1.0e6,1.0e6)
        return (-tf.reduce_mean(log_prob))
    
    def invert(self,X_batch):
        return self.flow.bijector.inverse(X_batch)
    
    
    def save_checkpoint(self,path,optimizer,fname='best_so_far.ckpt'):
        pathlib.Path(path).mkdir(exist_ok=True,parents=True)
        ckpt=tf.train.Checkpoint(model=self,optimizer=optimizer)
        ckpt.save(os.path.join(path,fname))
        
    def load_from_checkpoint(self,path,optimizer,fname='best_so_far.ckpt'):
        ckpt=tf.train.Checkpoint(model=self,optimizer=optimizer)
        ckpt.restore(tf.train.latest_checkpoint(path))
        print(f'Restored checkpoints from {path}')

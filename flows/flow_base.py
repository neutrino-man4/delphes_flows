import tensorflow as tf
import tensorflow_probability as tfp
from typing import Sequence
tfd=tfp.distributions
tfb=tfp.bijectors

class NormalizingFlow(tf.keras.models.Model):

    def __init__(self,base_distribution,flow_layers:Sequence,**kwargs) -> None:
        self.base_distribution=base_distribution
        self.flow_layers=flow_layers
        bijector=tfb.Chain(list(reversed(self.flow_layers)))
        self.flow=tfd.TransformedDistribution(distribution=self.base_distribution,bijector=bijector)
        super(NormalizingFlow,self).__init__(**kwargs)

    def __call__(self,*inputs):
        return self.flow.bijector.forward(*inputs)
    
    @tf.function
    def train_step(self,X,optimizer):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.flow.log_prob(X, training=True))
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
class JetFlow(tf.keras.models.Model):

    def __init__(self,base_distribution,target_distribution,flow_layers:Sequence,**kwargs) -> None:
        self.base_distribution=base_distribution
        self.target_distribution=target_distribution
        self.flow_layers=flow_layers
        bijector=tfb.Chain(list(reversed(self.flow_layers)))
        self.flow=tfd.TransformedDistribution(distribution=self.base_distribution,bijector=bijector)
        super(NormalizingFlow,self).__init__(**kwargs)

    def __call__(self,*inputs):
        return self.flow.bijector.forward(*inputs)
    
    @tf.function
    def train_step(self,X,optimizer):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.flow.log_prob(X, training=True))
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
import tensorflow as tf
import tensorflow_probability as tfp
from vande.training import Stopper
import numpy as np
from vande.vae.losses import threeD_loss

tfd=tfp.distributions
tfb=tfp.bijectors
tfk=tf.keras
class Stopper():
    def __init__(self, optimizer1,optimizer2, min_delta=0.01, patience=4, max_lr_decay=10, lr_decay_factor=0.3):
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.min_delta = min_delta
        self.patience = patience
        self.patience_curr = 0
        self.max_lr_decay = max_lr_decay
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_n = 0

    def callback_early_stopping(self, loss_list, min_delta=0.1, patience=4):
        # increase loss-window width at each epoch
        self.patience_curr += 1
        if self.patience_curr < patience:
            return False
        # compute difference of the last #patience epoch losses
        mean = np.mean(loss_list[-patience:])
        deltas = np.absolute(np.diff(loss_list[-patience:])) 
        # return true if all relative deltas are smaller than min_delta
        return np.all((deltas / mean) < min_delta)

    def check_stop_training(self, losses):
        if self.callback_early_stopping(losses, min_delta=self.min_delta, patience=self.patience):
            print('-'*7 + ' Early stopping for last '+ str(self.patience)+ ' validation losses ' + str([l.numpy() for l in losses[-self.patience:]]) + '-'*7)
            if self.lr_decay_n >= self.max_lr_decay:
                # stop the training
                return True
            else: 
                # decrease the learning rate
                curr_lr1 = self.optimizer1.learning_rate.numpy()
                curr_lr2 = self.optimizer2.learning_rate.numpy()
                self.optimizer1.learning_rate.assign(curr_lr1 * self.lr_decay_factor)
                self.optimizer2.learning_rate.assign(curr_lr2 * self.lr_decay_factor)
                self.lr_decay_n += 1
                # reset patience window each time lr has been decreased
                self.patience_curr = 0
                print('decreasing learning rate from {:.3e} to {:.3e} for both optimizers'.format(curr_lr1, self.optimizer1.learning_rate.numpy()))
        return False
class Trainer():
    def __init__(self,optimizer,patience,min_delta,max_lr_decay,datalength,batchsize,lr_decay_factor):
        self.optimizer = optimizer
        self.patience = patience
        self.min_delta = min_delta
        self.max_lr_decay = max_lr_decay
        self.datalength = datalength
        self.batchsize = batchsize
        self.lr_decay_factor = lr_decay_factor
        self.train_stop = Stopper(optimizer, min_delta, patience, max_lr_decay,lr_decay_factor)
        self.best_loss_so_far = None    
    
    def training_epoch(self,model,train_ds,loss_fn):
        train_loss=0
        for step,(orig_batch,reco_batch) in enumerate(train_ds):
            loss=model.forward_step(reco_batch,orig_batch,self.optimizer,loss_fn,training=True) # model is propagated forward on reco batch
            train_loss+=loss
        return train_loss/(step+1)
    
    def valid_epoch(self,model,X_valid,Y_valid,loss_fn):
        valid_loss=0
        for step,X_batch,Y_batch in enumerate(zip(X_valid,Y_valid)):
            loss=model.forward_step(X_batch,Y_batch,self.optimizer,loss_fn,training=False)
            valid_loss+=loss
        return valid_loss/(step+1)    
    
    def train(self,model,train_ds,valid_ds,loss_fn,epochs,training=True):
        '''
        train_ds: should contain batches of both orig and reco jets in that order
        valid_ds: should contain two datasets: orig and reco, in that order
        '''
        losses_train=[]
        losses_valid=[]
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            train_loss=self.training_epoch(model,train_ds,loss_fn)
            valid_loss=self.valid_epoch(model,valid_ds,loss_fn)
            losses_train.append(train_loss)
            losses_valid.append(valid_loss)
            print(f'Average train loss: {train_loss} and average validation loss: {valid_loss}')
            if self.train_stop.check_stop_training(losses_valid):
                print('!!! stopping training !!!')
                break
                
        return losses_train,losses_valid


class JointTrainer():
    def __init__(self,optimizer_orig,optimizer_reco,patience,min_delta,max_lr_decay,datalength,batchsize,lr_decay_factor):
        self.event_dim=[-1,100,3]
        self.flat_dim=[-1,300]
        self.optimizer_reco=optimizer_reco
        self.optimizer_orig=optimizer_orig
        self.patience = patience
        self.min_delta = min_delta
        self.max_lr_decay = max_lr_decay
        self.datalength = datalength
        self.batchsize = batchsize
        self.lr_decay_factor = lr_decay_factor
        self.best_loss_so_far = None    
        self.train_stop = Stopper(optimizer_orig,optimizer_reco, min_delta, patience, max_lr_decay,lr_decay_factor)
    
    @tf.function
    def training_step(self,model_orig,model_reco,orig_batch,reco_batch,reg_factor=0.01):
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([model_orig.trainable_variables,model_reco.trainable_variables])
            #threedim_loss=threeD_loss(tf.reshape(orig_batch,[-1,100,3]),tf.reshape(self.reco_to_orig(reco_batch),[-1,100,3]))
            orig_flow_loss=model_orig.nll_loss(orig_batch,training=True)
            reco_flow_loss=model_reco.nll_loss(reco_batch,training=True)
            latent_loss=tf.reduce_mean(tf.math.squared_difference(model_reco.invert(reco_batch),model_orig.invert(orig_batch)))
            loss_orig=reg_factor*orig_flow_loss+latent_loss
            loss_reco=reg_factor*reco_flow_loss+latent_loss
        orig_gradients=tape.gradient(loss_orig,model_orig.trainable_variables)
        reco_gradients=tape.gradient(loss_reco,model_reco.trainable_variables)
        self.optimizer_orig.apply_gradients(zip(orig_gradients, model_orig.trainable_variables))
        self.optimizer_reco.apply_gradients(zip(reco_gradients, model_reco.trainable_variables))
        return loss_orig+loss_reco,reg_factor*(orig_flow_loss+reco_flow_loss),latent_loss
    
    def training_epoch(self,model_orig,model_reco,train_ds,reg_factor=0.01,epoch=0):
        train_loss=0.
        train_flow_loss=0.
        train_latent_loss=0.
        for step,(orig,reco) in enumerate(train_ds):
            
            orig_batch=tf.reshape(orig,self.flat_dim)
            reco_batch=tf.reshape(reco,self.flat_dim)
            loss,flow_loss,latent_loss=self.training_step(model_orig,model_reco,orig_batch,reco_batch,reg_factor=reg_factor)
            train_loss+=loss
            train_flow_loss+=flow_loss
            train_latent_loss+=latent_loss
            if step%100==0:
                print(f'At step: {step+1}, total loss = {(train_flow_loss/(step+1)):0.03f} flow + {(train_latent_loss/(step+1)):0.03f} latent')
        #import pdb;pdb.set_trace()
        return train_loss/(step+1),train_flow_loss/(step+1),train_latent_loss/(step+1)
    
    def reco_to_orig(self,model_orig,model_reco,X_batch):
        latent_reco=model_reco.invert(X_batch)
        X_batch_orig=model_orig.flow.bijector.forward(latent_reco)
        return X_batch_orig
    
    def orig_to_reco(self,model_orig,model_reco,X_batch):
        latent_orig=model_orig.invert(X_batch)
        X_batch_reco=model_reco.flow.bijector.forward(latent_orig)
        return X_batch_reco
    
    def valid_epoch(self,model_orig,model_reco,valid_ds,reg_factor=0.01,epoch=0):
        valid_loss=0
        valid_flow_loss=0
        valid_latent_loss=0
        for step,(orig,reco) in enumerate(valid_ds):
            orig_batch=tf.reshape(orig,self.flat_dim)
            reco_batch=tf.reshape(reco,self.flat_dim)
            loss_reco=model_reco.nll_loss(reco_batch,training=False) # model is propagated forward on reco batch
            loss_orig=model_orig.nll_loss(orig_batch,training=False) # model is propagated forward on reco batch
            latent_diff=tf.math.squared_difference(model_reco.invert(reco_batch),model_orig.invert(orig_batch))
            latent_loss=tf.reduce_mean(latent_diff)
            flow_loss=reg_factor*(loss_reco+loss_orig)
            loss=flow_loss+2*latent_loss
            valid_loss+=loss
            valid_flow_loss+=flow_loss
            valid_latent_loss+=2*latent_loss
            print(f'At step: {step+1}, total loss = {(valid_flow_loss/(step+1)):0.03f} flow + {(valid_latent_loss/(step+1)):0.03f} latent')
        #if epoch>15:
        #import pdb;pdb.set_trace()
        
        return valid_loss/(step+1),valid_flow_loss/(step+1),valid_latent_loss/(step+1)
    
    def train(self,model_orig,model_reco,train_ds,valid_ds,epochs,reg_factor=0.01):
        '''
        train_ds: should contain batches of both orig and reco jets in that order
        valid_ds: should contain two datasets: orig and reco, in that order
        '''
        losses_train=[]
        losses_valid=[]
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            train_loss,train_flow_loss,train_latent_loss=self.training_epoch(model_orig,model_reco,train_ds,reg_factor=reg_factor,epoch=epoch)
            valid_loss,valid_flow_loss,valid_latent_loss=self.valid_epoch(model_orig,model_reco,valid_ds,reg_factor=reg_factor,epoch=epoch)
            losses_train.append(train_loss)
            losses_valid.append(valid_loss)
            print(f'Average train loss: {train_loss:0.03f} and average validation loss: {valid_loss:0.03f}')
            print(f'Average train flow loss: {train_flow_loss} and average validation flow loss: {valid_flow_loss:0.03f}')
            print(f'Average train latent loss: {train_latent_loss:0.03f} and average validation latent loss: {valid_latent_loss:0.03f}')
            if self.train_stop.check_stop_training(losses_valid):
                print('!!! stopping training !!!')
                break
            
        return losses_train,losses_valid

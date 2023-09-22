import tensorflow as tf
import tensorflow_probability as tfp
from vande.training import Stopper

tfd=tfp.distributions
tfb=tfp.bijectors
tfk=tf.keras

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
            loss=model.forward_step(X_batch,Y_batch,self.optimizer,loss_fn,training=True)
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
            valid_loss=self.valid_epoch(model,X_valid,Y_valid,loss_fn)
            losses_train.append(train_loss)
            losses_valid.append(valid_loss)
            print(f'Average train loss: {train_loss} and average validation loss: {valid_loss}')
            if self.train_stop.check_stop_training(losses_valid):
                print('!!! stopping training !!!')
                break
                
        return losses_train,losses_valid

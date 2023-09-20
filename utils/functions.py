import tensorflow_probability as tfp

tfb=tfp.bijectors
tfpl=tfp.layers
tfd=tfp.bijectors

hidden_units=[5,5,5]
activation='elu'

made = tfb.AutoregressiveNetwork(params=2, event_shape=[2], hidden_units=hidden_units, activation=activation)

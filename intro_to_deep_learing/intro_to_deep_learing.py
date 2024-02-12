# Neuron - y = w*x+b
# For the input x, what reaches the neuron is w * x. A neural network "learns" by modifying its weights.
# The b is a special kind of weight we call the bias. The bias doesn't have any input data associated with it;
# The bias enables the neuron to modify the output independently of its inputs.


# # Create a network with 1 linear unit; unit = output
# model = keras.Sequential([
#     layers.Dense(units=1, input_shape=[3])
# ])
# To compile a model
# model.compile(
#     optimizer="adam",
#     loss="mae",
# )
# To fit a model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_valid, y_valid),
#     batch_size=256,
#     epochs=10,
# )

# Layer - typically organize their neurons into layers. 
# Dense layer - connected linear units that are having a common set of inputs.

# An activation function - some function we apply to each of a layer's outputs (its activations). 
# The most common is the rectifier function  max(0,x) so max(0, w * x + b) called ReLU (rectified linear unit)
# Hidden layers - layers before the output

# loss function -  measures how good the network's predictions are commonly mean absolute error (MAE).
# optimizer - tells the network how to change its weights commonly stochastic gradient descent (SGD)

# Batch - one iteration's sample of training data
# Epoch - complete round of training data = multiple batches
# learning rate - "Tthe shift" of every batch. A smaller learning rate means the network needs to see more batches before its weights converge to their best values.
# Underfitting the training set - when the loss is not as low as it could be because the model hasn't learned enough signal. 
# Overfitting the training set - when the loss is not as low as it could be because the model learned too much noise.
# the validation loss begins to rise very early, while the training loss continues to decrease = overfitting
# A model's capacity refers to the size and complexity of the patterns it is able to learn. 
# For neural networks, this will largely be determined by how many neurons it has and how they are connected together. 
# If it appears that your network is underfitting the data, you should try increasing its capacity.

# model = keras.Sequential([
#     layers.Dense(16, activation='relu'),
#     layers.Dense(1),
# ])

# wider = keras.Sequential([
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1),
# ])

# deeper = keras.Sequential([
#     layers.Dense(16, activation='relu'),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(1),
# ])

# from tensorflow.keras import callbacks

# early_stopping = callbacks.EarlyStopping(
#     min_delta=0.001, # minimium amount of change to count as an improvement
#     patience=5, # how many epochs to wait before stopping
#     restore_best_weights=True,
# )

# dropout laye - randomly drop out some fraction of a layer's input units every step of training, making it much harder for the network to learn those spurious patterns in the training data
# batch normalization - A batch normalization layer looks at each batch as it comes in, first normalizing the batch with its own mean and standard deviation, 
# and then also putting the data on a new scale with two trainable rescaling parameters
# keras.Sequential([
#     # ...
#     layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
#     layers.Dense(16),
#     # ...
# ])
# layers.Dense(16, activation='relu'),
# layers.BatchNormalization(),


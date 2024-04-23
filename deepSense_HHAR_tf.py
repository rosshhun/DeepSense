import tensorflow as tf 
import numpy as np

import plot

import time
import math
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

layers = tf.keras.layers 

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2
CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
OUT_DIM = 6#len(idDict)
WIDE = 20
CONV_KEEP_PROB = 0.8

BATCH_SIZE = 64
TOTAL_ITER_NUM = 1000000000

select = 'a'

metaDict = {'a':[119080, 1193], 'b':[116870, 1413], 'c':[116020, 1477]}
TRAIN_SIZE = metaDict[select][0]
EVAL_DATA_SIZE = metaDict[select][1]
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))


# prints = []
# vals = {}

# def plot(name, value):
#     if not name in vals:
#         vals[name] = []
#     vals[name].append(value)

# def tick():
#     pass

# def flush():
#     for name, values in vals.items():
#         prints.append("{}\\t{}".format(name, np.mean(np.array(values))))
#     for line in prints:
#         print(line)
#     prints.clear()
#     vals.clear()

###### Import training data
def read_audio_csv(value):
    line = value
    defaultVal = [[0.] for idx in range(WIDE * FEATURE_DIM + OUT_DIM)]
    fileData = tf.io.decode_csv(line, record_defaults=defaultVal)
    features = fileData[:WIDE * FEATURE_DIM]
    features = tf.reshape(features, [WIDE, FEATURE_DIM])
    labels = fileData[WIDE * FEATURE_DIM:]
    return features, labels

# def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None):
#     dataset = tf.data.Dataset.from_tensor_slices(filenames)
#     if shuffle_sample:
#         dataset = dataset.shuffle(buffer_size=len(filenames))
#     dataset = dataset.flat_map(tf.data.TextLineDataset)
#     dataset = dataset.map(read_audio_csv)
#     if num_epochs is not None:
#         dataset = dataset.repeat(num_epochs)
#     dataset = dataset.batch(batch_size)
#     return dataset

def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None):
    # Create a dataset from filenames
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Shuffle if needed
    if shuffle_sample:
        dataset = dataset.shuffle(buffer_size=len(filenames))

    # Read and parse each CSV line
    dataset = dataset.flat_map(tf.data.TextLineDataset)
    dataset = dataset.map(read_audio_csv)

    # Repeat for multiple epochs
    if num_epochs is not None:
        dataset = dataset.repeat(num_epochs)

    # Batch the data
    dataset = dataset.batch(batch_size)

    # Prefetch for performance (optional)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

######

# def batch_norm_layer(inputs, phase_train, scope=None):
# 	return tf.cond(phase_train,  
# 		lambda: layers.batch_norm(inputs, is_training=True, scale=True, 
# 			updates_collections=None, scope=scope),  
# 		lambda: layers.batch_norm(inputs, is_training=False, scale=True,
# 			updates_collections=None, scope=scope, reuse = True)) 

def batch_norm_layer(inputs, train, scope=None):
	# inputs = tf.expand_dims(inputs, axis=-1)
    # batch_norm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
    # if train:
    #     return batch_norm(inputs, training=True)
    # else:
    #     return batch_norm(inputs, training=False)
	# Add a dummy channel dimension
    inputs = tf.expand_dims(inputs, axis=-1)

    batch_norm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
    if train:
        outputs = batch_norm(inputs, training=True)
    else:
        outputs = batch_norm(inputs, training=False)

    # Remove the dummy channel dimension
    outputs = tf.squeeze(outputs, axis=-1)

    return outputs
	
class DropoutGRUCell(tf.keras.layers.GRUCell):
	def __init__(self, units, dropout_rate, **kwargs):
		super().__init__(units, **kwargs)
		self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

	def call(self, inputs, states, training=None):
		outputs, new_states = super().call(inputs, states, training=training)
		outputs = self.dropout_layer(outputs, training=training)
		return outputs, new_states
	
gru_cell1 = DropoutGRUCell(INTER_DIM, dropout_rate=0.5)
gru_cell2 = DropoutGRUCell(INTER_DIM, dropout_rate=0.5)

def deepSense(inputs, train, reuse=False, name='deepSense'):
	with tf.name_scope(name):
		used = tf.sign(tf.reduce_max(tf.abs(inputs), axis=2))  # (BATCH_SIZE, WIDE)
		length = tf.reduce_sum(used, axis=1)  # (BATCH_SIZE)
		
		mask = tf.sign(tf.reduce_max(tf.abs(inputs), axis=2, keepdims=True))
		mask = tf.tile(mask, [1, 1, INTER_DIM])  # (BATCH_SIZE, WIDE, INTER_DIM)
		avgNum = tf.reduce_sum(mask, axis=1)  # (BATCH_SIZE, INTER_DIM)
		# inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM)
		sensor_inputs = tf.expand_dims(inputs, axis=3)
		# sensor_inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM, CHANNEL=1)
		acc_inputs, gyro_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=2)

		acc_conv1 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, 2*3*CONV_LEN),
                                   strides=(1, 2*3), padding='VALID', activation=None, data_format='channels_last')(acc_inputs)
		acc_conv1 = batch_norm_layer(acc_conv1, train, scope='acc_BN1')
		acc_conv1 = tf.nn.relu(acc_conv1)
		acc_conv1_shape = acc_conv1.shape
		acc_conv1 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(acc_conv1, training=train)

		acc_conv2 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, CONV_LEN_INTE),
										strides=(1, 1), padding='VALID', activation=None, data_format='channels_last')(acc_conv1)
		acc_conv2 = batch_norm_layer(acc_conv2, train, scope='acc_BN2')
		acc_conv2 = tf.nn.relu(acc_conv2)
		acc_conv2_shape = acc_conv2.shape
		acc_conv2 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(acc_conv2, training=train)

		acc_conv3 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, CONV_LEN_LAST),
										strides=(1, 1), padding='VALID', activation=None, data_format='channels_last')(acc_conv2)
		acc_conv3 = batch_norm_layer(acc_conv3, train, scope='acc_BN3')
		acc_conv3 = tf.nn.relu(acc_conv3)
		acc_conv3_shape = acc_conv3.shape
		acc_conv_out = tf.reshape(acc_conv3, [acc_conv3_shape[0], acc_conv3_shape[1], 1, acc_conv3_shape[2], acc_conv3_shape[3]])

		gyro_conv1 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, 2*3*CONV_LEN),
                                    strides=(1, 2*3), padding='VALID', activation=None, data_format='channels_last')(gyro_inputs)
		gyro_conv1 = batch_norm_layer(gyro_conv1, train, scope='gyro_BN1')
		gyro_conv1 = tf.nn.relu(gyro_conv1)
		gyro_conv1_shape = gyro_conv1.shape
		gyro_conv1 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(gyro_conv1, training=train)

		gyro_conv2 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, CONV_LEN_INTE),
											strides=(1, 1), padding='VALID', activation=None, data_format='channels_last')(gyro_conv1)
		gyro_conv2 = batch_norm_layer(gyro_conv2, train, scope='gyro_BN2')
		gyro_conv2 = tf.nn.relu(gyro_conv2)
		gyro_conv2_shape = gyro_conv2.shape
		gyro_conv2 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(gyro_conv2, training=train)

		gyro_conv3 = tf.keras.layers.Conv2D(CONV_NUM, kernel_size=(1, CONV_LEN_LAST),
											strides=(1, 1), padding='VALID', activation=None, data_format='channels_last')(gyro_conv2)
		gyro_conv3 = batch_norm_layer(gyro_conv3, train, scope='gyro_BN3')
		gyro_conv3 = tf.nn.relu(gyro_conv3)
		gyro_conv3_shape = gyro_conv3.shape
		gyro_conv_out = tf.reshape(gyro_conv3, [gyro_conv3_shape[0], gyro_conv3_shape[1], 1, gyro_conv3_shape[2], gyro_conv3_shape[3]])

		sensor_conv_in = tf.concat([acc_conv_out, gyro_conv_out], 2)
		senor_conv_shape = sensor_conv_in.get_shape().as_list()	
		sensor_conv_in_shape = sensor_conv_in.shape
		sensor_conv_in = tf.reshape(sensor_conv_in, [sensor_conv_in_shape[0], sensor_conv_in_shape[1], sensor_conv_in_shape[2] * sensor_conv_in_shape[3], sensor_conv_in_shape[4]])
		
		sensor_conv1 = tf.keras.layers.Conv2D(CONV_NUM2, kernel_size=(2, CONV_MERGE_LEN),
                                      strides=(1, 1), padding='SAME', activation=None, data_format='channels_last')(sensor_conv_in)
		
		sensor_conv1 = batch_norm_layer(sensor_conv1, train, scope='sensor_BN1')
		sensor_conv1 = tf.nn.relu(sensor_conv1)
		sensor_conv1_shape = sensor_conv1.shape
		sensor_conv1 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(sensor_conv1, training=train)

		sensor_conv2 = tf.keras.layers.Conv2D(CONV_NUM2, kernel_size=(2, CONV_MERGE_LEN2),
											strides=(1, 1), padding='SAME', activation=None, data_format='channels_last')(sensor_conv1)
		sensor_conv2 = batch_norm_layer(sensor_conv2, train, scope='sensor_BN2')
		sensor_conv2 = tf.nn.relu(sensor_conv2)
		sensor_conv2_shape = sensor_conv2.shape
		sensor_conv2 = tf.keras.layers.Dropout(rate=1-CONV_KEEP_PROB)(sensor_conv2, training=train)

		sensor_conv3 = tf.keras.layers.Conv2D(CONV_NUM2, kernel_size=(2, CONV_MERGE_LEN3),
											strides=(1, 1), padding='SAME', activation=None, data_format='channels_last')(sensor_conv2)
		sensor_conv3 = batch_norm_layer(sensor_conv3, train, scope='sensor_BN3')
		sensor_conv3 = tf.nn.relu(sensor_conv3)
		sensor_conv3_shape = sensor_conv3.shape
		sensor_conv_out = tf.reshape(sensor_conv3, [sensor_conv3_shape[0], sensor_conv3_shape[1], sensor_conv3_shape[2] * sensor_conv3_shape[3]])

		# class DropoutGRUCell(tf.keras.layers.GRUCell):
		# 	def __init__(self, units, dropout_rate, **kwargs):
		# 		super().__init__(units, **kwargs)
		# 		self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

		# 	def call(self, inputs, states, training=None):
		# 		outputs, new_states = super().call(inputs, states, training=training)
		# 		outputs = self.dropout_layer(outputs, training=training)
		# 		return outputs, new_states

		# gru_cell1 = DropoutGRUCell(INTER_DIM, dropout_rate=0.5)
		# gru_cell2 = DropoutGRUCell(INTER_DIM, dropout_rate=0.5)

		# cell = tf.keras.layers.StackedRNNCells([gru_cell1, gru_cell2])
		# init_state = cell.get_initial_state(batch_size=BATCH_SIZE)

		# cell_output, final_stateTuple, _ = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)(sensor_conv_out, mask=mask, initial_state=init_state)

		# gru_cell1 = DropoutGRUCell(INTER_DIM, dropout_rate=0.5)
		# gru_cell2 = DropoutGRUCell(INTER_DIM, dropout_rate=0.5)
		
		# loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)

		cell = tf.keras.layers.StackedRNNCells([gru_cell1, gru_cell2])

		cell_output, final_stateTuple, _ = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)(sensor_conv_out, mask=mask)


		sum_cell_out = tf.reduce_sum(cell_output * mask, axis=1, keepdims=False)
		avg_cell_out = sum_cell_out/avgNum

		logits = tf.keras.layers.Dense(OUT_DIM, activation=None, name='output')(avg_cell_out)

		# t_vars = []
		# for layer in [gru_cell1, gru_cell2]:
		# 	t_vars += layer.trainable_variables

		# loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)

		# regularizers = 0.
		# for var in t_vars:
		# 	regularizers += tf.nn.l2_loss(var)
		# loss += 5e-4 * regularizers

		# discOptimizer = tf.keras.optimizers.Adam(
		# 		learning_rate=1e-4, 
		# 		beta_1=0.5,
		# 		beta_2=0.9
		# 	)
		
		# # Get gradients
		# gradients = discOptimizer.compute_gradients(loss, var_list=t_vars)

		# # Apply gradients
		# discOptimizer.apply_gradients(zip(gradients, t_vars))

		return logits

csvFileList = []
csvDataFolder1 = os.path.join('sepHARData_'+select, "train")
orgCsvFileList = os.listdir(csvDataFolder1)
for csvFile in orgCsvFileList:
	if csvFile.endswith('.csv'):
		csvFileList.append(os.path.join(csvDataFolder1, csvFile))

csvEvalFileList = []
csvDataFolder2 = os.path.join('sepHARData_'+select, "eval")
orgCsvFileList = os.listdir(csvDataFolder2)
for csvFile in orgCsvFileList:
	if csvFile.endswith('.csv'):
		csvEvalFileList.append(os.path.join(csvDataFolder2, csvFile))

global_step = tf.Variable(0, trainable=False)

# batch_feature, batch_label = input_pipeline(csvFileList, BATCH_SIZE)
# batch_eval_feature, batch_eval_label = input_pipeline(csvEvalFileList, BATCH_SIZE, shuffle_sample=False)
dataset = input_pipeline(csvFileList, BATCH_SIZE)
eval_dataset = input_pipeline(csvEvalFileList, BATCH_SIZE, shuffle_sample=False)

# train_status = tf.placeholder(tf.bool)
# trainX = tf.cond(train_status, lambda: tf.identity(batch_feature), lambda: tf.identity(batch_eval_feature))
# trainY = tf.cond(train_status, lambda: tf.identity(batch_label), lambda: tf.identity(batch_eval_label))

# logits = deepSense(trainX, train_status, name='deepSense')
for batch_feature, batch_label in dataset:
	logits = deepSense(batch_feature, True, name='deepSense')

predict = tf.argmax(logits, axis=1)

# batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=trainY)
batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
loss = tf.reduce_mean(batchLoss)
for batch_eval_feature, batch_eval_label in eval_dataset:
	logits_eval = deepSense(batch_eval_feature, False, reuse=True, name='deepSense')
predict_eval = tf.argmax(logits_eval, axis=1)
loss_eval = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_eval, labels=batch_eval_label))


tf.compat.v1.global_variables_initializer()
coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord)

for iteration, (batch_feature, batch_label) in enumerate(dataset):
	logits = deepSense(batch_feature, True, name='deepSense')
	# _, lossV, _trainY, _predict = sess.run([discOptimizer, loss, trainY, predict], feed_dict = {
	# 	train_status: True
	# 	})
		# Calculate loss
  
	print("Input feature shape:", batch_feature.shape)
	print("Input label shape:", batch_label.shape)
	print("Input feature values:", batch_feature[0])
	print("Input label values:", batch_label[0])
  
	print("Logits shape:", logits.shape)
	print("Logits values:", logits[0])

	batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
	loss = tf.reduce_mean(batchLoss)

	print("Batch Loss shape:", batchLoss.shape)
	print("Batch Loss values:", batchLoss[0])
	print("Loss value:", loss)

	# Get trainable variables
	t_vars = []
	for layer in [gru_cell1, gru_cell2]:
		t_vars += layer.trainable_variables
		print("Trainable variables:", t_vars)

	# Add regularization
	regularizers = 0.
	for var in t_vars:
		regularizers += tf.nn.l2_loss(var)
	loss += 5e-4 * regularizers

	# Calculate gradients using GradientTape
	with tf.GradientTape() as tape:
		# Forward pass (calculate loss again within the tape context)
		logits = deepSense(batch_feature, True, name='deepSense')
		batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
		loss = tf.reduce_mean(batchLoss) + 5e-4 * regularizers  # Include regularization
		print("Batch Loss shape:", batchLoss.shape)
		print("Loss value:", loss)

	gradients = tape.gradient(loss, t_vars)

	# Check if gradients are not empty
	for grad, var in zip(gradients, t_vars):
		if grad is None:
			print(f"Gradient for variable {var.name} is None")
		else:
			print(f"Gradient shape for variable {var.name}: {grad.shape}")

	discOptimizer = tf.keras.optimizers.Adam()

	# Compute gradients
	# gradients = tf.gradients(loss, t_vars)

	# Apply gradients
	discOptimizer.apply_gradients(zip(gradients, t_vars))
	# _, lossV, _trainY, _predict = sess.run([discOptimizer, loss, batch_label, predict])
	lossV = loss.numpy()  # Get the loss value as a numpy array
	_trainY = batch_label.numpy()  # Get labels as a numpy array
	_predict = np.argmax(logits.numpy(), axis=1)
	#before
	_label = np.argmax(_trainY, axis=1)
	_accuracy = np.mean(_label == _predict)
	plot.plot('train cross entropy', lossV)
	plot.plot('train accuracy', _accuracy)


	if iteration % 100 == 0:
		dev_accuracy = []
		dev_cross_entropy = []
		for eval_idx in range(EVAL_ITER_NUM):
			# eval_loss_v, _trainY, _predict = sess.run([loss, trainY, predict], feed_dict ={train_status: False})
			# eval_loss_v, _trainY, _predict = sess.run([loss, batch_eval_label, predict_eval])

			# With this:
			eval_loss_v = loss.numpy()  # Get loss value as a NumPy array
			_trainY = batch_eval_label.numpy()  # Get evaluation labels as a NumPy array
			_predict = predict_eval.numpy()  # Get evaluation predictions as a NumPy array 
			_label = np.argmax(_trainY, axis=1)
			_accuracy = np.mean(_label == _predict)
			dev_accuracy.append(_accuracy)
			dev_cross_entropy.append(eval_loss_v)
		plot.plot('train cross entropy', loss.numpy())
		plot.plot('train accuracy', _accuracy)


	# if (iteration < 5) or (iteration % 50 == 49):
	plot.flush()

	plot.tick()
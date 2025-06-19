#LSTM Recurrent Network for Text Generation
#MODEL CREATION AND SAVING WEIGHTS

import os
import importlib.util

######################################################################
##### CHECK REQUIREMENTS AND OPTIONAL GPU FOR TRAINING            ####
######################################################################

# create abs path to package_utilities by concat relative path to os abs path
relative_path_validate_requirements="../package_utilities.py"

validate_requirements_path= os.path.abspath(relative_path_validate_requirements)

spec= importlib.util.spec_from_file_location("package_utilities.py",relative_path_validate_requirements)
package_utilities= importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_utilities)

relative_path_req_file="../requirements.txt"
req_file=os.path.abspath(relative_path_req_file)

package_utilities.validate_requirements(req_file)
package_utilities.check_cuda()

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.layers import BatchNormalization

######################################################################
##### LOAD DATASET AND CREATE MAP OF UNIQUE CHARACTERS           #####
######################################################################

# Load the text corpus
relative_path_bee_text_corpus = "../training_text/Bee_movie.txt"
bee_text_corpus = os.path.abspath(relative_path_bee_text_corpus)

raw_text = open(bee_text_corpus, 'r', encoding='utf-8').read()

# Convert all the text to lowercase
raw_text = raw_text.lower()

# Create a map that maps each unique character in the text to a unique integer value
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# Display the total number of characters (n_chars) and the vocabulary (the number of unique characters)
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

######################################################################
##### CREATE TRAINING PATTERNS                                   #####
######################################################################

# Create the patterns to be used for training
seq_length = 16	# fixed length sliding window for training pattern
num_memory_cell = 256
dataX = []			# input sequences
dataY = []			# outputs

for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

######################################################################
##### TRANSFORM DATA TO BE SUITABLE FOR KERAS                    #####
######################################################################

# Reshape dataX to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# Rescale integers mapped to characters to the range 0-to-1 to accommodate learning using sigmoid function
X = X / float(n_vocab)

# One hot encode the output variable
y = to_categorical(dataY)

######################################################################
##### BUILD THE LSTM MODEL                                       #####
######################################################################

# Build sequential model containing 2 LSTM layers and 2 Dropout layers, followed by a dense output layer
model = Sequential()

# First LSTM layer
model.add(LSTM(num_memory_cell, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# Seconds LSTM Layer
model.add(LSTM(num_memory_cell, return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# Third LSTM Layer
model.add(LSTM(num_memory_cell))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# Output Layer
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model using the Adam optimizer and categorical crossentropy for the loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define the checkpoint; i.e., saved past state of the model
# save_best_only is odd to use here because the loss usually goes down after every epoch, set to false 
#remove save best parameter to not save any of the checkpoints
# verbose displays logs of increasing detail beginning with 0 (no output)
# ".keras" supported by TF(2.11) as TF custom keras format (faster but less portability), use hdf5 forTF(<=2.10) and greater portability
filepath = "../weights/checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,  mode='min',save_weights_only=False)
callbacks_list = [checkpoint]

######################################################################
##### TRAINING                                                   #####
######################################################################

# Train the model
history=model.fit(X, y, epochs=25, batch_size=64, callbacks=callbacks_list)

# Save only the best model after training (lowest loss)
# I used key to map epoch index to the corresponding loss value. It will only save smallest, rather than saving after every epoch with save_best_only
#min(iterable, key=lambda i: list[value][i])
best_epoch=min(range(len(history.history["loss"])),key= lambda i: history.history['loss'][i])
best_loss= history.history["loss"][best_epoch]

# Manually save the best model with the lowest loss at the end
best_model_filepath = f"../weights/best_checkpoints_bee/best_model_BeeMovie-lstm_ThreeLayer_seqLen16_{best_epoch+1:02d}-{best_loss:.4f}.hdf5"
model.save(best_model_filepath)
print(f"Best model saved at epoch {best_epoch+1} with loss {best_loss:.4f}")

# clear the checkpoints folder, keep only the best epoch
dir_path="../weights/checkpoints/"
checkpoint_directories=[]
for checkpoint in os.listdir(dir_path):
	checkpoint_path=os.path.join(dir_path,checkpoint)
	checkpoint_directories.append(checkpoint_path)
	print(checkpoint_path)

print_directories_response=input("Would you like to view+delete the checkpoint directories\nThis EXCLUDES the best epoch (Y/N): ")

if print_directories_response.lower() in ['y','yes']:
	for checkpoint_path in checkpoint_directories:
		os.remove(checkpoint_path)
		print(f'{checkpoint_path} has been deleted')
else:
	print(f'skipping {checkpoint_path}')
	
#LSTM Recurrent Network for Text Generation
#TEXT GENERATION WITH PRE-DEFINED WEIGHTS

import sys #returns num characters for .write()

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical

# File management
import os
import importlib.util
import glob
######################################################################
##### LOAD DATASET AND CREATE MAP OF UNIQUE CHARACTERS           #####
######################################################################

# Load the text corpus
relative_path_cat_text_corpus = "../../training_text/Cat_in_the_Hat.txt"
cat_text_corpus=os.path.abspath(relative_path_cat_text_corpus)
raw_text = open(cat_text_corpus, 'r', encoding='utf-8').read()

# Convert all the text to lowercase
raw_text = raw_text.lower()

# Create a map that maps each unique character in the text to a unique integer value
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# Also create a reverse map to be able to output the character that maps to a specific integer
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Display the total number of characters (n_chars) and the vocabulary (the number of unique characters)
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

######################################################################
##### CREATE TRAINING PATTERNS                                   #####
######################################################################

# Create the patterns to be used for training
seq_length = 100	# fixed length sliding window for training pattern
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
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))

######################################################################
##### LOAD THE SAVED NETWORK WEIGHTS FROM CHECKPOINT             #####
######################################################################

# Load the network weights from the specified checkpoint file
filename_pattern = "../../weights/best_checkpoints_cat/best_model_.5dropouts_*.hdf5"
matched_files=glob.glob(filename_pattern)

if not matched_files:
	print("No saved models found")
	exit()

for idx,weights_file in enumerate(matched_files):
	print(f"{idx+1}. {weights_file}")


load_weights_file=""
while True:
	try:
		
		selected_model=int(input("\nEnter the number of the model to load (or 0 to cancel): "))
		if selected_model==0:
			print("No model selected")
			exit()
		elif(1<=selected_model and selected_model<=len(matched_files)):
			load_weights_file=matched_files[selected_model-1]
			print(f"Loading: {load_weights_file}")
			model.load_weights(load_weights_file)
			print("Model weights loaded successfully")
			break
		else:
			print("Invalid selection! Please make a valid selection")
	except ValueError:
		print("Invalid input, please enter a number.")


# Compile the model using the Adam optimizer and categorical crossentropy for the loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')

######################################################################
##### TEXT GENERATION                                            #####
######################################################################

# Pick a random seed and display it
start = numpy.random.randint(0, len(dataX) - 1)

pattern = dataX[start]

print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

#save seed to txt
seed_sequence="\"" + ''.join([int_to_char[value] for value in pattern])+ "\""

# Generate 1000 characters
output_sequence=""
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	output_sequence=output_sequence+result
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

output_text_file=f"../../generated_text_OUTPUTS/OUTPUT - {load_weights_file[46:-5]}.txt"


with open (output_text_file,"w") as file:
	file.write("Seed Sequence:\n\n" + seed_sequence + "\n\nModel Output (1000 characters):\n\n" +output_sequence)
# Text generation complete	
print("\nOutput saved in the generated_text_OUTPUTS directory")
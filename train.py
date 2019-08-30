import glob
import pickle

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Bidirectional
import numpy
from keras.utils import np_utils
from music21 import instrument, note, stream, chord, converter, duration

TIMESTEP = 0.25
SEQUENCE_LEN = int(8 / TIMESTEP)
MODEL_NAME = 'my_music_model'



def get_notes():
    notes = []

    for file in glob.glob("midi/*.mid"):
        print("Parsing %s" % file)
        try:
            midi = converter.parse(file)
        except IndexError as e:
            print(f"Could not parse {file}")
            print(e)
            continue

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        prev_offset = 0.0
        for element in notes_to_parse:
            if isinstance(element, note.Note) or isinstance(element, chord.Chord):
                duration = element.duration.quarterLength
                if isinstance(element, note.Note):
                    name = element.pitch
                elif isinstance(element, chord.Chord):
                    name = ".".join(str(n) for n in element.normalOrder)
                notes.append(f"{name}${duration}")

                rest_notes = int((element.offset - prev_offset) / TIMESTEP - 1)
                for _ in range(0, rest_notes):
                    notes.append("NULL")

            prev_offset = element.offset

    with open("notes/" + 'notes_model', "wb") as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # get all pitch names
    pitch_names = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number + 1) for number, note in enumerate(pitch_names))
    note_to_int["NULL"] = 0

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - SEQUENCE_LEN, 1):
        sequence_in = notes[i: i + SEQUENCE_LEN]
        sequence_out = notes[i + SEQUENCE_LEN]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, SEQUENCE_LEN, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    print("Input shape ", network_input.shape)
    print("Output shape ", n_vocab)
    """ create the structure of the neural network """
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(512, return_sequences=True),
            input_shape=(network_input.shape[1], network_input.shape[2]),
        )
    )
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    return model


def start(notes_file=None):
    """ Train a Neural Network to generate music """
    if not notes_file:
        notes = get_notes()
    else:
        with open(notes_file, "rb") as filepath:
            notes = pickle.load(filepath)

    # get amount of pitch names
    n_vocab = len(set(notes))
    print("n_vocab", n_vocab)

    network_input, network_output = prepare_sequences(notes, n_vocab)

    my_model = create_network(network_input, n_vocab)

    train(network_input, network_output,my_model)
    file_name = MODEL_NAME + ".hdf5"
    my_model.save(file_name)
    print(f"Model saved to {file_name}")


def train(network_input, network_output, model):
    """ train the neural network """
    filepath = (
            MODEL_NAME + "-weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    )
    checkpoint = ModelCheckpoint(
        filepath, monitor="loss", verbose=0, save_best_only=True, mode="min"
    )
    callbacks_list = [checkpoint]
    model.fit(
        network_input,
        network_output,
        epochs=15,
        batch_size=64,
        callbacks=callbacks_list,
    )



start('notes/notes_model')
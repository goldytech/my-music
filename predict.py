import os
import pickle
import random
import numpy
from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation
from music21 import instrument, duration, note, stream, chord
from numpy.core._multiarray_umath import ndarray

TIMESTEP = 0.25
SEQUENCE_LEN = int(8 / TIMESTEP)


def generate():
    """ Generate a piano midi file """
    # load the notes used to train the model
    with open('notes/notes_model', "rb") as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))
    network_input, normalized_input = prepare_sequences(
        notes, pitchnames, n_vocab
    )
    weights = 'models/my_music_model-weights-improvement-03-4.1090-bigger.hdf5'
    model = create_network(normalized_input, n_vocab)
    model.load_weights(weights)

    prediction_output = generate_notes(
        model, network_input, pitchnames, n_vocab
    )
    create_midi(prediction_output )


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


def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number + 1) for number, note in enumerate(pitchnames))
    note_to_int["NULL"] = 0

    network_input = []
    output = []
    for i in range(0, len(notes) - SEQUENCE_LEN, 1):
        sequence_in = notes[i: i + SEQUENCE_LEN]
        sequence_out = notes[i + SEQUENCE_LEN]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, SEQUENCE_LEN, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)
    return (network_input, normalized_input)


def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    int_to_note = dict((number + 1, note) for number, note in enumerate(pitchnames))
    int_to_note[0] = "NULL"

    def get_start():
        # pick a random sequence from the input as a starting point for the prediction
        start = numpy.random.randint(0, len(network_input) - 1)
        pattern = network_input[start]
        prediction_output = []
        return pattern, prediction_output

    # generate verse 1
    verse1_pattern, verse1_prediction_output = get_start()
    for note_index in range(4 * SEQUENCE_LEN):
        prediction_input: ndarray = numpy.reshape(verse1_pattern, (1, len(verse1_pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=1)

        # index = numpy.argmax(prediction)
        index = random.randint(0, prediction.size -1)
        print("index", index)
        result = int_to_note[index]
        verse1_prediction_output.append(result)

        verse1_pattern.append(index)
        verse1_pattern = verse1_pattern[1: len(verse1_pattern)]

    # generate verse 2
    verse2_pattern = verse1_pattern
    verse2_prediction_output = []
    for note_index in range(4 * SEQUENCE_LEN):
        prediction_input = numpy.reshape(
            verse2_pattern, (1, len(verse2_pattern), 1)
        )
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        # index = numpy.argmax(prediction)
        index = random.randint(0, prediction.size - 1)
        print("index", index)
        result = int_to_note[index]
        verse2_prediction_output.append(result)

        verse2_pattern.append(index)
        verse2_pattern = verse2_pattern[1: len(verse2_pattern)]

    # generate chorus
    chorus_pattern, chorus_prediction_output = get_start()
    for note_index in range(4 * SEQUENCE_LEN):
        prediction_input = numpy.reshape(
            chorus_pattern, (1, len(chorus_pattern), 1)
        )
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        # index = numpy.argmax(prediction)
        index = random.randint(0, prediction.size - 1)
        print("index", index)
        result = int_to_note[index]
        chorus_prediction_output.append(result)

        chorus_pattern.append(index)
        chorus_pattern = chorus_pattern[1: len(chorus_pattern)]

    # generate bridge
    bridge_pattern, bridge_prediction_output = get_start()
    for note_index in range(4 * SEQUENCE_LEN):
        prediction_input = numpy.reshape(
            bridge_pattern, (1, len(bridge_pattern), 1)
        )
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        # index = numpy.argmax(prediction)
        index = random.randint(0, prediction.size - 1)
        print("index", index)
        result = int_to_note[index]
        bridge_prediction_output.append(result)

        bridge_pattern.append(index)
        bridge_pattern = bridge_pattern[1: len(bridge_pattern)]

    return (
            verse1_prediction_output
            + chorus_prediction_output
            + verse2_prediction_output
            + chorus_prediction_output
            + bridge_prediction_output
            + chorus_prediction_output
    )


def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        if "$" in pattern:
            pattern, dur = pattern.split("$")
            if "/" in dur:
                a, b = dur.split("/")
                dur = float(a) / float(b)
            else:
                dur = float(dur)

        # pattern is a chord
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.duration = duration.Duration(dur)
            output_notes.append(new_chord)
        # pattern is a rest
        elif pattern is "NULL":
            offset += TIMESTEP
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.duration = duration.Duration(dur)
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += TIMESTEP

    midi_stream = stream.Stream(output_notes)

    output_file = 'output/out.mid'
    print("output to " + output_file)
    midi_stream.write("midi", fp=output_file)


generate()

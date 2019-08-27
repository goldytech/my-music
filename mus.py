from typing import Optional, Any

from music21 import converter, instrument, note, chord, stream, duration


def get_notes(midi_file):
    midi: Optional[Any] = converter.parse(midi_file)
    print(f"Parsing file {midi_file}")
    time_signature = midi.getTimeSignatures()[0]
    # midi.show()
    notes = []
    notes_to_parse = None
    try:
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
            # notes.append(element.pitch.fullName)
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
            # notes.append(element.fullName)

    return notes


def convert_notes_to_number(notes):
    pitch_names = sorted(set(item for item in notes))
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))
    return note_to_int


def create_midi_file():
    patterns = []
    offset = 0

    patterns.append("F4")
    patterns.append("G#4")
    patterns.append("G4")
    patterns.append("F4")
    patterns.append("F4")
    patterns.append("G#4")
    patterns.append("G4")
    patterns.append("F4")
    patterns.append("G#4")
    patterns.append("G4")
    patterns.append("F4")
    patterns.append("E4")
    patterns.append("E4")
    patterns.append("C4")
    patterns.append("G4")
    patterns.append("G4")
    patterns.append("F4")
    patterns.append("F4")
    midi_notes = []
    for pattern in patterns:
        midi_note = note.Note(pattern)
        midi_note.storedInstrument = instrument.Piano()
        midi_note.offset = offset
        # midi_note.beat = beat
        midi_note.duration = duration.Duration(quarterLength=1.25)
        midi_notes.append(midi_note)
        offset += 0.95

    midi_stream = stream.Stream(midi_notes)
    midi_stream.write('midi', fp='test.mid')


if __name__ == '__main__':
    notes = get_notes('midi/test.mid')
    # convert_notes_to_number(notes)
    # create_midi_file()

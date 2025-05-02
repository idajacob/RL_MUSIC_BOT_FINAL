import pretty_midi
import mido

def save_melody_as_midi(note_list, filepath, velocity=100, duration=0.5):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    start = 0.0

    for note in note_list:
        n = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note),
            start=start,
            end=start + duration
        )
        piano.notes.append(n)
        start += duration

    midi.instruments.append(piano)
    midi.write(filepath)

def extract_melody_from_midi(filepath):
    midi_data = pretty_midi.PrettyMIDI(filepath)
    melody = []

    if not midi_data.instruments:
        return melody

    instrument = midi_data.instruments[0]
    instrument.notes.sort(key=lambda note: note.start)

    for note in instrument.notes:
        melody.append(note.pitch)

    return melody

def extract_intervals_from_midi(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = [note.pitch for instrument in midi.instruments for note in instrument.notes if not instrument.is_drum]
    intervals = [notes[i+1] - notes[i] for i in range(len(notes)-1)]
    return intervals

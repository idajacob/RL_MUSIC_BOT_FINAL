import pretty_midi

def save_melody_as_midi(note_list, output_file="output_pretty.mid"):
    """
    Gemmer en melodi som en MIDI-fil ved hjælp af PrettyMIDI.
    
    Parameters:
    - note_list: Liste af tuples (note, duration), hvor note er en MIDI-tone (0-127) og duration er varigheden i sekunder.
    - output_file: Navnet på outputfilen.
    """
    # Opret et PrettyMIDI-objekt og et instrument
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    # Starttiden for hver tone
    current_time = 0.0

    for note, duration in note_list:
        # Opret en tone med start- og slut-tidspunkt
        midi_note = pretty_midi.Note(
            velocity=100,         # Styrken af tonen
            pitch=note,           # Tonen i MIDI-format
            start=current_time,   # Starttidspunkt for tonen
            end=current_time + duration  # Sluttidspunkt for tonen
        )
        instrument.notes.append(midi_note)
        current_time += duration  # Opdater tiden til næste tone

    # Tilføj instrumentet til MIDI-filen
    midi_data.instruments.append(instrument)

    # Gem som MIDI-fil
    midi_data.write(output_file)
    print(f"MIDI-fil gemt som: {output_file}")
    

def extract_intervals_from_midi(file_path):
    """
    Ekstraherer intervaller fra en MIDI-fil ved hjælp af PrettyMIDI.
    
    Parameters:
    - file_path: Sti til MIDI-filen.
    
    Returnerer:
    - Liste af intervaller mellem efterfølgende toner.
    """
    midi_data = pretty_midi.PrettyMIDI(file_path)
    notes = []

    # Iterér over alle instrumenter, men undgå trommer
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append(note.pitch)

    # Beregn intervaller mellem toner
    intervals = [notes[i] - notes[i - 1] for i in range(1, len(notes))]
    return intervals


def extract_melody_from_midi(filepath):
    """
    Ekstraherer en melodi (toner i rækkefølge) fra en MIDI-fil.
    
    Parameters:
    - filepath: Sti til MIDI-filen.
    
    Returnerer:
    - En liste af toner (MIDI-numre) i rækkefølge som de spilles.
    """
    midi_data = pretty_midi.PrettyMIDI(filepath)
    melody = []

    # Hvis der ikke er nogen instrumenter, returner en tom liste
    if not midi_data.instruments:
        return melody

    # Antag, at melodien er på det første instrument
    instrument = midi_data.instruments[0]
    # Sortér noderne efter starttidspunkt
    instrument.notes.sort(key=lambda note: note.start)

    # Tilføj toner til listen
    for note in instrument.notes:
        melody.append(note.pitch)

    return melody
""

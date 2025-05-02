from utils.midi_tools import record_midi_input, save_melody_as_midi

# Brug din enhedsnavn præcis som vist i listen
port_name = "Keystation Mini 32 0"

# Optag i 8 sekunder
melody = record_midi_input(port_name=port_name, duration=8)
print("Spillede toner:", melody)

# Gem som MIDI-fil
save_melody_as_midi(melody, "data/output_midi/user_input.mid")
print("Gemte MIDI til: data/output_midi/user_input.mid")

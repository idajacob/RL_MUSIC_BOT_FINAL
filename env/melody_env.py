import gym
import numpy as np
from gym import spaces
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.midi_tools import extract_intervals_from_midi, save_melody_as_midi

# === Tone-område ===
# Dette er de toner, som agenten kan vælge imellem
FULL_RANGE = list(range(60, 90))

# === Mulige rytmer (i sekunder) ===
# Agenten kan vælge varigheden af hver tone
RHYTHM_VALUES = [0.25, 0.5, 1.0, 2.0]  # 1/4 node, 1/2 node, 1 node, 2 node

# === MelodyEnv (Miljø) ===
# Dette miljø er ansvarligt for at modtage handlinger fra agenten, evaluere dem
# og returnere en belønning samt den nye state.
class MelodyEnv(gym.Env):
    def __init__(self, reference_melody=None):
        super(MelodyEnv, self).__init__()

        # Definerer miljøparametre
        self.max_length = 32                 # Maksimal længde på melodien
        self.min_note = 60                   # Minimum MIDI-tone
        self.max_note = 90                   # Maksimum MIDI-tone
        self.reference_melody = reference_melody
        self.reference_intervals = extract_intervals_from_midi("data/input_midi/Piano_chopin.mid")[:self.max_length - 1]

        # Action space udvidet til både tone og rytme
        self.action_space = spaces.MultiDiscrete([
            self.max_note - self.min_note + 1,  # Toneområde
            len(RHYTHM_VALUES)                  # Rytmer
        ])

        # Observation space: toner og intervaller
        self.observation_space = spaces.MultiDiscrete([self.action_space.nvec[0]] * self.max_length * 2)

        # Nulstil miljøet
        self.reset()

    def reset(self):
        """Nulstiller miljøet til starttilstand"""
        self.melody = []
        self.steps = 0
        # State er en kombination af toner og rytmer (med padding)
        return np.zeros(self.max_length * 2, dtype=int), {}

    def step(self, action):
        """Udfører en handling og returnerer den nye state, reward og om episoden er færdig"""
        # Opdel handling i tone og rytme
        tone_action, rhythm_action = action
        
        # === Sikring af at actions er inden for gyldigt interval ===
        if not (0 <= rhythm_action < len(RHYTHM_VALUES)):
            print(f"[ADVARSEL] rhythm_action {rhythm_action} er uden for intervallet. Sætter til standardværdi 0.")
            rhythm_action = 0
        
        if not (0 <= tone_action < len(FULL_RANGE)):
            print(f"[ADVARSEL] tone_action {tone_action} er uden for intervallet. Sætter til standardværdi 0.")
            tone_action = 0
        
        # Konverter handling til en faktisk MIDI-tone og rytme
        note = self.min_note + tone_action
        rhythm = RHYTHM_VALUES[rhythm_action]

        # Find nuværende tone
        current_note = self.melody[-1][0] if self.melody else note

        # Begræns næste tone til maks. ±3 halvtoner
        lower_bound = max(self.min_note, current_note - 3)
        upper_bound = min(self.max_note, current_note + 3)

        if note < lower_bound:
            note = lower_bound
        elif note > upper_bound:
            note = upper_bound

        # Blacklist system for gentagelser
        if len(self.melody) > 0 and note == self.melody[-1][0]:
            possible_notes = [n for n in range(lower_bound, upper_bound + 1) if n != self.melody[-1][0]]
            if possible_notes:
                note = np.random.choice(possible_notes)

        reward = 0

        # Belønning for korrekt rytme
        if len(self.reference_melody) > len(self.melody):
            ref_tone, ref_rhythm = self.reference_melody[len(self.melody)]
            if rhythm == ref_rhythm:
                reward += 2.0


        # Straf for gentagelser af samme tone
        if len(self.melody) > 3 and note == self.melody[-1][0]:
            reward -= 5.0

        # Belønning for variation
        # Belønning for variation
        if len(self.melody) > 5 and all(note != n[0] for n in self.melody[-5:]):
            reward += 2.0

        # Tilføj tonen og rytmen til melodien
        self.melody.append((note, rhythm))
        self.steps += 1

        # Tjek om episoden er færdig
        done = self.steps >= self.max_length

        # Skab state som kombination af toner og intervaller
        obs = [n[0] for n in self.melody] + [0] * (self.max_length - len(self.melody))
        intervals = [self.melody[i][0] - self.melody[i-1][0] for i in range(1, len(self.melody))]
        obs_intervals = intervals + [0] * (self.max_length - len(intervals))
        full_state = np.concatenate((obs, obs_intervals))

        return full_state, reward, done, {}

    def render(self, mode='human'):
        """Printer den nuværende melodi og intervaller"""
        print("Melody with Rhythm:")
        for note, rhythm in self.melody:
            print(f"Note: {note}, Duration: {rhythm}")

    def save_to_midi(self, filename="output_pretty.mid"):
        """Gemmer melodien som en MIDI-fil"""
        save_melody_as_midi(self.melody, output_file=filename)


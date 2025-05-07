import gym
import numpy as np
from gym import spaces
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.midi_tools import extract_intervals_from_midi

# === Tone-område (C3 til C5) ===
# Dette er de toner, som agenten kan vælge imellem
FULL_RANGE = list(range(60, 84))

# === MelodyEnv (Miljø) ===
# Dette miljø er ansvarligt for at modtage handlinger fra agenten, evaluere dem
# og returnere en belønning samt den nye state.
class MelodyEnv(gym.Env):
    def __init__(self, reference_melody=None):
        super(MelodyEnv, self).__init__()

        # Definerer miljøparametre
        self.max_length = 64                 # Maksimal længde på melodien
        self.min_note = 60                   # Minimum MIDI-tone (C3)
        self.max_note = 84                   # Maksimum MIDI-tone (C5)
        self.reference_melody = reference_melody
        self.reference_intervals = extract_intervals_from_midi("data/input_midi/chpn_op7_1.MID")[:self.max_length - 1]

        # Action space og observation space
        self.action_space = spaces.Discrete(self.max_note - self.min_note + 1)
        self.observation_space = spaces.MultiDiscrete([self.action_space.n] * self.max_length * 2)

        # Nulstil miljøet
        self.reset()

    def reset(self):
        """Nulstiller miljøet til starttilstand"""
        self.melody = []
        self.steps = 0
        # State er en kombination af toner og intervaller (med padding)
        return np.zeros(self.max_length * 2, dtype=int), {}

    def step(self, action):
        """Udfører en handling og returnerer den nye state, reward og om episoden er færdig"""
        # Konverter handling til en faktisk MIDI-tone
        note = self.min_note + action
        reward = 0

        # === STRAF: Undgå store spring først ===
        if self.melody:
            jump = abs(note - self.melody[-1])
            if 5 <= jump < 8:
                reward -= 5.0  # Øget straf for mellemstore spring
            elif 8 <= jump < 12:
                reward -= 10.0 # Øget straf for store spring
            elif jump >= 12:
                reward -= 15.0 # Øget straf for meget store spring

        # Straf for gentagne store spring
        if len(self.melody) >= 3:
            interval_1 = abs(self.melody[-1] - self.melody[-2])
            interval_2 = abs(self.melody[-2] - self.melody[-3])
            if interval_1 > 4 and interval_2 > 4:
                reward -= 5.0  # Ekstra straf for gentagne store spring

        # Straf for gentagelser af samme tone
        if len(self.melody) > 3 and note == self.melody[-1]:
            reward -= 5.0

        # Straf for lange gentagelser (tre på stribe)
        if len(self.melody) >= 3 and self.melody[-1] == self.melody[-2] == self.melody[-3]:
            reward -= 5.0

        # Straf for gentagelser over 4 toner
        if len(self.melody) >= 4 and self.melody[-1] == self.melody[-2] == self.melody[-3] == self.melody[-4]:
            reward -= 3.0

        # === BELØNNINGER: Variation og flow ===
        if len(self.melody) > 5 and note not in self.melody[-5:]:
            reward += 2.0

        if len(self.melody) >= 3:
            last_three = self.melody[-3:]
            if last_three == sorted(last_three) or last_three == sorted(last_three, reverse=True):
                reward += 5.0

        if self.melody and abs(note - self.melody[-1]) in [1, 2, 3, 4]:
            reward += 3.0

        # 🔹 Belønning for 4 sammenhængende små trin
        if len(self.melody) >= 4:
            last_four_steps = [abs(self.melody[i] - self.melody[i-1]) for i in range(-3, 0)]
            if all(step in [1, 2] for step in last_four_steps):
                reward += 5.0

        if len(self.melody) >= 4:
            last_four = self.melody[-4:]
            if last_four == sorted(last_four) or last_four == sorted(last_four, reverse=True):
                reward += 10.0

        # === BELØNNINGER: Reference-melodi ===
        if self.reference_melody and len(self.melody) < len(self.reference_melody):
            ref_note = self.reference_melody[len(self.melody)]
            if note == ref_note:
                reward += 10.0
            elif abs(note - ref_note) <= 2:
                reward += 5.0

        # Beløn for variation i toner og intervaller
        unique_notes = len(set(self.melody))
        reward += unique_notes * 1.0

        if len(self.melody) >= 3:
            intervals = [self.melody[i] - self.melody[i-1] for i in range(1, len(self.melody))]
            unique_intervals = len(set(intervals))
            reward += unique_intervals * 1.5

        # Opdater miljøets tilstand
        self.melody.append(note)
        self.steps += 1

        # Tjek om episoden er færdig
        done = self.steps >= self.max_length

        # Skab state som kombination af toner og intervaller
        obs = self.melody + [0] * (self.max_length - len(self.melody))
        intervals = [self.melody[i] - self.melody[i-1] for i in range(1, len(self.melody))]
        obs_intervals = intervals + [0] * (self.max_length - len(intervals))
        full_state = np.concatenate((obs, obs_intervals))

        return full_state, reward, done, {}

    def render(self, mode='human'):
        """Printer den nuværende melodi og intervaller"""
        print("Melody:", self.melody)
        if len(self.melody) > 1:
            intervals = [self.melody[i] - self.melody[i-1] for i in range(1, len(self.melody))]
            print("Intervals:", intervals)

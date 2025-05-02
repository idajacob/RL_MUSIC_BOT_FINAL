import gym
import numpy as np
from gym import spaces
from utils.midi_tools import extract_intervals_from_midi

C_MAJOR = [n for n in range(48, 85) if (n % 12) in [0, 2, 4, 5, 7, 9, 11]] #<-- C-dur skala

class MelodyEnv(gym.Env):
    def __init__(self, reference_melody=None):
        super(MelodyEnv, self).__init__()
        self.max_length = 16
        self.min_note = 48  # C3
        self.max_note = 84  # C6

        self.reference_melody = reference_melody
        if reference_melody:
            self.reference_intervals = [reference_melody[i+1] - reference_melody[i] for i in range(len(reference_melody)-1)]
        else:
            self.reference_intervals = extract_intervals_from_midi("data/input_midi/DEB_CLAI.MID")[:self.max_length - 1]

        self.action_space = spaces.Discrete(self.max_note - self.min_note + 1)
        self.observation_space = spaces.MultiDiscrete([self.action_space.n] * self.max_length)

        # Belønningsvægtning – gør det let at justere
        self.weights = {
            "in_scale": 1.0,
            "repeat_penalty": -1.0,
            "jump_penalty": -1.0,
            "interval_match": 1.0,
            "interval_near": 0.5,
            "melody_match": 2.0,
            "melody_near": 1.0,
            "same_interval_penalty": -1.0
        }

        self.reset()

    def reset(self):
        self.melody = []
        self.steps = 0
        return np.zeros(self.max_length, dtype=int), {}

    def step(self, action):
        note = self.min_note + action
        reward = 0

        # ✅ Beløn hvis noten er i skala
        if note in C_MAJOR:
            reward += self.weights["in_scale"]

        # 🚫 Straf gentagelser
        if self.melody and note == self.melody[-1]:
            reward += self.weights["repeat_penalty"]

        # 🚫 Straf store spring
        if self.melody and abs(note - self.melody[-1]) > 12:
            reward += self.weights["jump_penalty"]

        # 🔁 Beløn intervaller der matcher reference-intervaller
        if self.reference_intervals and len(self.melody) >= 2 and len(self.melody) - 2 < len(self.reference_intervals):
            interval = self.melody[-1] - self.melody[-2]
            ref_interval = self.reference_intervals[len(self.melody) - 2]
            if interval == ref_interval:
                reward += self.weights["interval_match"]
            elif abs(interval - ref_interval) <= 2:
                reward += self.weights["interval_near"]

        # 🚫 Straf gentagelse af samme interval
        if len(self.melody) >= 2:
            i1 = self.melody[-1] - self.melody[-2]
            i2 = note - self.melody[-1]
            if i1 == i2:
                reward += self.weights["same_interval_penalty"]

        # 🎯 Match direkte mod reference-melodi
        if self.reference_melody and len(self.melody) < len(self.reference_melody):
            ref_note = self.reference_melody[len(self.melody)]
            if note == ref_note:
                reward += self.weights["melody_match"]
            elif abs(note - ref_note) <= 2:
                reward += self.weights["melody_near"]

        # Opdater tilstand
        self.melody.append(note)
        self.steps += 1

        done = self.steps >= self.max_length
        obs = self.melody + [0] * (self.max_length - len(self.melody))
        return np.array(obs), reward, done, {}

    def render(self, mode='human'):
        print("Melody:", self.melody)

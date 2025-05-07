import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym
import numpy as np
from gym import spaces
from utils.midi_tools import extract_intervals_from_midi

# MIDI-noter for C-dur skala fra C3 til C6
C_MAJOR = [n for n in range(48, 85) if (n % 12) in [0, 2, 4, 5, 7, 9, 11]]  # C-dur skala

class MelodyEnv(gym.Env):
    def __init__(self, reference_melody=None):
        super(MelodyEnv, self).__init__()
        self.max_length = 64
        self.min_note = 48  # C3
        self.max_note = 84  # C6
        self.reference_melody = reference_melody

        if reference_melody:
            self.reference_intervals = [reference_melody[i+1] - reference_melody[i] for i in range(len(reference_melody)-1)]
        else:
            self.reference_intervals = extract_intervals_from_midi("data/input_midi/chpn_op7_1.MID")[:self.max_length - 1]

        self.action_space = spaces.Discrete(self.max_note - self.min_note + 1)
        self.observation_space = spaces.MultiDiscrete([self.action_space.n] * self.max_length)

        self.weights = {
            "in_scale": 10.0,
            "repeat_penalty": -10.0,
            "jump_penalty": -5.0,
            "interval_variation": 2.0,
            "interval_match": 10.0,
            "interval_near": 1.0,
            "melody_match": 10.0,
            "melody_near": 5.0,
            "octave_bonus": 1.0,
            "same_interval_penalty": -10.0,
            "long_repeat_penalty": -10.0
        }

        self.reset()

    def reset(self):
        self.melody = []
        self.steps = 0
        return np.zeros(self.max_length, dtype=int), {}

    def step(self, action):
        note = self.min_note + action
        reward = 0

        # Beløn hvis note er i C-dur
        if note in C_MAJOR:
            reward += self.weights["in_scale"]

        # Straf gentagelser
        if self.melody and note == self.melody[-1]:
            reward += self.weights["repeat_penalty"]
        
        # Straf lange gentagelser
        if len(self.melody) >= 2 and note == self.melody[-1] == self.melody[-2]:
            reward += self.weights["long_repeat_penalty"]


        # Straf store spring (> oktav)
        if self.melody and abs(note - self.melody[-1]) > 12:
            reward += self.weights["jump_penalty"]

        # Beløn variation i intervaller
        if len(self.melody) >= 2:
            interval1 = self.melody[-1] - self.melody[-2]
            interval2 = note - self.melody[-1]
            if interval1 == interval2:
                reward += self.weights["same_interval_penalty"]
            else:
                reward += self.weights["interval_variation"]

        # Beløn brug af flere oktaver
        octaves_used = set([n // 12 for n in self.melody + [note]])
        reward += len(octaves_used) * self.weights["octave_bonus"]

        # Match mod reference-intervaller
        if self.reference_intervals and len(self.melody) >= 2:
            generated_interval = self.melody[-1] - self.melody[-2]
            ref_interval = self.reference_intervals[len(self.melody) - 2]
            if generated_interval == ref_interval:
                reward += self.weights["interval_match"]
            elif abs(generated_interval - ref_interval) <= 2:
                reward += self.weights["interval_near"]

        # Match mod reference-melodi
        if self.reference_melody and len(self.melody) < len(self.reference_melody):
            ref_note = self.reference_melody[len(self.melody)]
            if note == ref_note:
                reward += self.weights["melody_match"]
            elif abs(note - ref_note) <= 2:
                reward += self.weights["melody_near"]

        self.melody.append(note)
        self.steps += 1

        done = self.steps >= self.max_length
        obs = self.melody + [0] * (self.max_length - len(self.melody))
        return np.array(obs), reward, done, {}

    def render(self, mode='human'):
        print("Melody:", self.melody)


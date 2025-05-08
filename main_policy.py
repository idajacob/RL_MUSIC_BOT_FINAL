import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
import pretty_midi

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.melody_env import MelodyEnv, FULL_RANGE, RHYTHM_VALUES

from utils.midi_tools import save_melody_as_midi, extract_melody_from_midi
from agent.policy_agent import PolicyAgent
from plots.plot_utils import (
    plot_training_progress, plot_melody, plot_policy_rewards,
    plot_melody_comparison, plot_octave_distribution,
    plot_moving_average, plot_tone_histogram, plot_average_probabilities
)

# === Indlæs reference-melodi fra MIDI ===
# Reference-melodien bruges som mål for træningen
reference_melody = extract_melody_from_midi('data/input_midi/Piano_chopin.mid')[:32]
print('Reference-melodi:', reference_melody)

# === Initialisér miljø og agent ===
# Miljøet (env) håndterer feedback og belønninger, mens agenten (agent) vælger toner
env = MelodyEnv(reference_melody=reference_melody)
agent = PolicyAgent(state_size=env.max_length * 2, action_size=len(FULL_RANGE) + len(RHYTHM_VALUES))

# Historik for træning
avg_reward_history = []
window = 20
n_episodes = 2000
reward_history = []
best_reward = float('-inf')
best_melody = []

# Early Stopping parametre
patience = 500
no_improve_counter = 0

# === Træningsloop ===
# Gennemfører en række episoder, hvor agenten lærer at generere toner
for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    agent.tone_log = []
    agent.rhythm_log = []
    agent.prob_log = []

    # === Episode loop ===
    # Agenten vælger handlinger indtil episoden er færdig
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.store_transition(obs, action, reward)
        obs = next_obs
        total_reward += reward

    # Opdater policy-netværket efter hver episode
    agent.update()
    reward_history.append(total_reward)

    # Gem den bedste melodi og tjek for Early Stopping
    if total_reward > best_reward:
        best_reward = total_reward
        best_melody = env.melody[:]
        no_improve_counter = 0
    else:
        no_improve_counter += 1

    if no_improve_counter > patience:
        print(f'Early stopping triggered at episode {episode}')
        break

    # Udskriv status hver 50. episode
    if episode % 50 == 0:
        print(f'Episode {episode}: total reward = {total_reward:.2f} - Epsilon: {agent.epsilon:.4f}')

    if len(reward_history) >= window:
        moving_avg = np.mean(reward_history[-window:])
        avg_reward_history.append(moving_avg)

# === Gem og afspil bedste melodi ===
midi_path = 'data/output_midi/generated_by_policy_agent.mid'
save_melody_as_midi(best_melody, midi_path)
print('Bedste melodi gemt i:', midi_path)

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(midi_path)
pygame.mixer.music.play()
print('▶️ Afspiller MIDI...')
while pygame.mixer.music.get_busy():
    time.sleep(0.1)
print('✅ Afspilning færdig.')

# === Visualisering af resultater ===
plot_training_progress(reward_history)
plot_melody(best_melody, title='Bedste genererede melodi', save_path='plots/melody_plot.png')
plot_policy_rewards(reward_history, best_reward)
plot_melody_comparison(reference_melody, best_melody, save_path='plots/melody_comparison.png')
plot_octave_distribution(best_melody, save_path='plots/octave_distribution.png')
plot_moving_average(avg_reward_history, window)
plot_tone_histogram(agent.tone_log, env.action_space.nvec[0], save_path='plots/tone_histogram.png')
plot_average_probabilities(agent.prob_log, save_path='plots/probability_distribution.png')
""
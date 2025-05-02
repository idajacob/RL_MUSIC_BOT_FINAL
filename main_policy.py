import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
import pretty_midi

from env.melody_env import MelodyEnv
from utils.midi_tools import save_melody_as_midi, extract_melody_from_midi
from agent import policy_agent
from plots.plot_utils import plot_training_progress, plot_melody, plot_policy_rewards, plot_melody_comparison, plot_octave_distribution

# === Indlæs reference-melodi fra MIDI ===
# Denne bruges som "mål", som agenten forsøger at matche
reference_melody = extract_melody_from_midi("data/input_midi/DEB_CLAI.mid")[:16]
print("Reference-melodi:", reference_melody)

# === Initialisér miljø og agent ===
# Miljøet giver feedback (belønning), agenten vælger toner
env = MelodyEnv(reference_melody=reference_melody)
agent = policy_agent.PolicyAgent(state_size=env.max_length, action_size=env.action_space.n)

# === Træningsparametre ===
n_episodes = 500
reward_history = []
best_reward = float('-inf')
best_melody = []

# === Hovedløkke: Træn over flere episoder ===
for episode in range(n_episodes):
    obs, _ = env.reset()  # Nulstil miljø og få start-observation
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(obs)  # Agent vælger handling baseret på nuværende state
        next_obs, reward, done, _ = env.step(action)  # Udfør handling i miljøet

        agent.store_transition(obs, reward)  # Gem overgang til senere opdatering
        obs = next_obs
        total_reward += reward

    agent.update()  # Efter hver episode opdaterer vi policy-netværket

    reward_history.append(total_reward)
    if total_reward > best_reward:
        best_reward = total_reward
        best_melody = env.melody[:]

    if episode % 50 == 0:
        print(f"Episode {episode}: total reward = {total_reward:.2f}")

# === Gem og afspil bedste melodi ===
midi_path = "data/output_midi/generated_by_policy_agent.mid"
save_melody_as_midi(best_melody, midi_path)
print("Bedste melodi gemt i:", midi_path)

# Brug pygame til at afspille MIDI direkte
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(midi_path)
pygame.mixer.music.play()
print("▶️ Afspiller MIDI...")
while pygame.mixer.music.get_busy():
    time.sleep(0.1)
print("✅ Afspilning færdig.")


########## VISUALISERINGER ##########

plot_training_progress(reward_history)

plot_melody(best_melody, title="Bedste genererede melodi", save_path="plots/melody_plot.png")

plot_policy_rewards(reward_history, best_reward)

plot_melody_comparison(reference_melody, best_melody, save_path="plots/melody_comparison.png")

plot_octave_distribution(best_melody, save_path="plots/octave_distribution.png")

import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
import pretty_midi
import csv  # Til log af handlinger og sandsynligheder

from env.melody_env import MelodyEnv
from utils.midi_tools import save_melody_as_midi, extract_melody_from_midi
from agent.policy_agent import PolicyAgent
from plots.plot_utils import plot_training_progress, plot_melody, plot_policy_rewards, plot_melody_comparison, plot_octave_distribution, plot_moving_average, plot_tone_histogram, plot_average_probabilities


# === Indlæs reference-melodi fra MIDI ===
# Denne bruges som "mål", som agenten forsøger at matche
reference_melody = extract_melody_from_midi("data/input_midi/chpn_op7_1.mid")[:16]
print("Reference-melodi:", reference_melody)

# === Initialisér miljø og agent ===
# Miljøet giver feedback (belønning), agenten vælger toner
env = MelodyEnv(reference_melody=reference_melody)
agent = PolicyAgent(state_size=env.max_length, action_size=env.action_space.n)

avg_reward_history = []
window = 20 # antal episoder der medregnes (gennemsnit)

# === Træningsparametre ===
n_episodes = 1000
reward_history = []
best_reward = float('-inf')
best_melody = []

# === Hovedløkke: Træn over flere episoder ===
for episode in range(n_episodes):
    obs, _ = env.reset()  # Nulstil miljø og få start-observation
    done = False
    total_reward = 0

    # Ryd log for denne episode
    agent.tone_log = []
    agent.prob_log = []
    agent.probs_log = []

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

    if len(reward_history) >= window:
        moving_avg = np.mean(reward_history[-window:])
        avg_reward_history.append(moving_avg)

# === Gem og afspil bedste melodi ===
midi_path = "data/output_midi/generated_by_policy_agent.mid"
save_melody_as_midi(best_melody, midi_path)
print("Bedste melodi gemt i:", midi_path)

agent.save_action_log("logs/tone_log.csv")

# Brug pygame til at afspille MIDI direkte
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(midi_path)
pygame.mixer.music.play()
print("▶️ Afspiller MIDI...")
while pygame.mixer.music.get_busy():
    time.sleep(0.1)
print("✅ Afspilning færdig.")

# === Gem logfil med valgte toner og log-sandsynligheder ===
def save_action_log(path):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Tone (action index)', 'Log-sandsynlighed'])
        for tone, prob in zip(agent.tone_log, agent.prob_log):
            writer.writerow([tone, prob])


########## VISUALISERINGER ##########

plot_training_progress(reward_history)

plot_melody(best_melody, title="Bedste genererede melodi", save_path="plots/melody_plot.png")

plot_policy_rewards(reward_history, best_reward)

plot_melody_comparison(reference_melody, best_melody, save_path="plots/melody_comparison.png")

plot_octave_distribution(best_melody, save_path="plots/octave_distribution.png")

plot_moving_average(avg_reward_history, window)

plot_tone_histogram(agent.tone_log, env.action_space.n, save_path="plots/tone_histogram.png")

# === Visualisér gennemsnitlig sandsynlighed for hver handling ===
plot_average_probabilities(agent.probs_log, save_path="plots/probability_distribution.png")

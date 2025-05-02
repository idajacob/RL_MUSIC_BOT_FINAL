import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
from env.melody_env import MelodyEnv
from utils.midi_tools import save_melody_as_midi, extract_melody_from_midi
from agent.rl_agent import QLearningAgent

# Indlæs reference-melodi
reference_melody = extract_melody_from_midi("data/input_midi/DEB_CLAI.mid")[:8]
print("Reference-melodi:", reference_melody)

# Initialisér miljø og agent
env = MelodyEnv(reference_melody=reference_melody)
agent = QLearningAgent(
    state_size=env.max_length,
    action_size=env.action_space.n,
    learning_rate=0.1,
    discount=0.95,
    epsilon=1.0,
    epsilon_decay=0.99,
    epsilon_min=0.1
)

n_episodes = 500
reward_history = []
epsilon_history = []
best_reward = float('-inf')
best_melody = []
early_stop_count = 0
patience = 50

for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward

    reward_history.append(total_reward)
    epsilon_history.append(agent.epsilon)

    if total_reward > best_reward:
        best_reward = total_reward
        best_melody = env.melody[:]
        early_stop_count = 0
    else:
        early_stop_count += 1

    print(f"Episode {episode}: reward = {total_reward}, epsilon = {agent.epsilon:.3f}")

    if early_stop_count >= patience:
        print(f"Tidlig stop aktiveret efter {episode + 1} episoder")
        break

# Gem den bedste melodi som MIDI
midi_path = "data/output_midi/generated_by_agent.mid"
save_melody_as_midi(best_melody, midi_path)
print("Bedste melodi gemt i:", midi_path)

# Afspilning af MIDI med pygame
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(midi_path)
pygame.mixer.music.play()
print("▶️ Afspiller MIDI...")
while pygame.mixer.music.get_busy():
    time.sleep(0.1)
print("✅ Afspilning færdig.")


# VISUALISERINGER

# total reward
plt.figure(figsize=(10, 4))
plt.plot(reward_history, label="Reward")
plt.axhline(best_reward, color="red", linestyle="--", label="Bedste reward")
plt.title("Reward pr. episode")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/reward_plot.png")
plt.show()

# epsilon
plt.figure(figsize=(10, 4))
plt.plot(epsilon_history)
plt.title("Epsilon pr. episode")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/epsilon_plot.png")
plt.show()

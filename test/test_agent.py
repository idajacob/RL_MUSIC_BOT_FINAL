from agent.rl_agent import QLearningAgent
from env.melody_env import MelodyEnv
from utils.midi_tools import extract_melody_from_midi, save_melody_as_midi
import numpy as np

# Indlæs reference fra tidligere optagelse
reference = extract_melody_from_midi("data/output_midi/user_input.mid")

# Initialisér miljø og agent
env = MelodyEnv(reference_melody=reference)
agent = QLearningAgent(state_size=env.max_length, action_size=env.action_space.n)

# Indlæs gemt Q-table
agent.load("data/output_midi/q_table.npy")

# Afspil én episode med den lærte agent
state, _ = env.reset()
done = False

while not done:
    action = agent.get_action(state)
    state, reward, done, _, _ = env.step(action)

# Vis og gem output
env.render()
save_melody_as_midi(env.melody, "data/output_midi/generated_from_saved_agent.mid")
print("Melodi genereret fra gemt agent gemt som: generated_from_saved_agent.mid")

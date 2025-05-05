import matplotlib.pyplot as plt
import numpy as np

# Visualiser træningsprogression
# Viser reward over episoder

def plot_training_progress(reward_history):
    plt.figure(figsize=(10, 4))
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total reward pr. episode")
    plt.savefig("plots/training_progress.png")
    plt.show()


# Visualiser bedste melodi
# Bruges til at vise den bedst genererede sekvens

def plot_melody(melody, title, save_path=None):
    plt.figure(figsize=(10, 2))
    plt.plot(melody, marker='o')
    plt.title(title)
    plt.xlabel("Takt (position i sekvens)")
    plt.ylabel("MIDI tonehøjde")
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Visualiser policy rewards (total reward pr. episode + bedste reward)

def plot_policy_rewards(reward_history, best_reward):
    plt.figure(figsize=(10, 4))
    plt.plot(reward_history, label="Total Reward")
    plt.axhline(best_reward, color='red', linestyle='--', label="Bedste reward")
    plt.xlabel("Episode")
    plt.ylabel("Belønning")
    plt.title("Reward pr. episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/policy_rewards.png")
    plt.show()


# Sammenlign reference og genereret melodi

def plot_melody_comparison(reference, generated, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(reference, label="Reference-melodi", marker='o')
    plt.plot(generated, label="Agentens melodi", marker='x')
    plt.title("Sammenligning af melodier")
    plt.xlabel("Tidssstep (note-position)")
    plt.ylabel("MIDI-toneværdi")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Oktavfordeling – hvor mange toner ligger i hvilke oktaver

def plot_octave_distribution(melody, save_path=None):
    octaves = [note // 12 for note in melody]  # MIDI-note divideret med 12 giver oktav
    plt.figure(figsize=(8, 4))
    plt.hist(octaves, bins=range(min(octaves), max(octaves)+2), align='left', rwidth=0.8)
    plt.xlabel("Oktav")
    plt.ylabel("Antal toner")
    plt.title("Oktavfordeling i agentens melodi")
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Glidende gennemsnit af reward

def plot_moving_average(avg_reward_history, window, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(avg_reward_history)
    plt.title("Gennemsnitlig reward (moving avg)")
    plt.xlabel(f"Episode (fra episode {window})")
    plt.ylabel("Gns. reward")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Histogram over hvilke handlinger (toner) der blev valgt

def plot_tone_histogram(tone_log, action_size, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.hist(tone_log, bins=range(action_size + 1), align='left', rwidth=0.8)
    plt.xlabel("Handling (MIDI tone index)")
    plt.ylabel("Antal gange valgt")
    plt.title("Fordeling af valgte toner")
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Visualiser gennemsnitlig sandsynlighed for hver handling (tone)
# Input: probs_log = liste af arrays, hvor hver array indeholder sandsynligheder for handlinger i ét step
# Output: søjlediagram med gennemsnit for hver handling

def plot_average_probabilities(probs_log, save_path=None):
    if not probs_log:
        print("⚠️ probs_log er tom – kan ikke plotte")
        return

    # Konverter til numpy array: (n_steps, action_size)
    probs_array = np.array(probs_log)
    avg_probs = np.mean(probs_array, axis=0)

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(avg_probs)), avg_probs)
    plt.xlabel("Handling (MIDI tone index)")
    plt.ylabel("Gns. sandsynlighed")
    plt.title("Gennemsnitlig valg-sandsynlighed for hver handling")
    if save_path:
        plt.savefig(save_path)
    plt.show()

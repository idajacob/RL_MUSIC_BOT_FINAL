import matplotlib.pyplot as plt

def plot_training_progress(rewards, save_path="plots/reward_policy_plot.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(rewards)
    plt.title("Total reward pr. episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_melody(notes, title="Melodi", save_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 2))
    plt.plot(range(len(notes)), notes, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Takt (position i sekvens)")
    plt.ylabel("MIDI tonehøjde")
    plt.ylim(55, 80)  # typisk område for C-dur
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_policy_rewards(reward_history, best_reward, save_path="plots/policy_reward_plot.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(reward_history, label="Total Reward")
    plt.axhline(best_reward, color="red", linestyle="--", label="Bedste reward")
    plt.title("Reward pr. episode")
    plt.xlabel("Episode")
    plt.ylabel("Belønning")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_melody_comparison(reference, generated, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(reference, label="Reference-melodi", marker="o")
    plt.plot(generated, label="Agentens melodi", marker="x")
    plt.title("Sammenligning af melodier")
    plt.xlabel("Tidsstep (note-position)")
    plt.ylabel("MIDI-toneværdi")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_octave_distribution(melody, save_path=None):
    octaves = [(note // 12) for note in melody]  # MIDI tone 60 = oktav 5 (C4)
    plt.figure(figsize=(8, 4))
    plt.hist(octaves, bins=range(min(octaves), max(octaves) + 2), align='left', rwidth=0.8)
    plt.title("Oktavfordeling i agentens melodi")
    plt.xlabel("Oktav")
    plt.ylabel("Antal toner")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os

# Policy Network er et neuralt netværk der fungerer som agentens strategi ("policy") i reinforcement learning.
# Det tager observationer (tilstande) som input og giver en sandsynlighedsfordeling over mulige handlinger.
# Agenten bruger dette netværk til at beslutte, hvilke handlinger der skal udføres.
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),   # Lineært lag: beregner en vægtet sum af input og tilføjer en bias
            nn.ReLU(),                   # ReLU: Aktiveringsfunktion der sætter negative værdier til 0 (introducerer ikke-linearitet)
            nn.Linear(64, output_size),  # Andet lineære lag: transformerer skjulte neuroner til antallet af mulige handlinger
            nn.Softmax(dim=-1)           # Softmax: konverterer output til en sandsynlighedsfordeling over handlinger
        )

    def forward(self, x):
        return self.model(x)

# PolicyAgent der bruger Policy Gradient algoritmen (REINFORCE)
class PolicyAgent:
    def __init__(self, state_size, action_size, min_note=36, lr=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.min_note = min_note  # Minimum MIDI-tone

        self.policy_net = PolicyNetwork(state_size, action_size)  # Initialiser netværket
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)  # Optimeringsalgoritme
        self.gamma = 0.99  # Diskonteringsfaktor for fremtidige belønninger

        # Gemmer transitions for en episode
        self.states = []
        self.actions = []
        self.rewards = []

        # Logning til analyse og visualisering
        self.tone_log = []          # Liste over valgte toner (action index)
        self.prob_log = []          # Liste over hele sandsynlighedsfordelingen pr. step

    def get_action(self, state):
        # Konverterer til PyTorch tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Tilføj batch-dimension
        probs = self.policy_net(state_tensor)  # Beregn sandsynligheder for handlinger
        dist = torch.distributions.Categorical(probs)  # Opret en sandsynlighedsfordeling
        action = dist.sample()  # Sample en handling baseret på sandsynlighed (stochastic valg)

        # Gem overgang og sandsynligheder
        self.states.append(state_tensor)
        self.actions.append(dist.log_prob(action))
        self.rewards.append(0)  # Midlertidig reward, opdateres i step

        # Log til analyse
        self.tone_log.append(action.item())
        self.prob_log.append(probs.squeeze().detach().numpy())

        return action.item()

    def store_transition(self, state, reward):
        # Gemmer reward til nuværende step
        self.rewards[-1] = reward

    def update(self):
        # === Trin 1: Beregn return-værdier ===
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # === Trin 2: Normalisér returns ===
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # === Trin 3: Beregn policy loss ===
        loss = 0
        for log_prob, G in zip(self.actions, returns):
            loss -= log_prob * G  # Negativt fordi vi vil maksimere reward

        # === Trin 4: Tilbageløb og opdater netværket ===
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # === Trin 5: Ryd hukommelse (kun transitions, ikke log) ===
        self.states = []
        self.actions = []
        self.rewards = []

    def save_action_log(self, filepath):
        # Sørg for at mappen eksisterer
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Skriv log til CSV
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Tone (action index)", "Log-sandsynlighed"])
            for tone, prob in zip(self.tone_log, self.prob_log):
                # Tjek at tone er en gyldig indeks i prob-arrayet
                if 0 <= tone < len(prob):
                    writer.writerow([tone, np.log(prob[tone])])
                else:
                    writer.writerow([tone, "NaN"])  # Hvis noget er gået galt

__all__ = ["PolicyAgent"]

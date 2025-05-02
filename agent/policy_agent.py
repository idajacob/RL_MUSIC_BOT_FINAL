import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, state_size, action_size, lr=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = PolicyNetwork(state_size, action_size)  # Initialiser netværket
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)  # Optimeringsalgoritme
        self.gamma = 0.99  # Diskonteringsfaktor for fremtidige belønninger

        # Gemmer transitions for en episode
        self.states = []
        self.actions = []
        self.rewards = []

    def get_action(self, state):
        # Konverterer til PyTorch tensor
        state = torch.FloatTensor(state).unsqueeze(0)  # Tilføj batch-dimension
        probs = self.policy_net(state)  # Beregn sandsynligheder for handlinger
        dist = torch.distributions.Categorical(probs)  # Opret en sandsynlighedsfordeling
        action = dist.sample()  # Sample en handling baseret på sandsynlighed (stochastic valg)

        # Bemærk forskellen: Vi kunne også bruge probs.argmax() for altid at vælge den mest sandsynlige handling (det deterministiske valg).
        # Men i RL er exploration vigtig - ved at sample stochastisk giver vi agenten mulighed for at prøve nye handlinger
        # og derved opdage bedre strategier over tid.

        self.actions.append(dist.log_prob(action))  # Gem log-sandsynligheden af den valgte handling
        return action.item()

    def store_transition(self, state, reward):
        self.states.append(state)  # Gem tilstand
        self.rewards.append(reward)  # Gem belønning

    def update(self):
        # === Trin 1: Beregn return-værdier ===
        # Return er den diskonterede sum af fremtidige belønninger fra hvert tidspunkt i episoden
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # === Trin 2: Normalisér returns ===
        # Dette hjælper med at stabilisere træningen ved at fjerne skala og centere værdierne
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # === Trin 3: Beregn policy loss ===
        # For hver handling multipliceres log-sandsynligheden med den return-værdi den førte til
        # Dette forstærker gode handlinger og dæmper dårlige
        loss = 0
        for log_prob, G in zip(self.actions, returns):
            loss -= log_prob * G  # Negativt fordi vi minimerer loss, men vil maksimere belønning

        # === Trin 4: Opdater netværket ===
        # Tilbageløb og optimering
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # === Trin 5: Ryd hukommelsen ===
        # Så agenten er klar til næste episode
        self.states = []
        self.actions = []
        self.rewards = []


__all__ = ["PolicyAgent"]

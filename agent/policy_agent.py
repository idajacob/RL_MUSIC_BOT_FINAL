import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv

# === Policy Network ===
# Dette neurale netværk forsøger at forudsige den bedste næste handling (tonevalg) givet en nuværende state.
# Netværket består af tre lag:
# 1. Input lag, der modtager state (melodi + intervaller)
# 2. To skjulte lag med ReLU-aktiveringer for ikke-linearitet
# 3. Output lag med Softmax for at beregne sandsynligheden for hver handling

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        
        # Definition af netværkets lag
        self.model = nn.Sequential(
            nn.Linear(state_size, 256),  # Input layer -> Hidden layer 1
            nn.ReLU(),
            nn.Linear(256, 128),         # Hidden layer 1 -> Hidden layer 2
            nn.ReLU(),
            nn.Linear(128, action_size), # Hidden layer 2 -> Output layer
            nn.Softmax(dim=-1)           # Softmax for at få sandsynligheder
        )

    # Fremadrettet propagation af state gennem netværket
    def forward(self, state):
        return self.model(state)


# === Policy Agent ===
# Agenten interagerer med miljøet og bruger PolicyNetwork til at vælge handlinger
# Den gemmer også transitions (state, action, reward) for at kunne opdatere policy'en senere
class PolicyAgent:
    def __init__(self, state_size, action_size, lr=0.005):
        # Initialisering af agentens parametre og netværk
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = PolicyNetwork(state_size, action_size)
        
        # Optimizer til gradient descent
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Diskonteringsfaktor for fremtidige belønninger
        self.gamma = 0.99

        # Lager for transitions
        self.states = []
        self.actions = []
        self.rewards = []
        
        # Logs til analyse og debugging
        self.tone_log = []
        self.prob_log = []

        # Exploration parametre
        self.epsilon = 0.1          # Startværdi for exploration
        self.epsilon_min = 0.01     # Minimum exploration
        self.epsilon_decay = 0.995  # Hvor hurtigt exploration falder

    # === Vælg handling baseret på nuværende state ===
    # Agenten modtager state fra miljøet, beregner sandsynlighederne for hver handling
    # og vælger en handling baseret på en stokastisk prøve
    def get_action(self, state):
        # Konverterer til PyTorch tensor
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        dist = torch.distributions.Categorical(probs)

        # Exploration vs. exploitation med epsilon-decay
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            action = dist.sample().item()

        # Decay epsilon for hver handling
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Gem logs til analyse
        self.tone_log.append(action)
        self.prob_log.append(probs.squeeze(0).detach().numpy())

        return action

    # === Gem overgang (transition) ===
    # Hver overgang (state, action, reward) gemmes til senere brug i policy-opdateringen
    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # === Opdater policy-netværket ===
    # Når en episode er færdig, opdateres policy'en baseret på de oplevede belønninger
    def update(self):
        R = 0
        returns = []

        # Baseline-værdi (minimum reward)
        BASELINE = -500

        # Beregn diskonterede fremtidige belønninger baglæns
        for reward in reversed(self.rewards):
            R = max(reward, BASELINE) + self.gamma * R
            returns.insert(0, R)

        # Normaliser de beregnede returns for mere stabil læring
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Konverter lister til tensorer
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)

        # Kør states gennem policy-netværket og beregn policy gradient
        probs = self.policy_net(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        # Loss-funktion for policy gradient
        loss = -(log_probs * returns).mean()

        # Optimeringsstep for at opdatere netværket
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Nulstil transitions efter opdatering
        self.states = []
        self.actions = []
        self.rewards = []


    # === Gem tone log til CSV-fil ===
    # Logger de valgte toner og sandsynligheder til en CSV-fil for analyse
    def save_action_log(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Tone (action index)", "Log-sandsynlighed"])
            
            for tone, probs in zip(self.tone_log, self.prob_log):
                if 0 <= tone < len(probs):
                    writer.writerow([tone, np.log(probs[tone])])
                else:
                    writer.writerow([tone, "NaN"])

__all__ = ["PolicyAgent"]

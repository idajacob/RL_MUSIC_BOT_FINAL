import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.melody_env import FULL_RANGE, RHYTHM_VALUES

# === Policy Network ===
# Dette neurale netværk forsøger at forudsige den bedste næste handling (tonevalg og rytmevalg) givet en nuværende state.
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
        self.rhythm_log = []
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

        # === Valider dimensioner ===
        total_length = len(FULL_RANGE) + len(RHYTHM_VALUES)
        if probs.size(1) != total_length:
            print(f"[FEJL] Dimensionen af policy-netværkets output ({probs.size(1)}) matcher ikke forventet længde ({total_length})!")
            print("Fallback til ens sandsynligheder.")
            tone_probs = torch.ones((1, len(FULL_RANGE))) / len(FULL_RANGE)
            rhythm_probs = torch.ones((1, len(RHYTHM_VALUES))) / len(RHYTHM_VALUES)
        else:
            tone_probs = probs[:, :len(FULL_RANGE)]   # Toner
            rhythm_probs = probs[:, len(FULL_RANGE):]  # Rytmer
        
        # Sample handlinger baseret på sandsynlighederne
        try:
            tone_dist = torch.distributions.Categorical(tone_probs)
            rhythm_dist = torch.distributions.Categorical(rhythm_probs)
            tone_action = tone_dist.sample().item()
            rhythm_action = rhythm_dist.sample().item()
        except Exception as e:
            print(f"[FEJL] Kunne ikke sample handlinger: {e}")
            tone_action = 0
            rhythm_action = 0

        # Exploration vs. exploitation med epsilon-decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Gem logs til analyse
        self.tone_log.append(tone_action)
        self.rhythm_log.append(rhythm_action)
        self.prob_log.append((tone_probs.squeeze(0).detach().numpy(), rhythm_probs.squeeze(0).detach().numpy()))

        # === Logging af handlinger ===
        print(f"Valgt tone: {tone_action}, Valgt rytme: {rhythm_action} ({RHYTHM_VALUES[rhythm_action]})")

        return tone_action, rhythm_action

    # === Gem overgang (transition) ===
    # Hver overgang (state, action, reward) gemmes til senere brug i policy-opdateringen
    def store_transition(self, state, action, reward):
        """Gemmer transitionen til opdatering af policy"""
        if state is None or action is None or reward is None:
            print("[FEJL] En af parametrene til 'store_transition' er None.")
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

    # === Opdater policy-netværket ===
    # Denne metode opdaterer agentens netværk baseret på de gemte transitions
    def update(self):
        R = 0
        returns = []

        # Gå baglæns gennem rewards og beregn diskonteret reward
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        # Normaliserer returns for at stabilisere træningen
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Konverter lister til tensorer
        states = torch.FloatTensor(np.array(self.states))
        tone_actions, rhythm_actions = zip(*self.actions)
        tone_actions = torch.LongTensor(tone_actions)
        rhythm_actions = torch.LongTensor(rhythm_actions)

        # Kør states gennem policy-netværket og beregn policy gradient
        probs = self.policy_net(states)
        tone_probs = probs[:, :len(FULL_RANGE)]
        rhythm_probs = probs[:, len(FULL_RANGE):]

        tone_dist = torch.distributions.Categorical(tone_probs)
        rhythm_dist = torch.distributions.Categorical(rhythm_probs)

        # Beregn log sandsynligheder
        tone_log_probs = tone_dist.log_prob(tone_actions)
        rhythm_log_probs = rhythm_dist.log_prob(rhythm_actions)

        # Loss-funktion for policy gradient
        loss = -(tone_log_probs + rhythm_log_probs) * returns
        loss = loss.mean()

        # Optimering af netværket
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Nulstil transitions
        self.states = []
        self.actions = []
        self.rewards = []



__all__ = ["PolicyAgent"]

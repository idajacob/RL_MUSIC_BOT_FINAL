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
    def __init__(self, state_size, action_size, lr=0.001):
        # Initialisering af agentens parametre og netværk
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = PolicyNetwork(state_size, action_size)
        
        # Optimizer til gradient descent
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Diskonteringsfaktor for fremtidige belønninger -- jeg har valgt en høj gamma hvor fremtidige belønninger tæller meget
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
    def get_action(self, state):
        # Konverterer til PyTorch tensor
        state = torch.FloatTensor(state).unsqueeze(0)

        # Kør forudsigelse gennem policy-netværket
        probs = self.policy_net(state)

        # === Debugging for NaN, Inf og ikke-finite værdier ===
        if torch.isnan(probs).any() or torch.isinf(probs).any() or not torch.isfinite(probs).all():
            print("[FEJL] probs havde ugyldige værdier. Fallback til ens sandsynligheder.")
            total_length = probs.size(1)
            
            if total_length == 0:
                total_length = 1
            
            probs = torch.ones((1, total_length)) / total_length

        # Normaliser værdierne, hvis de ikke summer til 1
        probs = probs / probs.sum(dim=1, keepdim=True)

        # Opdel sandsynlighederne
        tone_probs = probs[:, :len(FULL_RANGE)]
        rhythm_probs = probs[:, len(FULL_RANGE):]

        # === Valider dimensioner ===
        if tone_probs.size(1) != len(FULL_RANGE) or rhythm_probs.size(1) != len(RHYTHM_VALUES):
            print(f"[FEJL] Dimensionen af tone_probs eller rhythm_probs er forkert!")
            print(f"Størrelse af tone_probs: {tone_probs.size()} - Forventet: {len(FULL_RANGE)}")
            print(f"Størrelse af rhythm_probs: {rhythm_probs.size()} - Forventet: {len(RHYTHM_VALUES)}")

            # Fallback igen til ens fordeling
            tone_probs = torch.ones((1, len(FULL_RANGE))) / len(FULL_RANGE)
            rhythm_probs = torch.ones((1, len(RHYTHM_VALUES))) / len(RHYTHM_VALUES)

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

        # === Logging af handlinger ===
        # print(f"Valgt tone: {tone_action}, Valgt rytme: {rhythm_action} ({RHYTHM_VALUES[rhythm_action]})")

        return tone_action, rhythm_action


    # === Metode til at nulstille vægte, hvis der er NaN eller Inf ===
    def _weight_reset(self, layer):
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()

    # === Gem overgang (transition) ===
    def store_transition(self, state, action, reward):
        """Gemmer transitionen til opdatering af policy"""
        if state is not None and action is not None and reward is not None:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

    # === Opdater policy-netværket ===
    def update(self):
        R = 0
        returns = []

        # Beregn diskonteret reward
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        states = torch.FloatTensor(self.states)
        tone_actions, rhythm_actions = zip(*self.actions)
        tone_actions = torch.LongTensor(tone_actions)
        rhythm_actions = torch.LongTensor(rhythm_actions)

        # Kør states gennem policy-netværket
        probs = self.policy_net(states)
        tone_probs = probs[:, :len(FULL_RANGE)]
        rhythm_probs = probs[:, len(FULL_RANGE):]

        tone_dist = torch.distributions.Categorical(tone_probs)
        rhythm_dist = torch.distributions.Categorical(rhythm_probs)

        # Beregn log sandsynligheder
        tone_log_probs = tone_dist.log_prob(tone_actions)
        rhythm_log_probs = rhythm_dist.log_prob(rhythm_actions)

        # Policy gradient loss
        loss = -(tone_log_probs + rhythm_log_probs) * returns
        loss = loss.mean()

        # Optimering
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Nulstil transitions
        self.states = []
        self.actions = []
        self.rewards = []



__all__ = ["PolicyAgent"]

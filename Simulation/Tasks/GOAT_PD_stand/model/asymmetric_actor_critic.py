import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

class Asymmetric_Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, cfg):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, cfg["clip_actions"], cfg["clip_log_std"], cfg["min_log_std"], cfg["max_log_std"], cfg["reduction"])

        self.net = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU())
        
        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def act(self, inputs, role):
        actions, log_prob, outputs = GaussianMixin.act(self, inputs, role)
        
        return actions, log_prob, outputs
    
    def compute(self, inputs, role):
        x = self.net(inputs["observations"])
        return self.mean_layer(x), self.log_std_parameter, {}

class Asymmetric_Critic(DeterministicMixin, Model):
    def __init__(self, state_space, action_space, device, cfg):
        Model.__init__(self, observation_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, cfg["clip_actions"])
        
        self.num_states = self.num_observations
        
        self.net = nn.Sequential(nn.Linear(self.num_states, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 1))

    def act(self, inputs, role):
        return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        x = self.net(inputs["states"])
        return x, {}
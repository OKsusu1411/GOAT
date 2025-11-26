import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

class Asymmetric_Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU())
        
        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def act(self, inputs, role):
        return GaussianMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        x = self.net(inputs["observations"])
        return self.mean_layer(x), self.log_std_parameter, {}

class Asymmetric_Critic(DeterministicMixin, Model):
    def __init__(self, state_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions)
        
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
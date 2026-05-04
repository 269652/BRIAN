"""Multi-Agent / Modular Reasoning for NeuroSLM

Implements a set of interacting agent modules with learned communication protocols.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentModule(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.core = nn.Linear(d_model, d_model)
        self.comm = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, comm_in: torch.Tensor) -> torch.Tensor:
        # Process own state and received communication
        return self.core(x) + self.comm(comm_in)

class MultiAgentReasoning(nn.Module):
    def __init__(self, d_model: int, n_agents: int = 4):
        super().__init__()
        self.n_agents = n_agents
        self.agents = nn.ModuleList([AgentModule(d_model) for _ in range(n_agents)])
        self.comm_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        # Each agent gets the same input, communicates, and updates
        comm = torch.zeros_like(x)
        for _ in range(2):  # two rounds of communication
            outs = [agent(x, comm) for agent in self.agents]
            comm = self.comm_proj(sum(outs) / self.n_agents)
        # Aggregate agent outputs
        out = sum(outs) / self.n_agents
        return out

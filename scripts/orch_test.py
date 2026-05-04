import sys
sys.path.insert(0, '.')
from neuroslm.intelligence.orchestrator import HomeostaticGate, NeuralOrchestrator
import torch

hg = HomeostaticGate(128)
print('HomeostaticGate OK, dtype:', hg.running_mean.dtype)

orch = NeuralOrchestrator(128, ['pfc','hippocampus'])
print('NeuralOrchestrator OK')

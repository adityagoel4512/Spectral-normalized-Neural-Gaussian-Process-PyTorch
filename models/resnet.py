import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetBackbone(nn.Module):
  def __init__(self, 
               input_features: int = 2, 
               num_hidden_layers: int = 5, 
               num_hidden: int = 128, 
               dropout_rate: float = 0.1,
               norm_multiplier: float = 0.9):
    super().__init__()
    self.num_hidden = num_hidden
    self.dropout_rate = dropout_rate
    self.input_layer = nn.Linear(in_features=input_features, out_features=num_hidden)
    self.hidden_layers = nn.Sequential(*[nn.Linear(in_features=num_hidden, out_features=num_hidden) for _ in range(num_hidden_layers)])
    
    if norm_multiplier is not None:
      self.norm_multiplier = norm_multiplier
      self.input_layer.register_full_backward_hook(self.spectral_norm_hook)
      for hidden_layer in self.hidden_layers:
        hidden_layer.register_full_backward_hook(self.spectral_norm_hook)

  def forward(self, input):
    input = self.input_layer(input)
    for hidden_layer in self.hidden_layers:
      residual = input
      input = F.dropout(F.relu(hidden_layer(input)), p=self.dropout_rate, training=self.training)
      input += residual
    return input
  
  def spectral_norm_hook(self, module, grad_input, grad_output):
    # applied to linear layer weights after gradient descent updates
    with torch.no_grad():
      norm = torch.linalg.matrix_norm(module.weight, 2)
      if self.norm_multiplier < norm:
        module.weight = nn.Parameter(self.norm_multiplier * module.weight / norm)
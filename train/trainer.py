import matplotlib.pyplot as plt
from models.gaussian_process_layer import RandomFeatureGaussianProcess
import torch
from torch.utils.data import DataLoader

class Trainer:
  def __init__(self, 
               model_config, 
               task_type='classification',
               model=RandomFeatureGaussianProcess):
    self.model = model(**model_config)
    print(self.model)
    criterions = {
        'classification': torch.nn.CrossEntropyLoss(reduction='mean'),
        'regression': torch.nn.MSELoss(reduction='mean')
        }
    self.criterion = criterions[task_type]
    self.update_precision_incrementally = model_config['momentum'] > 0

  def train(self, training_data, data_loader_config, epochs=100, lr=1e-3):
    training_loader = DataLoader(training_data, **data_loader_config)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    mse = torch.nn.MSELoss(reduction='mean')
    l2 = lambda betas: mse(betas, betas.new_zeros(betas.shape))
    
    def compute_loss(batch):
      X, y = batch
      logits, variance = self.model(X, with_variance=False, update_precision=self.update_precision_incrementally)

      neg_log_likelihood = self.criterion(logits, y)

      # RFF approximation reduced the GP problem to a Bayesian linear model
      # Since beta has gaussian prior, the objective can be written as a standard
      # MAP estimation as a regularised MLE objective
      
      betas = self.model.beta.weight
      l2_loss = 0.5 * l2(betas)

      # − log p(β|D) = − log p(D|β) + 0.5*||β||2
      neg_log_posterior = neg_log_likelihood + l2_loss
      return neg_log_posterior

    self.model.train(True)
    self.epoch_losses = []
    for epoch in range(epochs):
      running_loss = 0.
      for i, batch in enumerate(training_loader):
        loss = compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      running_loss /= i
      self.epoch_losses.append(running_loss)
      if epoch % 10 == 0:
        print(f'Avg Loss Epoch {epoch}: {running_loss}')
    self.model.train(False)
    if not self.update_precision_incrementally:
        self.model.update_precision(next(iter(DataLoader(training_data, **{**data_loader_config, **dict(batch_size=len(training_data))})))[0])
    return self.model 

  def plot_loss(self):
    if not hasattr(self, 'epoch_losses'):
        raise ValueError('plot_loss invoked without first training model')
    plt.plot(list(range(1, len(self.epoch_losses)+1)), self.epoch_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

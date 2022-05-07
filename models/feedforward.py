import torch
from torch.utils.data import DataLoader

from models.sngp.models.resnet import ResNetBackbone
import torch.nn.functional as F
import matplotlib.pyplot as plt

class FeedForwardResNet(torch.nn.Module):
    def __init__(self, input_features, num_hidden_layers, num_hidden, dropout_rate, output_features):
        super(FeedForwardResNet, self).__init__()
        self.backbone = ResNetBackbone(
            input_features=input_features,
            num_hidden_layers=num_hidden_layers,
            num_hidden=num_hidden,
            dropout_rate=dropout_rate,
            norm_multiplier=None
        )
        self.output_layer = torch.nn.Linear(in_features=num_hidden, out_features=output_features)
        self.dropout_rate = dropout_rate

    def forward(self, input):
        return self.output_layer(F.dropout(self.backbone(input), self.dropout_rate))


class FeedForwardNet(torch.nn.Module):
    def __init__(self, input_features, num_hidden_layers, num_hidden, output_features):
        super(FeedForwardNet, self).__init__()
        self.input_layer = torch.nn.Linear(in_features=input_features, out_features=num_hidden)
        self.hidden_layers = torch.nn.Sequential(
            *[torch.nn.Linear(in_features=num_hidden, out_features=num_hidden) for _ in range(num_hidden_layers)])
        self.output_layer = torch.nn.Linear(in_features=num_hidden, out_features=output_features)

    def forward(self, input):
        output = F.relu(self.input_layer(input))
        for layer in self.hidden_layers:
            output = F.relu(layer(output))
        output = self.output_layer(output)
        return output


class FeedForwardTrainer:
    def __init__(self,
                 model_config,
                 task_type='classification',
                 model=FeedForwardNet):
        self.model = model(**model_config)
        print(self.model)
        criterions = {
            'classification': torch.nn.CrossEntropyLoss(reduction='mean'),
            'regression': torch.nn.MSELoss(reduction='mean')
        }
        self.criterion = criterions[task_type]

    def train(self, training_data, data_loader_config, epochs=100, lr=1e-3):
        training_loader = DataLoader(training_data, **data_loader_config)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mse = torch.nn.MSELoss(reduction='mean')

        def compute_loss(batch):
            X, y = batch
            predictions = self.model(X)

            neg_log_likelihood = self.criterion(predictions, y)
            return neg_log_likelihood

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
        return self.model

    def plot_loss(self, title):
        if not hasattr(self, 'epoch_losses'):
            raise ValueError('plot_loss invoked without first training model')
        plt.plot(list(range(1, len(self.epoch_losses) + 1)), self.epoch_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Loss [{title}]')
        plt.show()

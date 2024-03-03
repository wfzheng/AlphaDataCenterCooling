import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self,input_size,input_normalization_minimum,input_normalization_maximum,output_low_limit=None,output_high_limit=None):
        super(MLP, self).__init__()

        self.input_normalization_minimum = torch.tensor(input_normalization_minimum, dtype=torch.float32)
        self.input_normalization_maximum = torch.tensor(input_normalization_maximum, dtype=torch.float32)
        if output_low_limit is not None:
            self.output_high_limit=torch.tensor(output_high_limit, dtype=torch.float32)
            self.output_low_limit=torch.tensor(output_low_limit, dtype=torch.float32)

        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def normalize_input(self, data):
        # 进行最小-最大归一化
        normalized_data = (data - self.input_normalization_minimum) / \
                          (self.input_normalization_maximum - self.input_normalization_minimum)

        return normalized_data
    def normalize_output(self, output):
        # 进行最小-最大归一化
        normalize_output = (output - self.output_low_limit) / \
                          (self.output_high_limit - self.output_low_limit)

        return normalize_output
    def inverse_normalize_data(self, normalized_data_tensor):
        original_data = normalized_data_tensor * (self.output_high_limit - self.output_low_limit) + self.output_low_limit
        return original_data

    def forward(self, x):
        x_normalized = self.normalize_input(x)
        return self.layers(x_normalized)
    def step(self, x):
        x_normalized = self.normalize_input(x)
        return self.inverse_normalize_data(self.layers(x_normalized))
import torch.nn as nn

class MLP(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()

		self.model = nn.Sequential(

			nn.Linear(input_dim, input_dim),
			nn.ReLU(),
			nn.Dropout(0.1),


			nn.Linear(input_dim, input_dim),
			nn.ReLU(),
			nn.Dropout(0.1),

			nn.Linear(input_dim, output_dim),
			#nn.Sigmoid()

		)

	def forward(self, x):
		output = self.model(x)
		return output
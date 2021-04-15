import torch
from torch import nn
import numpy as np


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(DQN,self).__init__()

		# input_shape should be image channels x width x height

		# Note, formula for calculating matrix size after conv:
		# Width2 = (Width - Kernel_Size + 2*Padding)/Stride + 1
		# Height2 = (Height - Kernel_Size + 2*Padding)/Stride + 1

		# using architecture from Chuchro & Gupta, 2017
		# http://cs231n.stanford.edu/reports/2017/pdfs/616.pdf
		# inputs are 4 stacked frames of 80x80x1 gray scale images
		self.conv_layers = nn.Sequential(
			nn.Conv2d(in_channels=input_shape[0],
					  out_channels=32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(in_channels=32, out_channels=64,
					  kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=64, out_channels=64,
					  kernel_size=3),
			nn.ReLU(),
			nn.Flatten()
		)

		cnn_out_size = self._get_conv_out(input_shape)

		self.fc_layers = nn.Sequential(
			nn.Linear(in_features=cnn_out_size, out_features=512),
			nn.ReLU(),
			nn.Linear(in_features=512, out_features=n_actions)  # set to env.action_space.n
		)

	def _get_conv_out(self,shape):
		o = self.conv_layers(torch.zeros(1,*shape))
		return int(np.prod(o.size()))

	def forward(self, x):

		conv_out = self.conv_layers(x)

		return self.fc_layers(conv_out)


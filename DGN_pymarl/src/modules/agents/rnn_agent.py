import torch.nn as nn
import torch.nn.functional as F
import torch

class AttModel(nn.Module):
	def __init__(self, din, hidden_dim):
		super(AttModel, self).__init__()
		self.fcv = nn.Linear(din, hidden_dim)
		self.fck = nn.Linear(din, hidden_dim)
		self.fcq = nn.Linear(din, hidden_dim)

	def forward(self, x, mask):
		v = F.relu(self.fcv(x))
		q = F.relu(self.fcq(x))
		k = F.relu(self.fck(x)).permute(0,2,1)
		att = F.softmax(torch.mul(torch.bmm(q,k), mask) - 9e15*(1 - mask),dim=2)

		out = torch.bmm(att,v)
		#out = torch.bmm(mask,v) #commnet
		#out = torch.add(out,v)
		return out

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.att1 = AttModel(args.rnn_hidden_dim,args.rnn_hidden_dim)
        self.att2 = AttModel(args.rnn_hidden_dim,args.rnn_hidden_dim)
       
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, masks):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
       
        h = h.reshape(-1, self.args.n_agents,self.args.rnn_hidden_dim)
        h = self.att1(h, masks)
        h = self.att2(h, masks)
        h = h.reshape(-1, self.args.rnn_hidden_dim)

        q = self.fc2(h)
        return q, h

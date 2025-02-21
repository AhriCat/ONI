
class HomeostaticController(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HomeostaticController, self).__init__()
        self.kan = KanSpikingNeuron(input_dim)
        self.dynamic_layer = DynamicSynapse(input_dim, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim)
        self.history = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, system_state):
        feedback_signal = self.calculate_feedback(system_state + self.history)
        x = self.kan(x)
        x = self.rnn(x)
        self.history = system_state + self.history
        fused_output = x + system_state
        self.history =torch.tanh(torch.matmul(fused_output, self.fc.weight.detach()))
        adjusted_output = self.dynamic_layer(x + feedback_signal)

        return adjusted_output, feedback_signal

    def calculate_feedback(self, system_state):
        return torch.randn_like(system_state)

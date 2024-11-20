import numpy as np

class CTRNN:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.states = np.random.uniform(-0.1, 0.1, self.num_neurons)

        self.weights = np.random.uniform(-1, 1, (self.num_neurons, self.num_neurons))
        self.biases = np.random.uniform(-1, 1, self.num_neurons)
        self.taus = np.ones(self.num_neurons) * 0.1  # Time constant for each neuron
        self.prev_output = np.zeros(self.num_neurons)
        # print("states")
        # print(self.states)
        # print("weights")
        # print(self.weights)
        # print("biases")
        # print(self.biases)

    def step(self, inputs, fitness):
        #Neuron Input
        total_input = np.dot(self.weights, self.states) + self.biases + inputs
        total_input *= (fitness + 1)

        # Update neuron states
        self.states += (1 / self.taus) * (-self.states + total_input)
        self.states = np.clip(self.states, -1e2, 1e2)  #Clip values

        #limit output between -1 and 1
        output = np.tanh(self.states)

        #output = 1 * self.prev_output + 1 * output
        #self.prev_output = output

        return output

    def clone(self):
        cloned_nn = CTRNN(self.num_neurons)
        cloned_nn.weights = np.copy(self.weights)
        cloned_nn.biases = np.copy(self.biases)
        cloned_nn.states = np.copy(self.states)
        cloned_nn.taus = np.copy(self.taus)
        return cloned_nn




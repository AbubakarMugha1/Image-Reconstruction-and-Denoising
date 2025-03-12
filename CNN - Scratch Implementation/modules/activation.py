import numpy as np


class ReLU:
    def __init__(self):
        self.inputs = None
        self.has_weights = False

    def forward(self, inputs):
        """Forward pass for the ReLU activation function

        Args:
            inputs (np.ndarray):
                input array, can have any shape

        Returns: (np.ndarray):
            array of the same shape as the input
        """

        self.inputs = inputs

        # ================ Insert Code Here ================
        return np.maximum(0, inputs)
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass for the ReLU activation function

        Args:
            d_outputs (np.ndarray): array of any shape

        Returns (dict):
            Dictionary containing the derivative of the loss with
            respect to the output of the layer. The key of the dictionary
            should be "d_out"
        """
        # ================ Insert Code Here ================
        d_in = d_outputs.copy()
        d_in[self.inputs <= 0] = 0
        return {"d_out": d_in}
        # ==================================================


class Sigmoid:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.has_weights = False

    def forward(self, inputs):
        """Forward pass for the Sigmoid activation function

        Args:
            inputs (np.ndarray):
                input array, can have any shape

        Returns: (np.ndarray):
            array of the same shape as the input
        """

        self.inputs = inputs

        # ================ Insert Code Here ================
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass for the Sigmoid activation function

        Args:
            d_outputs (np.ndarray): array of any shape

        Returns (dict):
            Dictionary containing the derivative of the loss with
            respect to the output of the layer. The key of the dictionary
            should be "d_out"
        """

        # ================ Insert Code Here ================
        d_inputs = d_outputs * self.outputs * (1 - self.outputs)
        return {"d_out": d_inputs}
        # ==================================================


class Softmax:
    def __init__(self):
        self.inputs = None
        self.has_weights = False
        self.outputs = None

    def forward(self, inputs):
        """Forward pass for the ReLU activation function

        Args:
            inputs (np.ndarray):
                input array, can have any shape

        Returns: (np.ndarray):
            array of the same shape as the input
        """
        self.inputs = inputs

        # ================ Insert Code Here ================
        shift_inputs = inputs - np.max(inputs, axis = 1, keepdims = True)
        exp = np.exp(shift_inputs)
        self.outputs = exp / np.sum(exp, axis = 1, keepdims = True)
        return self.outputs
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass for the Softmax activation function

        Args:
            d_outputs (np.ndarray): array of any shape

        Returns (dict):
            Dictionary containing the derivative of the loss with
            respect to the output of the layer. The key of the dictionary
            should be "d_out"
        """

        # ================ Insert Code Here ================
        d_inputs = np.zeros_like(d_outputs)

        for i, (softmax_out, d_out) in enumerate(zip(self.outputs, d_outputs)):
            s = softmax_out.reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            d_inputs[i] = np.dot(jacobian, d_out)
        
        return {"d_out": d_inputs}
        # ==================================================

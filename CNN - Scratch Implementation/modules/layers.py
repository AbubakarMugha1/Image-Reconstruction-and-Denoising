import numpy as np


class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weights = np.random.rand(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32)
        self.bias = np.random.rand(out_channels).astype(np.float32)

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):
        """Forward pass for a convolution layer

        Args:
            inputs (np.ndarray):
                array of shape
                (batch_size, in_channels, height, width)

        Returns: (np.ndarray):
            array of shape
            (batch_size, out_channels, new_height, new_width)
        """

        self.inputs = inputs

        # ================ Insert Code Here ================
        batch_size, _, in_height, in_width = inputs.shape
        kernel = self.kernel_size
        stride = self.stride

        out_height = (in_height - kernel) // stride + 1
        out_width = (in_width - kernel) // stride + 1

        output = np.zeros((batch_size, self.out_channels, out_height, out_width), dtype = np.float32)

        for i in range(batch_size):
            for oc in range(self.out_channels):
                for y in range(out_height):
                    for x in range(out_width):
                        y_start = y * stride
                        y_end = y_start + kernel
                        x_start = x * stride
                        x_end = x_start + kernel
                        region = inputs[i, :, y_start:y_end, x_start:x_end]
                        output[i, oc, y, x] = np.sum(region * self.weights[oc]) + self.bias[oc]
        
        return output
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass of convolution layer

        Args:
            d_outputs (np.ndarray):
                derivative of loss with respect to the output
                of the layer. Will have shape
                (batch_size, out_channels, new_height, new_width)

        Returns: (dict):
            Dictionary containing the derivatives of loss with respect to
            the weights and bias and input of the layer. The keys of
            the dictionary should be "d_weights", "d_bias", and "d_out"

        """

        # ================ Insert Code Here ================
        inputs = self.inputs
        batch_size, _, _, _ = inputs.shape
        kernel = self.kernel_size
        stride = self.stride
        _, _, out_height, out_width = d_outputs.shape

        d_weights = np.zeros_like(self.weights)
        d_bias = np.zeros_like(self.bias)
        d_inputs = np.zeros_like(inputs)

        for i in range(batch_size):
            for oc in range(self.out_channels):
                for y in range(out_height):
                    for x in range(out_width):
                        y_start = y * stride
                        y_end = y_start + kernel
                        x_start = x * stride
                        x_end = x_start + kernel

                        region = inputs[i, :, y_start:y_end, x_start:x_end]
                        d_bias[oc] += d_outputs[i , oc, y, x]
                        d_weights[oc] += d_outputs[i , oc, y, x] * region
                        d_inputs[i, :, y_start:y_end, x_start:x_end] += d_outputs[i , oc, y, x] * self.weights[oc]
        
        return {"d_weights": d_weights, "d_bias": d_bias, "d_out": d_inputs}

        # ==================================================

    def update(self, d_weights, d_bias, learning_rate):

        # ================ Insert Code Here ================
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        # ==================================================


class Flatten:
    def __init__(self):
        self.inputs_shape = None
        self.has_weights = False

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, d_outputs):
        return {"d_out": d_outputs.reshape(self.inputs_shape)}


class LinearLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = np.random.rand(out_features, in_features).astype(np.float32)
        self.bias = np.random.rand(out_features).astype(np.float32)

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):
        """Forward pass for a linear layer

        Args:
            inputs (np.ndarray):
                array of shape (batch_size, in_features)

        Returns: (np.ndarray):
            array of shape (batch_size, out_features)
        """

        # ================ Insert Code Here ================
        self.inputs = inputs
        return np.dot(inputs, self.weights.T) + self.bias
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass of Linear layer

        Args:
            d_outputs (np.ndarray):
                derivative of loss with respect to the output
                of the layer. Will have shape
                (batch_size, out_features)

        Returns: (dict):
            Dictionary containing the derivatives of loss with respect to
            the weights and bias and input of the layer. The keys of
            the dictionary should be "d_weights", "d_bias", and "d_out"
        """        
        # ================ Insert Code Here ================
        d_weights = np.dot(d_outputs.T, self.inputs)
        d_bias = np.sum(d_outputs, axis = 0)
        d_inputs = np.dot(d_outputs, self.weights)

        return {"d_weights": d_weights, "d_bias": d_bias, "d_out": d_inputs}
        # ==================================================

    def update(self, d_weights, d_bias, learning_rate):

        # ================ Insert Code Here ================
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        # ==================================================

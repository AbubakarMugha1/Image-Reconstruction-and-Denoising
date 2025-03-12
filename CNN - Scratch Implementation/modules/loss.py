import numpy as np


class CrossEntropy:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.inputs = None
        self.targets = None

    def forward(self, inputs, targets):
        """Forward pass for the cross entropy loss function

        Args:
            inputs (np.ndarray): predictions from the model
            targets (_type_): ground truth labels (one-hot encoded)

        Returns: (float):
            loss value
        """

        self.inputs = inputs
        self.targets = targets

        # ================ Insert Code Here ================
        batch_size = inputs.shape[0]
        loss = -np.sum(targets * np.log(inputs + self.eps)) / batch_size
        return loss
        # ==================================================

    def backward(self):
        """Backward pass for the cross entropy loss function

        Args:
            None

        Returns: (dict):
            Dictionary containing the derivative of the loss
            with respect to the inputs to the loss function.
            The key of the dictionary should be "d_out"
        """
        
        # ================ Insert Code Here ================
        d_inputs = -(self.targets / (self.inputs + self.eps))
        return {"d_out": d_inputs}
        # ==================================================

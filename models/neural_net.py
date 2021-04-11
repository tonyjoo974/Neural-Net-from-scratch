"""Neural network model."""

from typing import Sequence

import numpy as np

#TODO: ask in OH - backprop updating weights, finding gradient of bias, regularization

class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        return np.dot(x, W) + b
        
    def linear_grad(self, W: np.ndarray, dz: np.ndarray, x: np.ndarray):
        dx = dz @ W.T
        dW = np.reshape(x,(1,-1)).T @ np.reshape(dz,(1,-1))
        return dx, dW
        
    def relu(self, x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        return np.maximum(x, 0)

    def relu_grad(self, dz: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).

        Parameters:
            dz: the input data

        Returns:
            the output data
        """
        jacobian = np.diag((x > 0) * 1.0)
        return dz @ jacobian

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def softmax_grad(self, scores: np.ndarray, y: int, batch_size: int) -> np.ndarray:
        """Gradient of Softmax (and cross entropy loss)s.

        Parameters:
            X: the input data
            y: label data
            
        Returns:
            the output data
        """
        grad = np.copy(scores)
        grad[y] -= 1.0
        # grad /= batch_size
        return grad

    def two_layer_forward(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((X.shape[0], self.output_size))
        for i in range(X.shape[0]):
            self.outputs["data_"+str(i)] = X[i]
            # input layer
            W1 = self.params["W1"]
            b1 = self.params["b1"]
            train_data = self.outputs["data_"+str(i)]
            self.outputs["linear1_"+str(i)] = self.linear(W1, train_data, b1)
            linear1_out = self.outputs["linear1_"+str(i)]
            # hidden layer 1 (relu)
            self.outputs["relu_"+str(i)] = self.relu(linear1_out)
            relu_out = self.outputs["relu_"+str(i)]
            W2 = self.params["W2"]
            b2 = self.params["b2"]
            self.outputs["linear2_"+str(i)] = self.linear(W2, relu_out, b2)
            linear2_out = self.outputs["linear2_"+str(i)]
            # output layer (softmax)
            self.outputs["softmax_"+str(i)] = self.softmax(linear2_out)
            scores[i] = self.outputs["softmax_"+str(i)]
        return scores
        
    def three_layer_forward(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((X.shape[0], self.output_size))
        for i in range(X.shape[0]):
            self.outputs["data_"+str(i)] = X[i]
            # input layer
            W1 = self.params["W1"]
            b1 = self.params["b1"]
            train_data = self.outputs["data_"+str(i)]
            self.outputs["linear1_"+str(i)] = self.linear(W1, train_data, b1)
            linear1_out = self.outputs["linear1_"+str(i)]
            # hidden layer 1 (linear)
            W2 = self.params["W2"]
            b2 = self.params["b2"]
            self.outputs["linear2_"+str(i)] = self.linear(W2, linear1_out, b2)
            linear2_out = self.outputs["linear2_"+str(i)]
            # hidden layer 2 (relu)
            self.outputs["relu_"+str(i)] = self.relu(linear2_out)
            relu_out = self.outputs["relu_"+str(i)]
            W3 = self.params["W3"]
            b3 = self.params["b3"]
            self.outputs["linear3_"+str(i)] = self.linear(W3, relu_out, b3)
            linear3_out = self.outputs["linear3_"+str(i)]
            # output layer (softmax)
            self.outputs["softmax_"+str(i)] = self.softmax(linear3_out)
            scores[i] = self.outputs["softmax_"+str(i)]
        return scores

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        scores = np.zeros((X.shape[0], self.output_size))
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.
        if self.num_layers == 2:
            return self.two_layer_forward(X)
        elif self.num_layers == 3:
            return self.three_layer_forward(X)
        return 0.0
        
        
    def two_layer_backward(self, y: np.ndarray, reg: float) ->float:
        loss = 0.0
        # do backprop
        batch_size = y.shape[0]
        for i in range(batch_size):
            # add loss from training data
            scores = self.outputs["softmax_"+str(i)]
            log_likelihood = -np.log(scores[int(y[i])])
            loss += log_likelihood
            # take gradient of softmax error
            softmax_grad = self.softmax_grad(scores, int(y[i]), batch_size) 
            # take gradient of W2 error and update (linear)
            W2 = self.params["W2"]
            loss += ((reg/2)*(np.linalg.norm(self.params["W2"], 'fro')**2))
            dx2, dW2 = self.linear_grad(W2, softmax_grad, self.outputs["relu_"+str(i)])
            self.gradients["W2"] += (dW2 + reg*self.params["W2"]) 
            self.gradients["b2"] += softmax_grad 
            # take gradient of relu error
            relu_grad = self.relu_grad(dx2, self.outputs["linear1_"+str(i)])
            # take gradient of W1 error and update (linear)
            W1 = self.params["W1"]
            loss += ((reg/2)*(np.linalg.norm(self.params["W1"], 'fro')**2))
            dx1, dW1 = self.linear_grad(W1, relu_grad, self.outputs["data_"+str(i)])
            self.gradients["W1"] += (dW1 + reg*self.params["W1"])
            self.gradients["b1"] += relu_grad
        # update parameter grads
        for i in range(1, self.num_layers+1):
            self.gradients["W"+str(i)] /= batch_size
            self.gradients["b"+str(i)] /= batch_size
        # add regularization
        # for i in range(1, self.num_layers + 1):
        #     loss += ((reg/2)*np.linalg.norm(self.params["W"+str(i)], 2)/batch_size)

        return loss / batch_size
        
    def three_layer_backward(self, y: np.ndarray, reg: float) -> float:
        loss = 0
        batch_size = y.shape[0]
        for i in range(batch_size):
            # add loss from training data
            scores = self.outputs["softmax_"+str(i)]
            log_likelihood = -np.log(scores[int(y[i])])
            loss += log_likelihood
            # take gradient of softmax error
            softmax_grad = self.softmax_grad(scores, int(y[i]), batch_size) 
            # take gradient of W3 error and update (linear)
            W3 = self.params["W3"]
            loss += ((reg/2)*(np.linalg.norm(self.params["W3"], 'fro')**2))
            dx3, dW3 = self.linear_grad(W3, softmax_grad, self.outputs["relu_"+str(i)])
            self.gradients["W3"] += (dW3 + reg*self.params["W3"])
            self.gradients["b3"] += softmax_grad
            # take gradient of relu error
            relu_grad = self.relu_grad(dx3, self.outputs["linear2_"+str(i)])
            # take gradient of W2 error and update (linear)
            W2 = self.params["W2"]
            loss += ((reg/2)*(np.linalg.norm(self.params["W2"], 'fro')**2))
            dx2, dW2 = self.linear_grad(W2, relu_grad, self.outputs["linear1_"+str(i)])
            self.gradients["W2"] += (dW2 + reg*self.params["W2"])
            self.gradients["b2"] += relu_grad
            # take gradient of W1 error and update (linear)
            W1 = self.params["W1"]
            loss += ((reg/2)*(np.linalg.norm(self.params["W1"], 'fro')**2))
            dx1, dW1 = self.linear_grad(W1, dx2, self.outputs["data_"+str(i)])
            self.gradients["W1"] += (dW1 + reg*self.params["W1"])
            self.gradients["b1"] += dx2
        # update parameter grads
        for i in range(1, self.num_layers+1):
            self.gradients["W"+str(i)] /= batch_size
            self.gradients["b"+str(i)] /= batch_size
        # add regularization
        # for i in range(1, self.num_layers + 1):
        #     loss += ((reg/2)*np.linalg.norm(self.params["W"+str(i)], 2)/batch_size)

        return loss / batch_size
        
    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """Perform back-propagation and compute the gradients and losses.

        Note: both gradients and loss should include regularization.

        Parameters:
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        for n in range(1, self.num_layers+1):
            self.gradients["W"+str(n)] = np.zeros(self.params["W"+str(n)].shape)
            self.gradients["b"+str(n)] = np.zeros(self.params["b"+str(n)].shape)
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        if self.num_layers == 2:
            return self.two_layer_backward(y, reg)
        elif self.num_layers == 3:
            return self.three_layer_backward(y, reg)
        return

    def update(
        self,
        reg: float,
        opt: str,
        lr: float,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.

        Parameters:
            reg: Regularization constant
            t: t-variable for Adam
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
            for i in range(1, self.num_layers+1):
                # print(self.gradients["b" + str(i)])
                self.params["W" + str(i)] -= lr*(self.gradients["W"+str(i)]) #+ reg*self.params["W"+str(i)])
                self.params["b" + str(i)] -= lr*self.gradients["b" + str(i)]
        elif opt == "Adam":
            for i in range(1, self.num_layers + 1):
                optim = Adam(lr, b1, b2, eps)
                W = self.params["W" + str(i)]
                b = self.params["b" + str(i)]
                dW = self.gradients["W" + str(i)]
                db = self.gradients["b" + str(i)]
                self.params["W" + str(i)], self.params["b" + str(i)] = optim.step(W, b, dW, db)


class Adam:
    def __init__(self, lr, b1, b2, eps):
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0

    def step(self, W, b, dW, db):
        self.t += 1
        # update momentum
        self.m_w = self.b1 * self.m_w + (1-self.b1) * dW
        self.m_b = self.b1 * self.m_b + (1-self.b1) * db
        self.v_w = self.b2 * self.v_w + (1-self.b2) * (dW**2)
        self.v_b = self.b2 * self.v_b + (1-self.b2) * (db**2)
        # bias correction
        m_w_hat = self.m_w/(1-self.b1**self.t)
        m_b_hat = self.m_b/(1-self.b1**self.t)
        v_w_hat = self.v_w/(1-self.b2**self.t)
        v_b_hat = self.v_b/(1-self.b2**self.t)

        # update weight and biases
        W -= (self.lr * m_w_hat)/(np.sqrt(v_w_hat)+self.eps)
        b -= (self.lr * m_b_hat)/(np.sqrt(v_b_hat)+self.eps)
        return W, b


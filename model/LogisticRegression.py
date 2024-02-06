import numpy as np
from utils.lr_utils import sigmoid


class LogisticRegression:
    """
    Train, test and predict using logistic regression model
    """

    def __init__(self, train_dataset_x: np.ndarray, train_dataset_y: np.ndarray,
                 test_dataset_x: np.ndarray, test_dataset_y: np.ndarray, classes: str,
                 num_iterations: int, learning_rate: float, print_cost: bool):
        self.train_set_x = train_dataset_x
        self.train_set_y = train_dataset_y
        self.test_set_x = test_dataset_x
        self.test_set_y = test_dataset_y

        self.classes = classes
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost

        self.n_x = self.train_set_x.shape[0]  # no of input features
        self.m = train_dataset_x.shape[1]  # no of training data

        self.w = np.zeros([self.n_x, 1])
        self.b = 0.0

    def propagate(self):
        """Run forward and backward propagation once

        Returns:
            `({"dw": dw, "db" db}, cost)`: tuple containing a dict of gradients `dw` & `db` and cost
        """
        z = np.dot(self.w.T, self.train_set_x) + self.b
        a = sigmoid(z)

        # Introduced so that we don't run into log(0)
        epsilon = 1e-5

        loss = -self.train_set_y * \
            np.log(a + epsilon) - (1 - self.train_set_y) * \
            np.log(1 - a + epsilon)
        cost = np.sum(loss) / self.m

        dz = a - self.train_set_y
        dw = np.dot(self.train_set_x, dz.T) / self.m
        db = np.sum(dz) / self.m

        assert (dw.shape == self.w.shape)
        assert (db.dtype == float)
        assert (cost.dtype == float)
        
        grads = {"dw": dw, "db": db}

        return grads, cost

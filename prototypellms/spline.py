
# Define a simple B-Spline function
def b_spline(x, knots, coeffs, degree=3):
    k = degree
    t = np.concatenate(([knots[0]] * k, knots, [knots[-1]] * k))
    c = coeffs
    spline = BSpline(t, c, k)
    return spline(x)

# KAN Layer
class KANLayer:
    def __init__(self, input_dim, output_dim, spline_degree=3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spline_degree = spline_degree
        
        # Initialize B-spline parameters
        self.knots = [np.linspace(0, 1, 10) for _ in range(output_dim)]
        self.coeffs = [np.random.randn(10 + spline_degree - 1) for _ in range(output_dim)]

    def forward(self, x):
        out = np.zeros((x.shape[0], self.output_dim))
        for j in range(self.output_dim):
            sum_b_spline = np.sum([b_spline(x[:, i], self.knots[j], self.coeffs[j], self.spline_degree) for i in range(self.input_dim)], axis=0)
            out[:, j] = sum_b_spline
        return out

# KAN Model
class KANModel:
    def __init__(self, input_dim, hidden_dim, output_dim, spline_degree=3):
        self.layer1 = KANLayer(input_dim, hidden_dim, spline_degree)
        self.layer2 = KANLayer(hidden_dim, output_dim, spline_degree)

    def forward(self, x):
        out = self.layer1.forward(x)
        out = self.layer2.forward(out)
        return out

    # Loss function: Mean Squared Error (MSE)
    def mse_loss(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    # Training function
    def train_kan(model, x_train, y_train, epochs=1000, lr=0.001):
        for epoch in range(epochs):
            # Forward pass
            y_pred = model.forward(x_train)
            
            # Compute loss
            loss = mse_loss(y_pred, y_train)
            
            # Compute gradients (this is a placeholder; normally you would use backpropagation)
            grad = 2 * (y_pred - y_train) / y_train.size
            
            # Update B-spline coefficients
            for layer in [model.layer1, model.layer2]:
                for j in range(layer.output_dim):
                    for i in range(layer.input_dim):
                        # Update each coefficient
                        layer.coeffs[j] -= lr * grad[:, j].dot(b_spline(x_train[:, i], layer.knots[j], layer.coeffs[j], layer.spline_degree))
    
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

kan = KANModel

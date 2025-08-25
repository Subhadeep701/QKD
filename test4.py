import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Your system model:
# θ_{k+1} = θ_k + q_k  (state equation)
# y_k = x_k * θ_k + w_k  (measurement equation)

class KalmanNetTSP(nn.Module):
    def __init__(self, state_dim=1, measurement_dim=1):
        super(KalmanNetTSP, self).__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # Network layers (architecture based on KalmanNet_TSP)
        self.fc1 = nn.Linear(measurement_dim + state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, state_dim)

        self.activation = nn.ReLU()

    def forward(self, measurement, previous_state):
        # Concatenate measurement and previous state
        x = torch.cat([measurement, previous_state], dim=-1)

        # Forward pass through the network
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        state_estimate = self.fc3(x)

        return state_estimate


def generate_data(num_steps, theta_true, x_values, Q, R):
    """
    Generate synthetic data for the model
    θ_{k+1} = θ_k + q_k
    y_k = x_k * θ_k + w_k
    """
    theta = np.zeros(num_steps)
    y = np.zeros(num_steps)

    # Initialize
    theta[0] = theta_true

    for k in range(num_steps - 1):
        # Process noise
        q_k = np.random.normal(0, np.sqrt(Q))

        # State update
        theta[k + 1] = theta[k] + q_k

        # Measurement noise
        w_k = np.random.normal(0, np.sqrt(R))

        # Measurement
        y[k] = x_values[k] * theta[k] + w_k

    # Final measurement
    w_k = np.random.normal(0, np.sqrt(R))
    y[-1] = x_values[-1] * theta[-1] + w_k

    return theta, y


def main():
    # Parameters
    num_steps = 200
    Q = 0.1  # Process noise variance
    R = 0.5  # Measurement noise variance
    theta_true = 2.0  # True initial state

    # Generate x values (known inputs)
    x_values = np.random.normal(1, 0.5, num_steps)

    # Generate data
    theta_true_seq, measurements = generate_data(num_steps, theta_true, x_values, Q, R)

    # Initialize KalmanNet
    model = KalmanNetTSP()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Convert to tensors
    measurements_tensor = torch.FloatTensor(measurements).unsqueeze(1)
    x_values_tensor = torch.FloatTensor(x_values).unsqueeze(1)

    # Training
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Initialize state estimate
        state_estimate = torch.zeros(1, 1)
        state_estimates = []

        # Process each time step
        for k in range(num_steps):
            # Create input (measurement and previous state)
            measurement_input = measurements_tensor[k].unsqueeze(0)

            # Forward pass
            state_estimate = model(measurement_input, state_estimate)
            state_estimates.append(state_estimate)

        # Calculate loss (compare with true state)
        state_estimates_tensor = torch.cat(state_estimates)
        true_states_tensor = torch.FloatTensor(theta_true_seq).unsqueeze(1)

        loss = criterion(state_estimates_tensor, true_states_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        state_estimate = torch.zeros(1, 1)
        estimated_states = []

        for k in range(num_steps):
            measurement_input = measurements_tensor[k].unsqueeze(0)
            state_estimate = model(measurement_input, state_estimate)
            estimated_states.append(state_estimate.item())

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(theta_true_seq, 'b-', label='True State')
    plt.plot(measurements, 'r.', label='Measurements', alpha=0.5)
    plt.plot(estimated_states, 'g-', label='Estimated State')
    plt.title('KalmanNet-TSP State Estimation')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
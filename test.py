import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralCA(nn.Module):
    def __init__(self, n_states, n_kernels, kernel_size):
        super(NeuralCA, self).__init__()
        self.n_states = n_states
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size

        # Learnable kernels
        self.kernels = nn.Parameter(torch.randn(n_kernels, 1, kernel_size, kernel_size))
        # Learnable reactants and products
        self.reactants = nn.Parameter(torch.randn(n_kernels))
        self.products = nn.Parameter(torch.randn(n_kernels))

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Convolve input with each kernel
        outputs = []
        for i in range(self.n_kernels):
            kernel = self.kernels[i:i+1]
            N = nn.functional.conv2d(x, kernel, padding=self.kernel_size//2)
            # Differentiable condition using sigmoid approximation
            condition = torch.sigmoid(10 * (N - self.reactants[i]))
            # Update rule
            update = condition * (self.products[i] - x)
            outputs.append(update)

        # Sum updates from all kernels
        delta = torch.sum(torch.stack(outputs), dim=0)
        x = x + delta
        # Ensure the state remains within bounds
        x = torch.clamp(x, 0.0, 1.0)
        return x

def train_model(ca_model, initial_state, target_state, num_iterations, learning_rate, steps_per_sample):
    optimizer = optim.Adam(ca_model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    ca_model.train()
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Reset the state for each training sample
        state = initial_state.clone().to(device)

        # Simulate the CA for a number of steps
        for _ in range(steps_per_sample):
            state = ca_model(state)

        loss = loss_fn(state, target_state)
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")

    return ca_model

if __name__ == '__main__':
    # Configuration
    ratio = 16/9
    Y = 100  # Height
    X = int(Y / ratio)  # Width
    UPDATES = 100  # Number of updates for visualization
    n_states = 1  # Number of states
    n_kernels = 10  # Number of kernels
    kernel_size = 3  # Kernel size
    learning_rate = 0.001  # Learning rate
    num_iterations = 1000  # Training iterations
    steps_per_sample = 10  # Steps per training sample

    print(f'X: {X}, Y: {Y}')

    # Seed for reproducibility
    seed = np.random.randint(0, 100000000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'Seed: {seed}')

    # Initialize the CA model
    ca_model = NeuralCA(n_states=n_states, n_kernels=n_kernels, kernel_size=kernel_size).to(device)

    # Initial state: center pixel is active
    initial_state = torch.zeros(1, 1, Y, X).to(device)
    initial_state[0, 0, Y//2, X//2] = 1.0

    # Target state: define a pattern or use a random state
    # For demonstration, let's create a target state with a larger activated area
    target_state = torch.zeros(1, 1, Y, X).to(device)
    target_state[0, 0, Y//2 - 5:Y//2 + 5, X//2 - 5:X//2 + 5] = 1.0

    # Training the model
    print("Training the model...")
    ca_model = train_model(ca_model, initial_state, target_state, num_iterations, learning_rate, steps_per_sample)

    # Running the CA for visualization
    print("Generating CA evolution...")
    ca_model.eval()
    states = [initial_state.detach().cpu().numpy()]
    state = initial_state.clone()
    with torch.no_grad():
        for i in tqdm(range(UPDATES)):
            state = ca_model(state)
            states.append(state.detach().cpu().numpy())

    # Preparing images for GIF
    images = []
    for state in states:
        img = state[0, 0]
        img = (img * 255).astype(np.uint8)
        images.append(img)

    # Saving the GIF
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = '_MEDIA_New'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    imageio.mimsave(f'{folder_name}/{date}_{seed}.gif', images, duration=0.1)
    print(f"Saved GIF to {folder_name}/{date}_{seed}.gif")

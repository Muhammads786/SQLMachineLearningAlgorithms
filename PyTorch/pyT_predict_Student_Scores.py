# Step 1: Import libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Step 2: Generate synthetic data (Study hours vs Exam scores)
torch.manual_seed(0)
study_hours = torch.linspace(1, 10, 50).reshape(-1, 1)
actual_scores = 9 * study_hours + 10 + torch.randn(study_hours.size()) * 25  #y = 9x + 10 + noise

# Step 3: Define the neural network model
class StudentScoreNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),  # Input layer → 10 hidden neurons
            nn.ReLU(),         # Activation function
            nn.Linear(10, 1)   # Output layer
        )

    def forward(self, x):
        return self.net(x)

model = StudentScoreNN()

# Step 4: Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 5: Train the model
epochs = 500
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(study_hours)
    loss = criterion(predictions, actual_scores)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Step 6: Inference
model.eval()

with torch.no_grad():
    predicted_scores = model(study_hours)
    


    

# Step 7: Plot results
plt.figure(figsize=(10, 6))
plt.scatter(study_hours.numpy(), actual_scores.numpy(), label='Actual Scores')
plt.plot(study_hours.numpy(), predicted_scores.numpy(), color='green', label='Predicted Scores')
plt.title('Student Score Prediction from Study Hours')
plt.xlabel('Study Hours')
plt.ylabel('Exam Scores')
plt.legend()
plt.grid(True)
plt.show()

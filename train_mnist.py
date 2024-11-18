import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer
from torchvision import transforms, datasets
import wandb

# Hyperparameters
dim_model = 64
num_encoder_layers = 2
num_decoder_layers = 2
dim_ff = 256
max_len = 16  
output_dim = 10
batch_size = 256
patch_size = 7  # 7x7 patches

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, 
                             transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, 
                            transform=transforms.ToTensor())

# Convert entire dataset to tensors in memory
X_train = train_dataset.data.float().view(-1, 784) / 255.0  # Flatten and normalize
y_train = train_dataset.targets
X_test = test_dataset.data.float().view(-1, 784) / 255.0
y_test = test_dataset.targets

# Model, loss, and optimizer
model = Transformer(
    dim_model=dim_model,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_ff=dim_ff,
    max_len=max_len,
    output_dim=output_dim
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize wandb
wandb.init(
    project="mnist-transformer-v2",
    config={
        "dim_model": dim_model,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "dim_ff": dim_ff,
        "max_len": max_len,
        "output_dim": output_dim,
        "learning_rate": 0.0001,
        "epochs": 30,
        "batch_size": batch_size
    }
)

# Training loop
num_epochs = 30
n_samples = len(X_train)
n_batches = n_samples // batch_size

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # Process mini-batches
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_x = X_train[start_idx:end_idx].to(device)  # [batch_size, 784]
        batch_y = y_train[start_idx:end_idx].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch_x)  # No need for tgt
        
        # Compute loss
        loss = criterion(output, batch_y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        accuracy = (predicted == batch_y).float().mean()
        
        # Log metrics
        wandb.log({
            "batch_loss": loss.item(),
            "batch_accuracy": accuracy.item(),
            "batch": i + epoch * n_batches
        })
        
        if i % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{n_batches}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / n_batches
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Process test data in batches
    test_predictions = []
    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i+batch_size]
        output = model(batch_x)  # Use the same format as in training
        _, predicted = torch.max(output, 1)
        test_predictions.extend(predicted.cpu().numpy())
    
    test_predictions = torch.tensor(test_predictions)
    accuracy = (test_predictions == y_test.cpu()).float().mean()
    print(f'Test Accuracy: {accuracy.item()*100:.2f}%')
    wandb.log({"final_test_accuracy": accuracy.item()})

    # Save the model
    torch.save(model.state_dict(), 'mnist_transformer.pth')
    wandb.save('mnist_transformer.pth')  # This will upload the model file to wandb

    # Create a table for test predictions
    columns = ["image", "true_label", "predicted_label"]
    test_results = wandb.Table(columns=columns)
    
    # Log example predictions with enhanced wandb logging
    for i in range(min(100, len(X_test))):  # Log first 100 test examples
        img = X_test[i].cpu().view(28, 28).numpy()
        true_label = y_test[i].cpu().item()
        pred_label = test_predictions[i].item()
        
        # Add row to the predictions table
        test_results.add_data(
            wandb.Image(img),
            true_label,
            pred_label
        )
    
    # Log the table to wandb
    wandb.log({"test_predictions": test_results})

wandb.finish()

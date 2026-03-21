import torch
import time
from utils import splitTrainVal
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from utils import VisionDataset

def trainTrajectoryModel(model, input_data, output_data, device, num_epochs=300, batch_size=64, learning_rate=0.001):
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"logs/{timestamp}")

    train_input, train_output, val_input, val_output = splitTrainVal(input_data, output_data)

    train_input = torch.tensor(train_input, dtype=torch.float32).to(device)
    train_output = torch.tensor(train_output, dtype=torch.float32).to(device)
    val_input = torch.tensor(val_input, dtype=torch.float32).to(device)
    val_output = torch.tensor(val_output, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(train_input, train_output), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_input, val_output), batch_size=batch_size)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_input, batch_output in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_input)
            loss = criterion(predictions, batch_output)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_input_batch, val_output_batch in val_loader:
                val_predictions = model(val_input_batch)
                val_loss += criterion(val_predictions, val_output_batch).item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "trajectory_model.pth")



def trainVisionModel(model, input_data, output_data, device, num_epochs=100, batch_size=8, learning_rate=0.001):
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"logs/{timestamp}")

    train_input, train_output, val_input, val_output = splitTrainVal(input_data, output_data)

    train_dataset = VisionDataset(train_input, train_output)
    val_dataset = VisionDataset(val_input, val_output)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_input, batch_output in train_loader:
            batch_input, batch_output = batch_input.to(device), batch_output.to(device)
            optimizer.zero_grad()
            predictions = model(batch_input)
            loss = criterion(predictions, batch_output)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_input_batch, val_output_batch in val_loader:
                val_input_batch, val_output_batch = val_input_batch.to(device), val_output_batch.to(device)
                val_predictions = model(val_input_batch)
                val_loss += criterion(val_predictions, val_output_batch).item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "vision_model.pth")
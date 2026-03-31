import time
import torch
from utils import splitTrainVal
from utils import VisionDataset, FusionDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

IMAGE_SIZE = (288, 512)

def trainTrajectoryModel(model, input_data, output_data, device, num_epochs=100, batch_size=64, learning_rate=0.001, group_ids=None, model_name="trajectory_model.pth"):
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"logs/{timestamp}")

    train_input, train_output, val_input, val_output = splitTrainVal(input_data, output_data, group_ids=group_ids)

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
            torch.save(model.state_dict(), model_name)



def trainVisionModel(
    model,
    input_data,
    output_data,
    device,
    num_epochs=60,
    batch_size=8,
    learning_rate=0.001,
    group_ids=None,
    loss_type="smooth_l1",
    weight_decay=1e-4,
    grad_clip_norm=1.0,
    early_stopping_patience=10,
    model_name="vision_model.pth",
):
    model.to(device)
    if loss_type == "mse":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.SmoothL1Loss(beta=0.02)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"logs/{timestamp}")

    train_input, train_output, val_input, val_output = splitTrainVal(input_data, output_data, group_ids=group_ids)

    train_dataset = VisionDataset(train_input, train_output, image_size=IMAGE_SIZE, augment=True)
    val_dataset = VisionDataset(val_input, val_output, image_size=IMAGE_SIZE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float('inf')
    epochs_without_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_input, batch_output in train_loader:
            batch_input, batch_output = batch_input.to(device), batch_output.to(device)
            optimizer.zero_grad()
            predictions = model(batch_input)
            loss = criterion(predictions, batch_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
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
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs}, LR: {current_lr:.6f}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), model_name)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break

    writer.close()


def trainFusionModel(model, input_data, output_data, device, num_epochs=30, batch_size=8, learning_rate=0.001, group_ids=None, model_name="fusion_model.pth"):
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"logs/{timestamp}")

    train_input, train_output, val_input, val_output = splitTrainVal(input_data, output_data, group_ids=group_ids)
    
    train_dataset = FusionDataset(train_input, train_output, image_size=IMAGE_SIZE, augment=True)
    val_dataset = FusionDataset(val_input, val_output, image_size=IMAGE_SIZE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for (batch_images, batch_traj), batch_output in train_loader:
            batch_images = batch_images.to(device)
            batch_traj = batch_traj.to(device)
            batch_output = batch_output.to(device)
            optimizer.zero_grad()
            predictions = model(batch_images, batch_traj)
            loss = criterion(predictions, batch_output)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (val_images_batch, val_traj_batch), val_output_batch in val_loader:
                val_images_batch = val_images_batch.to(device)
                val_traj_batch = val_traj_batch.to(device)
                val_output_batch = val_output_batch.to(device)
                val_predictions = model(val_images_batch, val_traj_batch)
                val_loss += criterion(val_predictions, val_output_batch).item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_name)
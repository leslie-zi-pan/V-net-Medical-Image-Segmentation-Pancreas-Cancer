import copy
import torch
from monai.losses import DiceCELoss
from torch import optim
from HelperFunctions import *


# Method to tune the weight decay hyper parameter
def tune_weight_decay_network(training_loader, network, weights, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), root_path=''):
    network.cuda(device)

    weights_plots = {}

    # Train network per weight to tune
    for weight in weights:
        network_cur_weight_iter = copy.deepcopy(network)

        optimizer = optim.Adam(network_cur_weight_iter.parameters(), lr=5e-5, weight_decay=weight)
        dice_ce_loss = DiceCELoss(include_background=False, ce_weight=CE_WEIGHTS)

        # Train the network
        network_cur_weight_iter.train(True)

        train_step = 1
        batch_loss = {}
        for epoch in range(3):
            for idx, batch_data in enumerate(training_loader):
                print(f'Weight: {weight} \tTraining Step: {train_step}/{len(training_loader)}')

                torch.cuda.empty_cache()  # Clear any unused variables
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"]  # Only pass to CUDA when required - preserve memory

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Feed input data into the network to train
                outputs = network_cur_weight_iter(inputs)

                # Input no longer in use for current iteration - clear from CUDA memory
                inputs = inputs.cpu()
                torch.cuda.empty_cache()

                # labels to CUDA
                labels = batch_data["label"].to(device)
                torch.cuda.empty_cache()

                # Calculate DICE CE loss, permute tensors to correct dimensions
                loss = dice_ce_loss(outputs.permute(0, 1, 3, 4, 2), labels.permute(0, 1, 3, 4, 2))

                # List of losses for current batch
                batch_loss[train_step - 1] = loss.detach().cpu().numpy()

                # Clear CUDA memory
                labels = labels.cpu()
                torch.cuda.empty_cache()

                # Backward pass
                loss.backward()

                # Optimize
                optimizer.step()

                train_step += 1

            # Store loss against the weight decay parameter
            weights_plots[str(weight)] = batch_loss

    return weights_plots


def train_network(training_loader, val_loader, network, pre_load_training=False, checkpoint_name='', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), root_path='', EPOCHS=10):
    network.cuda(device)

    optimizer = optim.Adam(network.parameters(), lr=5e-5, weight_decay=1e-3)
    # COMMENTED OUT - scheduler to increment the optimizer learning rates.
    # steps = lambda epoch: 1.25
    # scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, steps)

    dice_ce_loss = DiceCELoss(include_background=False, ce_weight=CE_WEIGHTS)

    epoch_checkpoint = 0

    losses = {}
    val_losses = {}

    # Test Learning rate dictionary for visualization
    scheduler_learning_rate_dict = {}

    if pre_load_training:
        checkpoint = torch.load(root_path + f'/{checkpoint_name}.pt')
        epoch_checkpoint = checkpoint['epoch'] + 1
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        losses = checkpoint['losses']
        val_losses = checkpoint['val_losses']
        # Only used to find learning rate
        # scheduler_learning_rate_dict = checkpoint['scheduler_learning_rate_dict']

    # Train the network
    for epoch in range(epoch_checkpoint, EPOCHS):
        network.train(True)

        print(f'losses: {losses}')
        print(f'val losses {val_losses}')

        train_step = 1
        batch_loss = []

        for batch_data in training_loader:
            print(f'Epoch {epoch}\tTraining Step: {train_step}/{len(training_loader)}')

            torch.cuda.empty_cache()  # Clear any unused variables
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"]  # Only pass to CUDA when required - preserve memory

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Feed input data into the network to train
            outputs = network(inputs)

            # Input no longer in use for current iteration - clear from CUDA memory
            inputs = inputs.cpu()
            torch.cuda.empty_cache()

            # labels to CUDA
            labels = batch_data["label"].to(device)
            torch.cuda.empty_cache()

            # Calculate DICE CE loss, permute tensors to correct dimensions
            loss = dice_ce_loss(outputs.permute(0, 1, 3, 4, 2), labels.permute(0, 1, 3, 4, 2))

            # COMMENTED OUT - Store learning rate variables and plot to fine tune lr hyperparamter
            # current_learning_rate = optimizer.param_groups[0]['lr']
            # print(f'type dict {type(scheduler_learning_rate_dict)}. type loss {type(loss)}')
            # scheduler_learning_rate_dict[current_learning_rate] = loss

            # List of losses for current batch
            batch_loss.append(loss.detach().cpu().numpy())

            # Clear CUDA memory
            labels = labels.cpu()
            torch.cuda.empty_cache()

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # COMMENTED OUT - UPDATE OPTIMIZER LEARNING RATES TO FIND BEST LEARNING RATE
            # Used for testing best lr
            # scheduler.step()

            train_step += 1

        # COMMENTED OUT - PLOT LOSS CHANGES WITH LEARNING RATES
        # =======================================================
        # Plot losscheduler_learning_rate_list = sorted(scheduler_learning_rate_dict.items())
        # x, y = zip(*scheduler_learning_rate_list)

        # plt.xscale('log')
        # plt.plot(x, y)
        # plt.xlabel('Learning Rate')
        # plt.ylabel('Loss')
        # plt.title('Training losses with varying learning rate')
        # plt.show()ses against learning rate
        # =======================================================

        # Get average loss for current batch
        losses[epoch] = np.mean(batch_loss)
        print(f'train losses {batch_loss} \nmean loss {losses[epoch]}')

        if epoch % 2 == 0:
            # Set network to eval mode
            network.train(False)
            # Disiable gradient calculation and optimise memory
            with torch.no_grad():
                # Initialise validation loss
                dice_ce_test_loss = 0
                for i, batch_data in enumerate(val_loader):
                    # Get inputs and labels from validation set
                    inputs = batch_data["image"].to(device)
                    labels = batch_data["label"]

                    # Make prediction
                    # sw_batch_size = 2
                    # roi_size = (96, 96, 16)
                    # outputs = sliding_window_inference(
                    #   inputs, roi_size, sw_batch_size, network
                    # )
                    outputs = network(inputs)

                    # Memory optimization
                    inputs = inputs.cpu()
                    torch.cuda.empty_cache()
                    labels = batch_data["label"].to(device)

                    # Accumulate DICE CE loss validation error
                    dice_ce_test_loss += dice_ce_loss(outputs.permute(0, 1, 3, 4, 2), labels.permute(0, 1, 3, 4, 2))

                # Get average validation DICE CE loss
                val_losses[epoch] = dice_ce_test_loss / i

                # Print errors
                print(
                    "==== Epoch: " + str(epoch) +
                    " | DICE CE loss: " + str(numpy_from_tensor(dice_ce_test_loss / i)) +
                    " | Total Loss: " + str(numpy_from_tensor((
                                                                  dice_ce_test_loss) / i)) + " =====")  # This is redundant code but will keep here incase we add more losses

                # View slice at halfway point
                half = outputs.shape[2] // 2

                # Show predictions for current iteration
                view_slice(numpy_from_tensor(inputs[0, 0, half, :, :]), f'Input Image Epoch {epoch}')
                view_slice(numpy_from_tensor(outputs[0, 0, half, :, :]), f'Predicted Background Epoch {epoch}')
                view_slice(numpy_from_tensor(outputs[0, 1, half, :, :]), f'Predicted Pancreas Epoch {epoch}')
                view_slice(numpy_from_tensor(outputs[0, 2, half, :, :]), f'Predicted Cancer Epoch {epoch}')
                view_slice(numpy_from_tensor(labels[0, 0, half, :, :]), f'Labels Background Epoch {epoch}')
                view_slice(numpy_from_tensor(labels[0, 1, half, :, :]), f'Labels Pancreas Epoch {epoch}')
                view_slice(numpy_from_tensor(labels[0, 2, half, :, :]), f'Labels Cancer Epoch {epoch}')

        # Save training checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'losses': losses,
            'val_losses': val_losses,
            # 'scheduler_learning_rate_dict':scheduler_learning_rate_dict
        }, root_path + f'/{checkpoint_name}.pt')

        # Confirm current epoch trained params are saved
        print(f'Saved for epoch {epoch}')

    return network
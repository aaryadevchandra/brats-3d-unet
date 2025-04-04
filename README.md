# 3D UNet for BraTS 2024 Post-Treatement Glioma Detection

The BraTS Vanilla 3D U-Net project involved developing a 3D U-Net model from scratch to segment enhancing tumors (ET) in the BraTS 2024 Post-Treatment Glioma Dataset. The initial model achieved a 70-75% Intersection over Union (IoU) for whole-tumor segmentation.




## Tech Stack

**Deep Learning:** PyTorch

**MLOps:** MLFlow, Optuna, WandB


## Screenshots

![Train Prediction](https://cdn.discordapp.com/attachments/762386959389556760/1333285697574141952/image.png)

![Test Prediction](https://cdn.discordapp.com/attachments/762386959389556760/1333285989338054707/image.png)



## Hyperparameters Tuning (Optuna)

In this project, hyperparameter optimization was performed using **Optuna**, an open-source framework for hyperparameter search. The optimization focused on tuning the learning rate, weight decay, and beta parameters of the **Adam optimizer** for training the 3D U-Net model. Below are the details of the hyperparameters explored and their significance:

### Hyperparameters
1. **Learning Rate (`lr`)**
   - **Range:** 0.05 to 0.1
   - **Description:** Controls the step size in updating the model's weights. Higher values speed up convergence but risk overshooting the optimum, while lower values ensure gradual convergence.

2. **Weight Decay (`weight_decay`)**
   - **Range:** 1e-5 to 1e-4
   - **Description:** Adds regularization to the model by penalizing large weights, helping to prevent overfitting.

3. **Beta1 (`beta1`)**
   - **Range:** 0.7 to 0.99
   - **Description:** The exponential decay rate for the first moment estimate in the Adam optimizer, influencing the momentum of gradients.

4. **Beta2 (`beta2`)**
   - **Range:** 0.95 to 0.99
   - **Description:** The exponential decay rate for the second moment estimate in the Adam optimizer, controlling the moving average of gradient variance.

### Objective Function
The objective function defined for Optuna trials evaluates the **loss** using the 3D U-Net model. The procedure includes:
1. Sampling hyperparameters using `trial.suggest_float`.
2. Initializing the Adam optimizer with sampled values.
3. Running a single forward pass on the training data batch to compute the loss.

```python
def objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 0.05, 0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4)
    beta1 = trial.suggest_float("beta1", 0.7, 0.99)
    beta2 = trial.suggest_float("beta2", 0.95, 0.99)
    
    optimizer = torch.optim.Adam(unet3d.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    
    data, masks = next(iter(train_dataloader))
    data, masks = data.to(device), masks.to(device)
    
    loss = loss_fn(unet3d(data), masks)
    
    del data
    del masks
    
    return loss
```

### Default Parameters
For initial experiments, the Adam optimizer was initialized with the following default parameters:
- `lr = 3e-4`
- `betas = (0.9, 0.999)`
- `weight_decay = 1e-4`

```python
optimizer = torch.optim.Adam(unet3d.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4)
```

## Loss Functions

The following loss functions were used in this project:  
- **Dice Loss:** Handles class imbalance and improves overlap accuracy.  
- **Tversky Loss:** Balances false positives and false negatives, with `alpha=0.4` and `beta=0.6`.  
- **Binary Cross-Entropy (BCE) Loss:** Measures the difference between predicted and actual binary labels.


## Metrics

- **Intersection over Union (IoU):** 70-75% Used to evaluate the overlap between predicted and ground truth segmentation masks.


## Hardware and Resource Requirements

- **GPU Used:** NVIDIA H100 PCIe with MIG configuration, providing up to 40GB of GPU memory.  
- **Reason for Resource Requirement:**  
  The project involves processing large NIfTI files, which demand high RAM and GPU memory for training. Most local machines lack the computational power and memory capacity required to handle the dataset efficiently, making access to high-performance computing resources essential for this work.







# Autonomous Warehouse Robot: Multi-Label Image Tagging

## Overview
This project implements a **Multi-Label Image Classification** system for an autonomous warehouse transport robot. 

Unlike traditional object detection models (like YOLO) which draw bounding boxes around objects, this system predicts the *presence* or *absence* of multiple objects simultaneously without localizing them. Because the robot's primary task is transporting materials from point A to point B, full object detection is often unnecessarily compute-heavy. Tagging the environment provides sufficient semantic awareness (e.g., "forklift", "pallet", "box") at a fraction of the computational cost, allowing the robot to reserve processing power for crucial navigation and obstacle avoidance tasks.

---

## 1. Mathematical Foundation: Multi-Label vs. Multi-Class

In traditional **Multi-Class Classification**, an image belongs to exactly *one* category. The network outputs a vector of raw scores (logits) $\mathbf{z} = [z_1, z_2, \dots, z_N]$, and a **Softmax** function is applied:

$$ \hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}} $$

Softmax forces the probabilities of all classes to sum to $1.0$, creating *mutual exclusivity*.

In **Multi-Label Classification** (our use case), an image can contain zero, one, or multiple objects simultaneously (e.g., an image can have both a `box` AND a `pallet` AND a `forklift`). Mutual exclusivity is detrimental here. Instead, we treat each of the $N$ classes as an independent binary classification problem.

We apply the **Sigmoid** activation function to each logit independently:

$$ \sigma(z_i) = \frac{1}{1 + e^{-z_i}} $$

This squashes each logit $z_i$ into a probability $p_i \in [0, 1]$, where $p_i$ represents the confidence that class $i$ is present in the image. The probabilities do *not* sum to 1.

---

## 2. The Loss Function: Binary Cross Entropy (BCE)

To train the network, we need to penalize the model when its predicted probabilities $\sigma(z_i)$ deviate from the ground truth labels $y_i \in \{0, 1\}$. 

For a single class $i$, the Binary Cross Entropy loss is defined as:

$$ L_i = - \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right] $$

*   If the object is present ($y_i = 1$), the second term vanishes, and the loss is $- \log(\sigma(z_i))$. To minimize this, the network must push $\sigma(z_i)$ towards $1$.
*   If the object is absent ($y_i = 0$), the first term vanishes, and the loss is $- \log(1 - \sigma(z_i))$. To minimize this, the network must push $\sigma(z_i)$ towards $0$.

The total loss for a single image across all $N$ classes is the average of the individual class losses:

$$ \mathcal{L}_{BCE} = \frac{1}{N} \sum_{i=1}^{N} L_i $$

### Numerical Stability (`BCEWithLogitsLoss`)
In PyTorch, applying `torch.sigmoid` followed by `nn.BCELoss` is numerically unstable due to floating-point precision limits when calculating logarithms of values very close to 0. 

Instead, this project utilizes `nn.BCEWithLogitsLoss`, which mathematically combines the Sigmoid and BCE operations into a single layer using the **Log-Sum-Exp trick**:

$$ L_i = \max(z_i, 0) - z_i y_i + \log(1 + e^{-|z_i|}) $$

This ensures gradients do not vanish or explode during backpropagation.

---

## 3. Network Architecture

The project supports multiple feature extractors (backbones) configured via `config.yaml`:
1.  **ResNet-50**: Deep residual network utilizing skip-connections to prevent vanishing gradients.
2.  **MobileNetV3**: Extremely lightweight and designed specifically for edge devices and mobile robots using depthwise separable convolutions.
3.  **EfficientNet-B0**: Balances depth, width, and resolution using a compound scaling method.

### The Classification Head
Regardless of the backbone, the final layers of the architecture operate as follows:

1.  **Feature Extraction**: The backbone transforms a $(3 \times H \times W)$ image into a high-dimensional feature map (e.g., $1280 \times \frac{H}{32} \times \frac{W}{32}$).
2.  **Global Average Pooling (GAP)**: Computes the spatial average of each feature map channel, drastically reducing parameters and preventing overfitting:
    $$ f_c = \frac{1}{H' W'} \sum_{x=1}^{H'} \sum_{y=1}^{W'} A_{c, x, y} $$
    where $A_c$ is the $c$-th channel of the feature map.
3.  **Fully Connected Layer**: A linear transformation maps the pooled features to the $N=25$ class logits:
    $$ \mathbf{z} = \mathbf{W} \mathbf{f} + \mathbf{b} $$
    where $\mathbf{W} \in \mathbb{R}^{25 \times 1280}$ and $\mathbf{b} \in \mathbb{R}^{25}$.

---

## 4. Optimization & Data Augmentation

### Optimizer: AdamW
The network is optimized using **AdamW** (Adam with decoupled Weight Decay). Adam uses exponentially moving averages of the gradient ($m_t$) and squared gradient ($v_t$) to adapt the learning rate for each parameter. AdamW improves generalization by decoupling the $L_2$ regularization penalty from the gradient update:

$$ \theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right) $$

where $\eta$ is the learning rate and $\lambda$ is the weight decay factor.

### Data Augmentation
To ensure the robot's vision system is robust to varying warehouse lighting and camera angles, the following stochastic transformations are applied during training:
*   **RandomResizedCrop**: Simulates objects appearing at different distances.
*   **ColorJitter**: Alters brightness, contrast, and saturation to simulate shadows, glare, and different lighting fixtures.
*   **Normalization**: Standardizes pixel values using ImageNet mean and standard deviation: $x_{norm} = \frac{x - \mu}{\sigma}$.

---

## 5. Repository Structure

```
├── README.md                           # Project documentation and mathematical foundations
├── src/
│   ├── config.yaml                     # Centralized hyperparameters and dataset paths
│   ├── data/
│   │   └── dataset.py                  # PyTorch Dataset for loading multi-label CSVs and images
│   ├── models/
│   │   └── tagging_model.py            # Neural network architecture definitions
│   └── training/
│       └── train.py                    # Training loop, evaluation, and checkpointing logic
├── build_notebook.py                   # Script to generate Jupyter notebooks programmatically
├── inference.ipynb                     # Interactive notebook for running live predictions
└── checkpoints/                        # Saved weights (best_model.pth)
```

## 6. How to Use

### Configuration
Update the `src/config.yaml` file to point to your image directories and target CSV files. You can also adjust training hyperparameters (epochs, learning rate, batch size) and select the model architecture (`mobilenet_v3`, `resnet50`, `efficientnet_b0`).

### Training
Execute the training pipeline. The script automatically handles GPU allocation, validation tracking, and saves the best model based on validation loss.
```bash
python3 src/training/train.py
```

### Inference
Open `inference.ipynb` to test the model on unseen images. The notebook loads `checkpoints/best_model.pth`, processes an image, applies the Sigmoid activation, and filters predictions based on a defined confidence threshold (e.g., > 0.5).

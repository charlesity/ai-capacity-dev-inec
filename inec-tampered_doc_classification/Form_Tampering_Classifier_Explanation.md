# Form Tampering Classifier (PyTorch + Streamlit)

<img src="screenshots/Screenshot%202026-07-05%20at%2020.51.25.png">
<img src="screenshots/Screenshot%202026-07-05%20at%2020.51.45.png">
This guide explains the code section by section.

## Note: This code is not meant to be written in app.py file
- Open app.py file in your favourite Code Editor and follow the steps.
- run the following code in cmd/terminal
``` python
pip install torch torchvision streamlit matplotlib pillow
```
## 1. Import Libraries

These libraries provide file handling, UI, deep learning, datasets,
image processing, and plotting.

``` python
import os
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import matplotlib.pyplot as plt
```

## 2. Configuration

Defines dataset location, model filename, GPU/CPU selection, and image
preprocessing.

``` python
DATA_DIR = "dataset"
MODEL_PATH = "tamper_resnet34.pth"
DEVICE = torch.device("cuda" if torch.torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

infer_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Explanation**

-   `DATA_DIR` stores training images.
-   `MODEL_PATH` stores the trained model.
-   `DEVICE` automatically uses CUDA if available.
-   Training uses augmentation.
-   Testing uses only resizing and normalization.

## 3. Model Definition

Creates a pretrained ResNet-34 model and replaces the classifier.

``` python
def get_model():
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model
```

**Explanation**

-   Loads pretrained ImageNet weights.
-   Freezes feature extractor.
-   Trains only the final layer.
-   Output classes: Authentic and Tampered.

## 4. Streamlit Interface

Creates the application title and checks dataset availability.

``` python
st.title("🕵️‍♂️ Form Tampering Classifier")
st.write("Train a frozen ResNet-34 on your seed images and test live documents.")

if not os.path.exists(DATA_DIR):
    st.error(f"Please create a '{DATA_DIR}/' folder containing 'authentic/' and 'tampered/' subdirectories.")
    st.stop()

tab1, tab2 = st.tabs(["🏋️‍♂️ Model Training", "🔍 Document Testing"])
```

## 5. Training Tab

Loads the dataset, trains the network, records metrics and saves the
model.

``` python
dataset = ImageFolder(root=DATA_DIR, transform=train_transforms)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = get_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3)
```

Training loop:

``` python
for epoch in range(epochs):
    model.train()

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

Save model:

``` python
torch.save(model.state_dict(), MODEL_PATH)
```

Plot metrics:

``` python
st.pyplot(fig)
```

**Explanation**

-   Loads images.
-   Creates mini-batches.
-   Computes loss.
-   Updates weights.
-   Saves trained parameters.
-   Displays learning curves.

## 6. Testing Tab

Allows users to upload and classify a document.

Upload:

``` python
uploaded_file = st.file_uploader(
    "Upload a form image to inspect...",
    type=["png", "jpg", "jpeg"]
)
```

Prediction:

``` python
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

tensor_img = infer_transforms(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(tensor_img)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, prediction = torch.max(probabilities, dim=1)
```

Display result:

``` python
if result == 'authentic':
    st.success(f"✅ Document Verified: Authentic ({conf_score:.2f}% Confidence)")
else:
    st.error(f"🚨 Security Alert: Tampered/Modified ({conf_score:.2f}% Confidence)")
```

## Run the App
```python
streamlit run app.py
```
## Congrats! You have successfully deployed a form tampering classifier.
## Workflow

1.  Load dataset.
2.  Apply augmentation.
3.  Train ResNet-34.
4.  Save model.
5.  Upload image.
6.  Preprocess image.
7.  Predict authenticity.
8.  Display confidence.

## Key Concepts

-   Transfer Learning
-   Data Augmentation
-   Fine-Tuning
-   Cross Entropy Loss
-   AdamW Optimizer
-   Softmax Probabilities
-   Streamlit User Interface

## Exercise.
-   Replace the images in the dataset folder with your own images
-   The images should be in the same folder.
-   The folder should be named 'authentic' and 'tampered'.
- The higher the number of images in each folder, the better the model will perform.
- 100% Accuracy is not a goal.

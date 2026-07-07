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


DATA_DIR = "dataset"
MODEL_PATH = "tamper_resnet34.pth"
DEVICE = torch.device("cuda" if torch.torch.cuda.is_available() else "cpu")

# Image transformations for both Training (Augmented) and Inference
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

infer_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_model():
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)

    # Freeze the backbone to prevent overfitting on 10 images
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for 2 classes (Authentic vs Tampered)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model


st.title("Form Tampering Classifier")
st.write("Train a frozen ResNet-34 on your seed images and test live documents.")

# Verification step to check if folders exist
if not os.path.exists(DATA_DIR):
    st.error(f"Please create a '{DATA_DIR}/' folder containing 'authentic/' and 'tampered/' subdirectories.")
    st.stop()

tab1, tab2 = st.tabs(["🏋️‍♂️ Model Training", "🔍 Document Testing"])

# --- TAB 1: TRAINING PANEL ---
with tab1:
    st.header("Training Settings")
    epochs = st.slider("Select Training Epochs", min_value=5, max_value=50, value=20, step=5)

    if st.button("Start Training"):
        with st.spinner("Training model with on-the-fly augmentations..."):
            # Load dataset
            dataset = ImageFolder(root=DATA_DIR, transform=train_transforms)
            classes = dataset.classes  # ['authentic', 'tampered']
            st.session_state['classes'] = classes

            # Using small batch size because total dataset size = 10 images
            loader = DataLoader(dataset, batch_size=2, shuffle=True)

            model = get_model().to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3)

            # Tracking loss for plotting
            loss_history = []
            acc_history = []

            # Training loop
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for images, labels in loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                epoch_loss = running_loss / len(dataset)
                epoch_acc = (correct / total) * 100
                loss_history.append(epoch_loss)
                acc_history.append(epoch_acc)

            # Save weights locally
            torch.save(model.state_dict(), MODEL_PATH)
            st.success("Training Complete! Model weights saved successfully.")

            # Plot metrics using Matplotlib
            fig, ax1 = plt.subplots(figsize=(10, 4))

            color = 'tab:red'
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss', color=color)
            ax1.plot(range(1, epochs + 1), loss_history, color=color, label='Loss', marker='o')
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Accuracy (%)', color=color)
            ax2.plot(range(1, epochs + 1), acc_history, color=color, label='Accuracy', marker='s')
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title('Training Progress (Augmented Data Metrics)')
            fig.tight_layout()

            # Display plot in Streamlit
            st.pyplot(fig)

# --- TAB 2: INFERENCE / TESTING PANEL ---
with tab2:
    st.header("Test a Document")

    if not os.path.exists(MODEL_PATH):
        st.info("Please train the model first in the 'Model Training' tab.")
    else:
        uploaded_file = st.file_uploader("Upload a form image to inspect...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Display uploaded document
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Document", use_container_width=True)

            # Run evaluation
            if st.button("Analyze Document Integrity"):
                # Load the trained model architecture and weights
                model = get_model()
                model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()

                # Preprocess uploaded image
                tensor_img = infer_transforms(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    outputs = model(tensor_img)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, prediction = torch.max(probabilities, dim=1)

                # Fetch target label class names dynamically
                class_names = st.session_state.get('classes', ['authentic', 'tampered'])
                result = class_names[prediction.item()]
                conf_score = confidence.item() * 100

                # Visual readout based on prediction result
                if result == 'authentic':
                    st.success(f"✅ Document Verified: **Authentic** ({conf_score:.2f}% Confidence)")
                else:
                    st.error(f"🚨 Security Alert: **Tampered/Modified** ({conf_score:.2f}% Confidence)")
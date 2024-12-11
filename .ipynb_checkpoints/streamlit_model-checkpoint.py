import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18

# Define the number of location classes (update this based on your model)
num_classes = 100  # Example: Number of locations (you can change this)

# Load the pre-trained or fine-tuned model
model = resnet18(pretrained=False)  # Assuming the model is ResNet18
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust num_classes based on your model
model.load_state_dict(torch.load("fine_tuned_model.pth"))  # Load your model's weights
model.eval()  # Set the model to evaluation mode

# Define the image transformation for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to fit the model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Map of predicted class labels to location names (e.g., city names)
location_labels = ["Paris", "London", "New York", "Tokyo", "Berlin", "Sydney", "Rome", "Madrid", "Los Angeles", "San Francisco"]

# Streamlit Interface
st.title("Location Prediction from Image")
st.write("Upload an image and get a predicted location!")

# File uploader for image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image to match the model's expected input format
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict the location from the image
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_label = torch.argmax(outputs, dim=1).item()  # Get the predicted class index

    # Get the location label based on the prediction
    predicted_location = location_labels[predicted_label]
    
    # Show the predicted location
    st.write(f"Predicted Location: {predicted_location}")

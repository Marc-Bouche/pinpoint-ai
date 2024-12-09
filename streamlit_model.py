import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import os

# Define the model class if you have one (replace 'YourModelClass' with your actual model class)
class FineTunedModel(nn.Module):
    def __init__(self):
        super(FineTunedModel, self).__init__()
        # Define your model architecture here
        # Example: A simple CNN with 2 convolution layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 224 * 224, 2)  # 2 output neurons for latitude and longitude
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layer
        x = self.fc(x)
        return x

# Check if the model file exists
model_path = '/Users/mbouch17/Desktop/Personal_Data_Project/PinpointAI/fine_tuned_model.pth'
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    # Load your model architecture
    fine_tuned_model = FineTunedModel()
    
    # Load the trained weights into the model
    fine_tuned_model.load_state_dict(torch.load(model_path))
    fine_tuned_model.eval()  # Set the model to evaluation mode

    # Upload image
    image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if image:
        # Display the uploaded image
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Transform the image for the model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Open the image
        pil_image = Image.open(image)

        # Apply transformations and add batch dimension
        image_tensor = transform(pil_image).unsqueeze(0)

        # Make predictions with no gradient tracking
        with torch.no_grad():
            outputs = fine_tuned_model(image_tensor)

        # Assuming the model output is in the range 0-1 (for latitude and longitude)
        st.write(f"Raw Model Output: {outputs}")

        if torch.all(outputs >= 0) and torch.all(outputs <= 1):
            # Assuming normalized coordinates between 0 and 1
            lat_min, lat_max = -90, 90  # Latitude range
            lon_min, lon_max = -180, 180  # Longitude range

            # Scaling the predicted coordinates to real-world values
            predicted_lat = outputs[0, 0].item() * (lat_max - lat_min) + lat_min
            predicted_lon = outputs[0, 1].item() * (lon_max - lon_min) + lon_min

            st.write(f"Predicted Latitude: {predicted_lat:.4f}")
            st.write(f"Predicted Longitude: {predicted_lon:.4f}")
        else:
            st.write("Model output is outside the expected range (0-1).")

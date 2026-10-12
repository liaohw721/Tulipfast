
import torch
import time
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Load pretrained ResNet-18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Remove classification layer to get image features
model.fc = torch.nn.Identity()
model.eval()

# Load real image
image = Image.open("test_image.jpg").convert("RGB")
preprocess = weights.transforms()
input_tensor = preprocess(image).unsqueeze(0)

# Warm up
with torch.inference_mode():
    for _ in range(10):
        model(input_tensor)

    timings = []

    for _ in range(100):
        start = time.perf_counter()
        features = model(input_tensor)
        timings.append(
            (time.perf_counter() - start) * 1000
        )

print("Feature shape:", features.shape)
print("Average latency:", round(sum(timings)/len(timings), 3), "ms")
print("Model: Pretrained ResNet-18")

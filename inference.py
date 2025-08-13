import torch
import main
from PIL import Image

trasforms = main.transforms
model = main.Vision()
model.load_state_dict(torch.load('vision.pth', map_location='cpu'))
model.eval()

image_p = 'dataset/Acidentada/19700102_143748.jpg'

img = Image.open(image_p).convert('RGB')

i_tensor = trasforms(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(i_tensor)
    _, predicted = torch.max(outputs, 1)
    class_idx = predicted.item()

classes = ['Acidentada', 'Engarrafado', 'Estrada completamente vazia', 'Movimentada', 'Muito acidentado', 'Muito engarrafada', 'um pouco vazia']

i_class = classes[class_idx]

print(i_class)
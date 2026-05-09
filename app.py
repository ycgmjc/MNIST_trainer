import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch.nn.functional as F

from Models import SimpleMNISTCNN

def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SimpleMNISTCNN().to(device)
    
    # load the checkpoint from the training result directory
    checkpoint_path = './Exps/Example_Training_1/model_best.pt'
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        print("Model loaded successfully!")
        return model, device
    except FileNotFoundError:
        print(f"Could not find model weights at {checkpoint_path}.")
        print("Make sure you have run the trainer first!")
        exit()

# load model globally
model, device = load_model()

def predict_digit(img_dict):
    if img_dict is None:
        return "Please draw a digit."
        
    # Gradio 4's Sketchpad returns a dictionary containing 'composite' (the drawn image)
    if isinstance(img_dict, dict):
        img = img_dict['composite']
    else:
        img = img_dict

    # convert image to Grayscale
    img = img.convert('L')
    
    # invert colors (MNIST has white numbers, black backgrounds)
    img = ImageOps.invert(img)
    
    # preprocess
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # convert to [1, 1, 28, 28]
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # run model inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output[0], dim=0)
        
    # format results for Gradio
    confidences = {str(i): float(probabilities[i]) for i in range(10)}
    return confidences

# create the Gradio interface
demo = gr.Interface(
    
    fn=predict_digit,
    # sketchpad
    inputs=gr.Sketchpad(type="pil", label="Draw a digit (0-9) here"), 
    # confidence bar chart
    outputs=gr.Label(num_top_classes=10, label="Model Prediction"),
    title="MNIST Digit Recognizer",
    description="Draw a single digit between 0 and 9 on the canvas below. The application will preprocess your drawing and pass it to your trained PyTorch model.",
    flagging_mode="never"
)

if __name__ == "__main__":
    # launch the web interface
    demo.launch(share=False)
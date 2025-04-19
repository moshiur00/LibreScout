from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def extract_image_details(image):
    inputs = processor(images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        max_length=50,
        num_beams=5,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"BLIP Model Description: {generated_text}")  # Debugging print statement
    return generated_text

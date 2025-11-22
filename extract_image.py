from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch
import easyocr

torch.cuda.empty_cache()
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
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    reader = easyocr.Reader(['en'])
    image_text = ' '.join([item[1] for item in reader.readtext(image)])
    generated_text = "Book Name: "+ image_text + " with caption " + caption
    print(f"BLIP Model Description: {generated_text}")  # Debugging print statement
    return generated_text
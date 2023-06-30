import torch
from PIL import Image
import open_clip
import os
import numpy as np

# Source: https://github.com/openai/CLIP

_path = "/data/photos_sample/"
names = [name for name in os.listdir(_path) if name != ".DS_Store"]

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
tokenizer = open_clip.get_tokenizer('ViT-B-16')

for name in names:

    image = preprocess(Image.open(os.path.join(_path, name))).unsqueeze(0)
    text = tokenizer([ "a photo", 'text', 'a poster', 'a drawing'])

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        ind = np.argmax(text_probs.numpy().squeeze(0))
        
        if ind != 0:
            os.remove(os.path.join(_path, name))
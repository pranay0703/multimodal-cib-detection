from transformers import BertModel, BertTokenizer
import torch
from PIL import Image
import clip

def get_text_embedding(text, model, tokenizer):
    """
    Generates a text embedding using a pre-trained BERT model.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embedding of the [CLS] token
    return outputs.last_hidden_state[:, 0, :].squeeze()

def get_image_embedding(image_path, model, preprocess):
    """
    Generates an image embedding using the CLIP model.
    """
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features.squeeze()

if __name__ == '__main__':
    # This script requires pre-trained models to be loaded.
    
    # --- Text Embedding Example ---
    print("--- Text Embedding Example ---")
    # Load a pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    sample_text = "This is an example sentence for text embedding."
    text_embedding = get_text_embedding(sample_text, model, tokenizer)
    print(f"Sample text: '{sample_text}'")
    print(f"Text embedding shape: {text_embedding.shape}") # Should be [768] for bert-base-uncased

    # --- Image Embedding Example ---
    # This requires an image file. We will create a dummy one for now.
    print("
--- Image Embedding Example ---")
    try:
        # Load the CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Create a dummy image for demonstration
        dummy_image = Image.new('RGB', (224, 224), color = 'red')
        dummy_image_path = 'dummy_image.png'
        dummy_image.save(dummy_image_path)
        
        image_embedding = get_image_embedding(dummy_image_path, clip_model, preprocess)
        print(f"Sample image: '{dummy_image_path}'")
        print(f"Image embedding shape: {image_embedding.shape}") # Should be [512] for ViT-B/32
    
    except Exception as e:
        print(f"Could not run image embedding example: {e}")
        print("This might be because of a missing dependency or issue with model loading.")


 # Use a pipeline as a high-level helper
from transformers import pipeline

text_to_music = pipeline("text-to-audio", model="facebook/musicgen-small", device=torch.device('cuda'))

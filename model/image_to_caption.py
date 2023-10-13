from transformers import pipeline
from utils import read_yaml


class image_to_caption:
    def __init__(self):
        config_file = read_yaml()
        self.model_name = config_file["IMAGETOCAPTION"]["MODEL"]
        self.model_type = config_file["IMAGETOCAPTION"]["TYPE"]
        self.captioner = pipeline(self.model_type, model=self.model_name)

    def image_to_caption(self, image):
        caption = self.captioner(image.get_image())[0]["generated_text"]
        return caption
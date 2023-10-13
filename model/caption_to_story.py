import os
from utils import read_yaml
from mcli.sdk import predict
from transformers import pipeline

class caption_to_story:
    def __init__(self):
        config_file = read_yaml()
        self.api = config_file['MOSAIC_ML']['API']
        self.model = config_file['MOSAIC_ML']['MODEL']
        self.host = config_file['MOSAIC_ML']['HOST']
        # self.story_generator = pipeline(self.model_type, model=self.model_name)

    def set_prompt(self):
        prompt = """Below is an instruction that describes a photo. Write a complete adventure story linked to the caption.
            ### Instruction: Write a story for an image with the caption : {caption}
            ### Response: """
        return prompt

    def set_api(self):
        os.system('mcli set api-key {}'.format(self.api))

    def create_story(self, caption):
        prompt = str(self.set_prompt().replace("{caption}", caption))
        story = predict(os.path.join(self.host, self.model), {"inputs": [prompt], "parameters": {"temperature": .2}})
        return story['outputs'][0].strip()

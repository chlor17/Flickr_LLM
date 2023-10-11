import os
from utils import read_yaml
from mcli.sdk import predict

class caption_to_story:
    def __init__(self):
        config_file = read_yaml()
        self.api = config_file['MOSAIC_ML']['API']
        self.model = config_file['MOSAIC_ML']['MODEL']
        self.host = config_file['MOSAIC_ML']['HOST']

        self.captioner = pipeline(self.model_type, model=self.model_name)

    def set_prompt(self):
        prompt = """Below is an instruction that describes a photo. Write an adventure story linked to the caption.
            ### Instruction: Write a story for an image with the caption : there is a man riding a bike down a winding road.
            ### Response: """
        return prompt

    def set_api(self):
        os.system('/mpt_set_api.sh {}' .format(str(self.api)))

    def create_story(self):
        predict(self.host + self.model, {"inputs": [self.set_prompt()], "parameters": {"temperature": .2}})






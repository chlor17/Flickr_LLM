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

    def set_prompt(self):
        prompt = """Below is an instruction that describes a photo. Write a complete description of the image linked to the caption.
            ### Instruction: Write a complete description of approximately 400 characters for an image with the caption. Image was taken at {datetime}, aperture at f {aperture}, iso {iso}, {exposure} seconds exposure: {caption}
            ### Response: """
        return prompt

    def set_api(self):
        os.system('mcli set api-key {}'.format(self.api))

    def create_story(self, photo):
        prompt = str(self.set_prompt().replace("{caption}", photo.caption[0]['generated_text']))
        prompt = prompt.replace("{datetime}", photo.createdate)
        prompt = prompt.replace("{aperture}", photo.aperture)
        prompt = prompt.replace("{iso}", photo.iso)
        prompt = prompt.replace("{exposure}", photo.exposuretime)
        print(prompt)
        story = predict(os.path.join(self.host, self.model), 
                        {
                            "inputs": [prompt], 
                            "parameters": 
                            {
                                "temperature": .2,
                                "max_new_tokens": 400
                            }
                        }
                        )
        print(story['outputs'])
        return story['outputs'][0].strip()

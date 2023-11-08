# Flickr_LLM

## Overview

This repository contains a data processing project that involves data acquisition from Flickr, image conversion, caption generation, combining data into prompts, description generation, and seamless integration with Flickr for updates. It's a comprehensive solution for managing and enhancing visual content.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Technologies](#technologies)
- [Contributing](#contributing)

## Features

- **Data Acquisition via Flickr:** Easily acquire images and data from Flickr using API. Download to Databricks volumes, can be modified to other paths.

- **Image Conversion:** Convert images to PIL.

- **Caption Generation:** Generate captions for your images using Salesforce/blip-image-captioning-large.

- **Combining Data into Prompts:** Seamlessly combine acquired metadata into prompts for LLM model, mpt-30b-instruct/v1.

- **Description Generation:** Generate descriptive content to enhance your visual assets and provide more context.

- **Integration with Flickr for Updates:** Keep your content up-to-date on Flickr image description with a smooth API integration process.

## Getting Started

1. Start by configuring your Flickr API credentials within the project. I suggest using postman following : https://medium.com/apis-with-valentine/oauth-1-0-authorization-flow-using-flickr-api-and-postman-3-legged-oauth-c2a9b46bd8b9
1. Set up a MosaicML api key, could be changed to a OpenAI model with code modification. Also possible to change to a hugging face model as well.
1. Databricks set up with a cluster with a runtime: 13.3 LTS ML

## Usage

Generate AI generated description to you public image on Flickr

## Technologies

- flickrapi
- PIL
- HuggingFace
- MosaicML

## Contributing

Feel free to add suggestion for improvements !

from flickrapi import FlickrAPI
from utils import read_yaml

class Flickr_conn:
    def __init__(self, key=None, secret=None):
        config_file = read_yaml()
        if key:
            self.key = key
        else:
            self.key = config_file['FLICKR']['API_KEY']
        if secret:
            self.secret = secret
        else:
            self.secret = config_file['FLICKR']['SECRET']

        self.conn = None

    def create_connection(self):
      self.conn = FlickrAPI(self.key, self.secret)
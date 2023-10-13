from flickrapi import FlickrAPI
from utils import read_yaml
from requests_oauthlib import OAuth1Session

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

class Flickr_sess:
    def __init__(self):
        config_file = read_yaml()
        self.flickr = OAuth1Session(config_file['FLICKR']['API_KEY'],
                                    client_secret=config_file['FLICKR']['SECRET'],
                                    resource_owner_key=config_file['FLICKR']['ACCESS_TOKEN'],
                                    resource_owner_secret=config_file['FLICKR']['TOKEN_SECRET'])
        url = 'https://www.flickr.com/services/rest?nojsoncallback=1&format=json&method=flickr.test.login'
        self.flickr.get(url)
    
    def post_description(self, photo):
        url = "https://www.flickr.com/services/rest/?method=flickr.photos.setMeta" 
        photo_id = photo.get_image_id()
        description = photo.story
        print(photo_id, description)
        data={"photo_id":photo_id, "description":description}
        r = self.flickr.post(url, data=data)
        print(r)
        return r
import time
from selenium import webdriver
import time
import requests
import os
from PIL import Image
import io
import hashlib
from scrap_functions import fetch_image_urls, persist_image

DRIVER_PATH = r'/home/martignoni/PycharmProjects/Scraping/chromedriver' //à modifier avec relative path!!!


if __name__ == '__main__':
    wd = webdriver.Chrome(executable_path=DRIVER_PATH)
    queries = ["tomato", "poivron", "carrot"]
    for query in queries:
        wd.get('https://google.com')
        search_box = wd.find_element_by_css_selector('input.gLFyf')
        search_box.send_keys(query)
        links = fetch_image_urls(query, 300, wd)
        images_path = 'data'
        for i in links:
            persist_image(images_path, query, i)
    wd.quit()


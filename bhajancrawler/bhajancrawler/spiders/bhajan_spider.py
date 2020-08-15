# -*- coding: utf-8 -*-

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import src.config as config

class BhajanSpider(CrawlSpider):

    name = "Bhajans"

    start_urls = [
        'https://sairhythms.org/'

    ]

    file_types = ['mp3', 'm4a']

    rules = (
        Rule(LinkExtractor(allow="/song/"), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        def extract_with_xpath(query):
            return response.xpath(query).xpath('string(.)').get()

        bhajan_info = {
            'link': response.xpath("""//div[contains(@class, 'audio-link')]//a[1]""").xpath('text()').get(),
            'deity': extract_with_xpath("""//div[contains(@class, 'deity-row')]//div[2]"""),
            'language': extract_with_xpath("""//div[contains(@class, 'language-row')]//div[2]"""),
            'raaga': extract_with_xpath("""//div[contains(@class, 'raga-title')]"""),
            'beat': extract_with_xpath("""//div[contains(@class, 'beat-row')]//div[2]"""),
            'level': extract_with_xpath("""//div[contains(@class, 'level-row')]//div[2]"""),
            'tempo': extract_with_xpath("""//div[contains(@class, 'tempo-row')]//div[2]""")
        }

        if bhajan_info['link'] is not None:
            if any(f".{key}" in bhajan_info['link'] for key in self.file_types):
                return bhajan_info

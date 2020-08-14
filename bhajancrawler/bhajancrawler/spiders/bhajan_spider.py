# -*- coding: utf-8 -*-
import scrapy

import os

os.chdir('..')

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor


class BhajanSpider(CrawlSpider):

    name = "Bhajans"
    # allowed_domains = ['sairhythms.org/song']
    # base_url = ['"https:/www.sairhythms.org/']

    start_urls = [
        'https://sairhythms.org/'

    ]

    rules = (
        Rule(LinkExtractor(allow="/song/"), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        def extract_with_xpath(query):
            return response.xpath(query).xpath('string(.)').get()

        return {
            'link': response.xpath("""//div[contains(@class, 'audio-link')]//a[1]""").xpath('text()').get(),
            'deity': extract_with_xpath("""//div[contains(@class, 'deity-row')]//div[2]"""),
            'language': extract_with_xpath("""//div[contains(@class, 'language-row')]//div[2]"""),
            'raaga': extract_with_xpath("""//div[contains(@class, 'raga-title')]"""),
            # excludes other ragas for songs with multiple ragas (only includes first)
            # 'raga': extract_with_xpath("""//div[contains(@class, 'raga-row')]//div[2]"""), # This gets all text but includes noise
            'beat': extract_with_xpath("""//div[contains(@class, 'beat-row')]//div[2]"""),
            'level': extract_with_xpath("""//div[contains(@class, 'level-row')]//div[2]"""),
            'tempo': extract_with_xpath("""//div[contains(@class, 'tempo-row')]//div[2]""")
        }

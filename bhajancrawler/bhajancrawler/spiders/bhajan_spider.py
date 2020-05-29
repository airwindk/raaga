# -*- coding: utf-8 -*-
import scrapy


class BhajanSpiderSpider(scrapy.Spider):
    name = 'bhajan_spider'
    allowed_domains = ['sairhythms.org']
    start_urls = ['http://sairhythms.org/']

    def parse(self, response):
        pass

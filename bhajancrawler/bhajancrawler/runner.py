from bhajancrawler.bhajancrawler.spiders.bhajan_spider import BhajanSpider
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings
import src.config as config


class Runner:

    def __init__(self, **kwargs):
        options = {
            'feed_uri': config.CRAWLER_SETTINGS['feed_uri'],
            'feed_format': config.CRAWLER_SETTINGS['feed_format'],
        }
        options.update(kwargs)
        self.settings = Settings({
            'FEED_URI': options['feed_uri'],
            'FEED_FORMAT': options['feed_format']
        })
        self.spiders = BhajanSpider
        self.process = CrawlerProcess(self.settings)

    def run(self):
        self.process.crawl(self.spiders)
        self.process.start()


if __name__ == "__main__":
    runner = Runner()
    runner.run()

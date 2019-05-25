# Scrape Wikipedia, saving page html code to wikipages directory 
# Most Wikipedia pages have lots of text 
# We scrape the text data creating a JSON lines file items.jl
# with each line of JSON representing a Wikipedia page/document
# Subsequent text parsing of these JSON data will be needed
# This example is for the topic robotics
# Replace the urls list with appropriate Wikipedia URLs
# for your topic(s) of interest

# ensure that NLTK has been installed along with the stopwords corpora
# pip install nltk
# python -m nltk.downloader stopwords

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import os.path
from lyrics.items import LyricsItem  # item class 


#%%
class LyricsScraperSpider2(CrawlSpider):
    name = 'lyrics_scraper2'
    allowed_domains = ['azlyrics.com']
    start_urls = ["https://www.azlyrics.com/r/realfriends.html",]
    
    rules = (
        Rule(LinkExtractor(allow=('', )), callback='parse_item'),
        )
    

    def parse_item(self, response):
        page = response.url.split("/")[4]
        page_dirname = 'songpages'
        filename = '%s.html' % page
        with open(os.path.join(page_dirname,filename), 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename) 

        # second part: extract text for the item for document corpus
        item = LyricsItem()
        item['url'] = response.url
        item['title'] = response.css('h1::text').extract_first()
        item['text'] = response.xpath('//div/text()').extract()  
        return item 

    
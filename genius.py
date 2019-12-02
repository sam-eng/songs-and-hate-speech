'''
genius.py utilizes a billboard chart downloader (install with pip install billboard.py) to scrape the past [x] years of top r
ap/hip-hop songs as ranked by Billboard, then uses Genius' API to build a dataset of lyrics from the songs.
'''

from bs4 import BeautifulSoup
from requests import get
import billboard
import time
import sys
from requests.exceptions import HTTPError
'''
Current issues: having trouble creating data set due to rate limits on server. Not quite sure what the limits are, so it's challenging.
TODO: pick a final date for ending web scraping
TODO: pick chart for rap songs, chart for hip-hop songs
TODO: send data to a csv? file, think of how to store it
TODO: use Genius API to pull lyrics of songs
'''
r_songs = set()
'''
r_chart = billboard.ChartData('hot-rap-tracks')
count = 0
while r_chart.previousDate:
    for song in r_chart:
        r_songs.add(song.title)
    try:
        r_chart = billboard.ChartData('hot-rap-tracks', r_chart.previousDate)
        count = count + 1
        time.sleep(2)
        print(r_chart.previousDate)
        # have a date limit
        if (r_chart.previousDate < "2019-01-01"):
            break
    except HTTPError as err:
        print("Error reached, waiting 2 minutes to try again...")
        time.sleep(120)
        print("current song list size: " + str(len(r_songs)))
'''
r_songs.add("Old Town Road")
r_songs.add("Thotiana")
r_songs.add("Sunflower (Spider-Man: Into The Spider-Verse)")

print(r_songs)
'''
genius.py utilizes a billboard chart downloader (install with pip install billboard.py) to scrape the past 29 years of top rap/hip-hop
songs as ranked by Billboard, then uses Genius' API to build a dataset of lyrics from the songs.
'''

from bs4 import BeautifulSoup
from requests import get
import billboard
import time
import sys
from requests.exceptions import HTTPError
import lyricsgenius
'''
Current issues: having trouble creating data set due to rate limits on server. Not quite sure what the limits are, so it's challenging.
TODO: pick chart for rap songs, chart for hip-hop songs
TODO: send data to a csv? file, think of how to store it

Potential issues: lyricsgenius may not recognzie the form of the song titles from Genius
'''
r_songs = {}
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
        if (r_chart.previousDate < "1990-01-01"):
            break
    except HTTPError as err:
        print("Error reached, waiting 2 minutes to try again...")
        time.sleep(120)
        print("current song list size: " + str(len(r_songs)))
'''
r_songs["Old Town Road"] = "Lil Nas X"
r_songs["Thotiana"] = "Blueface"
r_songs["Sunflower"] = "Post Malone"
genius = lyricsgenius.Genius("xNYwq4ZDn6Ytqu7pJ7Jvy-84TIuWLgwQTDQZ7cMe1ykdDIY7OwSCnblRToxw5N1y")
r_lyrics = {}
for song in r_songs:
    print(r_songs[song])
    g_song = genius.search_song(song, r_songs[song])
    r_lyrics[song] = g_song.lyrics
print(r_lyrics)
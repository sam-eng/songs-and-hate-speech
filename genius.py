'''
genius.py utilizes BeautifulSoup to scrape the past [x] years of top rap/hip-hop songs as ranked by Billboard,
then uses Genius' API to build a dataset of lyrics from the songs.
'''

from bs4 import BeautifulSoup
from requests import get

url = 'https://www.billboard.com/charts/rap-song/2005-01-02'
response = get(url)

soup = BeautifulSoup(response.text, 'html.parser')
data = soup.find_all(class_="chart-list-item__title-text")
song_titles = set()
for name in data:
    song_titles.add(name.text.strip())
print(song_titles)
# https://www.billboard.com/charts/r-b-hip-hop-songs
#https://www.billboard.com/charts/rap-song/2005-01-01
# chart-list-item__title-text
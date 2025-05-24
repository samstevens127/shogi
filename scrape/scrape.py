from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager


from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

from get_match import scrape
from tqdm import tqdm

BASE_URL = 'https://shogidb2.com'
PLAYERS_LIST_URL = f'{BASE_URL}/players'
DOWNLOAD_DIR = '../kif/'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get_player_links(): # get alphabetic order of names
    response = requests.get(PLAYERS_LIST_URL, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to access {PLAYERS_LIST_URL}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    # Adjust the selector based on the actual HTML structure
    player_links = [urljoin(BASE_URL, a['href']) for a in soup.select('a[href^="/players/"]')]
    del player_links[-2]
    return player_links

def get_kif_page(player_link): #gather list of players
    response = requests.get(player_link, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to access {PLAYERS_LIST_URL}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    # Adjust the selector based on the actual HTML structure
    player_links = [urljoin(BASE_URL, a['href']) for a in soup.select('a[href^="/player/"]')]
    return player_links

def loop_kif_pages(link): # gather list of games by player given the player page
    i = 1
    link = link + f'?q=&page={i}'
    response = requests.get(link, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to access {link}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    games = []

    while (len(soup.select('a[href^="/games/"]')) != 0 and i < 51):
        link = link + f'?q=&page={i}'
        response = requests.get(link, headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to access {link}")

        soup = BeautifulSoup(response.text, 'html.parser')
        games= games + [urljoin(BASE_URL, a['href']) for a in soup.select('a[href^="/games/"]')]
        i += 1
    return games

def get_kif_files(link):
    kif = scrape(link)
    if kif == None:
        return
    file_name = re.split("/+", link)[-1]
    with open (DOWNLOAD_DIR + file_name+ ".kif", "w") as f:
        f.write(kif)
    


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    player_links = get_player_links() # alphabetic order of names
    players = [] # list of players
    game_links = [] #list of games
    with ThreadPoolExecutor(max_workers=len(player_links)) as executor:
        alphabeticals = [executor.submit(get_kif_page, url) for url in player_links]
        for alphabetical in as_completed(alphabeticals):
            result = alphabetical.result()
            players.extend(result)
        print(f"{len(players)} players gathered")
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        names = [executor.submit(loop_kif_pages, url) for url in players]
        for name in tqdm(as_completed(names), total=len(players), desc="Scraping"):
            result = name.result()
            game_links.extend(result)

    game_links = list(set(game_links))
    print(f"{len(game_links)} games gathered")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        games = [executor.submit(get_kif_files, url) for url in game_links]
        for game in tqdm(as_completed(games), total=len(game_links), desc="Scraping KIF"):
            pass

if __name__ == '__main__':
    main()

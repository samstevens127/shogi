from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
import tempfile, os
import requests
import time

# Setup
def scrape(url):
    # 1. Download uBlock Origin XPI
    ublock_path = './ublock.xpi'

    # 3. Use FirefoxOptions instead of FirefoxProfile
    options = Options()
    options.headless = False

    # 4. Set the custom profile directory
    driver = webdriver.Firefox(options=options)#service=Service(GeckoDriverManager().install())
    driver.install_addon(ublock_path, temporary=True)
    wait = WebDriverWait(driver, 900)
    driver.get(url)

    try:
        # 1. Find and click the "KIF形式" link (Phoenix LiveView)
        #kif_button = 
        button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[phx-click="kif"]')))
        time.sleep(4)
        driver.execute_script("arguments[0].scrollIntoView(true);", button)
        #button.click()
        #(By.XPATH, "//a[contains(text(), 'KIF形式')]")
        #button = driver.find_element(By.CSS_SELECTOR, '[phx-click="kif"]')
        driver.execute_script("arguments[0].click();", button)

        # 2. Wait for the textarea to show up
        textarea = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "dialog#kifu-modal textarea"))
        )

        # 3. Wait until the textarea has non-empty content
        wait.until(lambda d: textarea.get_attribute("value").strip() != "")

        kif = textarea.get_attribute("value")
        return kif.strip()

    except Exception as e:
        import traceback
        print("❌ Error occurred:")
        traceback.print_exc()
        return None

    finally:
        driver.quit()

if __name__ == "__main__":
    scrape("https://shogidb2.com/games/1b31e5c577acfa3a166f55a706d8945197d8b9bf")

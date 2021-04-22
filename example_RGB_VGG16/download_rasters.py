import wget
import json
import zipfile
import requests, os
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from tqdm import tqdm

#Directory to store downloaded rasters
script_folder = os.path.dirname(__file__)
save_folder = os.path.join(script_folder, "RGB - Brasil - Niteroi")

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

def gen_url():

    url_database = "http://sigeo.niteroi.rj.gov.br/ortofoto/"
    
    #extract html with metadata to download
    session = HTMLSession()
    r = session.get(url_database)
    html = r.html.xpath("""//*[@id="ortomap2019"]""")[0].html

    # detect image names to download
    soup = BeautifulSoup(html, features="html.parser")
    imgs_num = [elem['alt'] for elem in soup.findAll('area')]

    # generate download link
    url_base0 = "http://www.sigeo.niteroi.rj.gov.br/ortofotos/2019/TIFF/"
    url_base2 = "-TIFF.zip"
    urls_download = [url_base0 + img_num + url_base2 for img_num in imgs_num]

    return urls_download

def download(urls_download, save_folder):
    """Download, extract and deleted from url.

    Args:
        urls_download [list]: list of download urls
        save_folder [string]: path to save rasters
    """


    for url in tqdm(urls_download):
        try:
            #define name to save file same as download name
            file_name = url.split('/')[-1]
            file_name = os.path.join(save_folder,file_name)

            #download by url
            wget.download(url, out=file_name)
            
            #unzip downloaded file
            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall(save_folder)

            #delete zip file downloaded because its already extracted
            os.remove(file_name)

        except:
            print('fail to download file', file_name, ' from url \n ', url)

    print('Downloads finished.')

urls_download = gen_url()

download(urls_download, save_folder)
######################################################################################################################################################
#
#   name:       HTTP_Utils.py
#   contains:   Functions for accessing data stored at generic HTTP links (usually by downloading)
#   created by: Mitchell Bonney
#
######################################################################################################################################################

# Built in
import os
import gzip
import shutil

from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Open source
import requests

from tqdm.notebook import tqdm

######################################################################################################################################################
# TODO ###############################################################################################################################################
######################################################################################################################################################

# Add numbering condition to downloadAll (where you specify where in file name numbers are and what numbers you want to download)
# Check if file already dwnloaded with downloadAll (currently overwrites)

######################################################################################################################################################
# Download ###########################################################################################################################################
######################################################################################################################################################

# Download all files from link that meet specified conditions.
def downloadAll(url, folder, file_ext):

    """
    Parameters:
    url (str): URL to website with files (https://website/with/files/)

    folder (str): Folder where you want to save the downloaded files ('C:/path/to/folder').

    file_ext (str): Extension of files to download (e.g., 'gz', 'pdf').

    Returns:
    Downloaded files in specified folder. 
    """

    # Set up
    response = requests.get(url) # This may lead to SSLCertVertificationError if NRCan certification not applied to cacert.pem
    soup = BeautifulSoup(response.text, 'html.parser')

    for link in tqdm(soup.select("a[href$='." + file_ext + "']")):
        filename = os.path.join(folder, link['href'].split('/')[-1])
        with open(filename, 'wb') as file:
            file.write(requests.get(urljoin(url, link['href'])).content)

######################################################################################################################################################
# Organize ###########################################################################################################################################
######################################################################################################################################################

# Replace all zipped files in folder of a certain extension with unzipped files.
def unzipAll(folder, file_ext):

    """
    Parameters:
    folder (str): Folder where you want to replace zipped files ('C:/path/to/folder').

    file_ext (str): Extension of files to unzip (e.g., 'gz').

    Returns:
    Unzipped files in specified folder.
    """

    # Folder to work in
    os.chdir(folder)

    # Loop through all items in folder
    for item in tqdm(os.listdir(folder)):
        if item.endswith(file_ext): # Check for specified extension 
            path = os.path.abspath(item) # Get full path of files
            name = (os.path.basename(path)).rsplit('.',1)[0] # Get file name for file within
            with gzip.open(path, 'rb') as f_in, open(name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out) # Unzip and copy out
            os.remove(path) # Delete zipped file   

######################################################################################################################################################
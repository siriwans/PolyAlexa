import requests
from bs4 import BeautifulSoup
import json, re

json_obj = {}
json_obj['Sections'] = []

def clean(txt):
    return re.sub(r'\[.*\]', '', txt)

try:
   response = requests.get(
	url="https://en.wikipedia.org/wiki/California_Polytechnic_State_University",
   )
except requests.ConnectionError:
    exit()

print(response.status_code)
soup = BeautifulSoup(response.content, 'html.parser')

big_headers = soup.find_all('h2')
small_headers = soup.find_all('h3')
smallest_headers = soup.find_all('h4')
paragraphs = soup.find_all('p')
data = soup.find_all(['h2', 'h3' ,'h4','p'])

current_section = {"Title":"", "Information":"", "Subsections": []}
subsection = {"Title":"","Information": "", "Subsections": []}
sub_subsection = {"Title":"","Information": ""}
has_subsection = False
has_sub_subsection = False
for line in data:
    text = clean(line.text)

    if line in big_headers: 
        json_obj['Sections'].append(current_section)
        current_section = {"Title": text, "Information":"", "Subsections": []}
        has_subsection = False
    if line in small_headers: 
        has_subsection = True
        subsection = {"Title": text, "Information": "", "Subsections": []}
        current_section["Subsections"].append(subsection)
    if line in smallest_headers: 
        has_sub_subsection = True
        sub_subsection = {"Title": text, "Information": ""}
        subsection["Subsections"].append(sub_subsection)
    if line in paragraphs: 
        if has_subsection == False and has_sub_subsection == False:
            current_section["Information"] = current_section["Information"] + text
        elif has_subsection == True and has_sub_subsection==False:
            subsection["Information"] = subsection["Information"] + text
        else: 
            sub_subsection["Information"] = sub_subsection["Information"] + text
            has_sub_subsection = False


sidebox_keys = [clean(x.text) for x in soup.table.find_all('th')]
sidebox_vals = [clean(x.text) for x in soup.table.find_all('td')[1:-1]]
sidebox = dict()
for (key, val) in zip(sidebox_keys, sidebox_vals):
    sidebox[key] = val
json_obj['sidebox'] = sidebox

with open('wikipedia.json','w') as jsonFile:
    json.dump(json_obj, jsonFile, indent='\t')

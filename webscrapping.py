import requests
from bs4 import BeautifulSoup
import json

json_obj = {}
json_obj['Sections'] = []


response = requests.get(
	url="https://en.wikipedia.org/wiki/California_Polytechnic_State_University",
)
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
    if line in big_headers: 
        json_obj['Sections'].append(current_section)
        current_section = {"Title":line.text, "Information":"", "Subsections": []}
        has_subsection = False
    if line in small_headers: 
        has_subsection = True
        subsection = {"Title":line.text,"Information": "", "Subsections": []}
    if line in smallest_headers: 
        has_sub_subsection = True
        sub_subsection = {"Title":line.text,"Information": ""}
    if line in paragraphs: 
        if has_subsection == False and has_sub_subsection == False: 
            current_section["Information"] = current_section["Information"]+ line.text
        elif has_subsection == True and has_sub_subsection==False:
            subsection["Information"] = subsection["Information"]+ (line.text)
            current_section["Subsections"].append(subsection)
        else: 
            sub_subsection["Information"] = sub_subsection["Information"]+(line.text)
            subsection["Subsections"].append(sub_subsection)
            current_section["Subsections"].append(subsection)
            has_sub_subsection = False


with open('wikipedia.json','w') as jsonFile:
    json.dump(json_obj, jsonFile)
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import datetime

page1 = requests.get('https://www.bnm.gov.my/opr-decision-and-statement')
soup = BeautifulSoup(page1.content, 'html.parser')
opr_table = soup.find("table", attrs={"class": "standard-table table-bordered table-hover text-center table-striped"})
opr_table_data = opr_table.find_all("tr")  # contains 2 rows

# Get all the headings of Lists
headings = []
for td in opr_table_data[1].find_all("td"):
    # remove any newlines and extra spaces from left and right
    headings.append(td.text.replace('\n', ' ').strip())

data = []
for i in range(2, len(opr_table_data)):
    temp_dict = {}
    tmp = opr_table_data[i].find_all("td")
    temp_dict['Date'] = tmp[0].text.replace('\n', '').strip()
    temp_dict['Change in OPR (%)'] = tmp[1].text.replace('\n', '').strip()
    temp_dict['New OPR Level (%)'] = tmp[2].text.replace('\n', '').strip()
    temp_link = 'https://www.bnm.gov.my' + \
        tmp[3].find_all('a', href=True)[0]['href']
    page_dyn = requests.get(temp_link)
    soup2 = BeautifulSoup(page_dyn.content, 'html.parser')
    raw_content = soup2.find_all(
        "div", attrs={"class": "article-content-cs"})[1]
    paragraph = raw_content.find_all('p')
    temp_dict['Monetary Policy Statement'] = ' '.join([para.get_text(strip=True) for para in paragraph])
    data.append(temp_dict)

final_data = pd.DataFrame(data)


def year_month(x):
    date_time_obj = datetime.datetime.strptime(str(x['Date']), '%d %b %Y')
    if date_time_obj.month < 4:
        q = "Q1"
    elif date_time_obj.month < 7:
        q = "Q2"
    elif date_time_obj.month < 10:
        q = "Q3"
    else:
        q = "Q4"
    x['Year'] = date_time_obj.year
    x['Quarter'] = q + "-" + str(date_time_obj.year)
    return x

final_data2 = final_data.apply(lambda x: year_month(x), axis=1)
final_data2.to_csv('bank_negara_mps.csv', index=False)


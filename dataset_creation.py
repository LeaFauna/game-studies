import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

def to_csv(dataframe, filename):
    path = r"C:/Users/lea/Desktop/DPHIL/GameStudies"
    full_path = f"{path}//{filename}"
    dataframe.to_csv(full_path, index=False)

def removables(soup):
    h_ref = soup.find(["h2", "h3", "h4", "b"], string = re.compile('([rR]eferences?)|([bB]ibliography)|([lL]iterature)')) # find first element that fulfills one of the conditions
    if h_ref:
        for e in h_ref.find_all_next():
            e.decompose()
    else:
        print("no references found")
    return soup

def extract_content(html_in_df):
    try:
        div_article = html_in_df.find('div', id='article')
        table_tag = html_in_df.find('table')
        if div_article:
            div_article = removables(div_article)
            print(type(div_article))
            return div_article.get_text()
        elif table_tag:
            table_tag = removables(table_tag)
            table_list = []
            for row in table_tag.find_all('tr'):
                row_data = [cell.get_text(strip=True) for cell in row.find_all(['th','td'])]
                table_list.append(row_data)
            table_df = pd.DataFrame(table_list)
            cell_value = table_df.iloc[5,2]
            return cell_value
        else:
            return "Not found" 
    except Exception as e:
        return f"Error: {e}"

# get html
url_archive = "https://gamestudies.org/2302/archive"
result = requests.get(url_archive)
content = result.text
soup = BeautifulSoup(content, 'html.parser')
pretty_soup = soup.prettify()
type(pretty_soup)
# get area of interest in "archive"
box = soup.find('dl', id='authorlist')

# save all links to websites from "archive" to list
urls = [url['href'] for url in box.find_all('a', href=True)]
print(urls)

# get issue from urls using regex
pattern = r'https://[^/]+/(\d+)/'
issues = []
for url in urls:
    match = re.search(pattern, url)
    if match:
        issues.append(match.group(1))
    else:
        issues.append("invalid url")

# add urls and issues to dataframe
df = pd.DataFrame({'url': urls, 'issue': issues})

# add dates to dataframe
date_mapping = {
    101: '07/2001', 102: '07/2002', 202: '12/2002', 301: '05/2003',
    302: '12/2003', 401: '07/2004', 501: '10/2005', 601: '12/2006',
    701: '08/2007', 801: '09/2008', 802: '12/2008', 901: '04/2009',
    902: '11/2009', 1001: '04/2010', 1101: '02/2011', 1102: '05/2011',
    1103: '12/2011', 1201: '09/2012', 1202: '12/2012', 1301: '09/2013',
    1302: '12/2013', 1401: '08/2014', 1402: '12/2014', 1501: '07/2015',
    1502: '12/2015', 1601: '10/2016', 1602: '12/2016', 1701: '07/2017',
    1702: '12/2017', 1801: '04/2018', 1802: '09/2018', 1803: '12/2018',
    1901: '05/2019', 1902: '10/2019', 1903: '12/2019', 2001: '02/2020',
    2002: '06/2020', 2003: '09/2020', 2004: '12/2020', 2101: '05/2021',
    2102: '07/2021', 2103: '09/2021', 2104: '12/2021', 2201: '03/2022',
    2202: '04/2022', 2203: '08/2022', 2301: '03/2023', 2302: '07/2023'
}

date_mapping = {key: pd.to_datetime(value) for key, value in date_mapping.items()}

for issue, date in date_mapping.items():
    mask=df['issue'] == issue
    df.loc[mask, 'date'] = date

df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')

# get html for all urls
df['html'] = None
for index, row in df.iterrows():
    result = requests.get(row["url"])
    html = result.text
    soup = BeautifulSoup(html,'html.parser')
    df.at[index, 'html'] = soup

# get relevant content of htmls
df['content'] = None
df['content'] = df['html'].apply(extract_content)

# save dataset to csv and xlsx
to_csv(df, "dataset.csv")
df.to_excel("dataset.xlsx")
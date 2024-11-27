import pandas as pd
#import os
from uuid import uuid4
from transformers import AutoTokenizer
from math import ceil
import requests
#from bs4 import BeautifulSoup
import re
#from collections import defaultdict
from tqdm import tqdm
#from datetime import datetime
import random
from bs4 import BeautifulSoup
from itertools import zip_longest


data_folder = '/raid/deallab/SF_RAG_Data/ASQA'
data_folder = '../data'

tokenizer = AutoTokenizer.from_pretrained(
   "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp" ## adjust tokenization model to the one that is used in the embedding/retriever achitecture # "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
)

token_length = 1024 # adjust to maximal token length
document_limit = 10000
dataset = 'dev'

# restricting legth (make room for cls token and paragraph seperators) 
TOK_LEN = token_length - 24

#helper
def get_wikipage_title(url):
    return url.split('/')[-1]

def split_list(a, max_len, o):
    o_len = len(a) + ceil(len(a)/max_len) * o 
    n = ceil(o_len/ max_len)
    k, m = divmod(o_len, n)
    return (a[i*(k-o)+min(i, m):(i+1)*k -i*o+min(i+1, m)] for i in range(n))

# api calls 
def get_ref_ids(session, title):
    url = 'https://en.wikipedia.org/w/api.php?'
    headers = {'User-Agent': 'sf_rag/1.0; lassejantsch@knu.ac.kr)'}
    params = {
        'action': 'query',
        'prop': 'revisions',
        'titles': title,
        'rvstart': '2020-02-01T00:00:00Z',
        'rvlimit': '1',
        'rvdir': 'older',
        'rvprop': 'ids|timestamp',
        'rvslots': 'main',
        'formatversion':'2',
        'format': 'json',
        'redirects': None,
    }
    response = None

    # get document text
    try:
        res =session.get(url + '&'.join([f'{k}={v}' if v != None else f'{k}'  for k,v in params.items()]), headers= headers)
        #parse docuements
        res_json = res.json()
        response = res_json['query']['pages'][0]['revisions'][0]['revid']
    except:
        print(res.status_code, res.url)
    return response

def get_document(session, ref_id):
    url = 'https://en.wikipedia.org/w/api.php?'
    headers = {'User-Agent': 'sf_rag/1.0; lassejantsch@knu.ac.kr)'}
    params = {
        'action': 'parse',
        'prop': 'text',
        'oldid': ref_id,
        'formatversion':'2',
        'format': 'json'
    }
    response = None

    # get document text
    try:
        res =session.get(url + '&'.join([f'{k}={v}' for k,v in params.items()]), headers= headers)
        #parse docuements
        res_json = res.json()
        response = res_json['parse']['text']
    except:
        print(res.status_code, res.url)
    return response



class DocElement():
    def __init__(self, text, tokens, level):
        self.text = text
        self.level = int(level)
        self.tokens = tokens
        self.length = len(tokens)
        
    def __len__(self):
        return self.length
    
    def __str__(self):
        return "\t"* (self.level+1) + self.text

    def get_text(self):
        return self.text
    
    def split_doc(self, max_tokens, title):
        raise NotImplementedError

class DocSection(DocElement):
    def __init__(self, title, tokens, level):
        super().__init__(title, tokens, level)
        self.title = self.text
        self.content = []
        self.has_subsection = False
        
    
    def __str__(self):
        print_str = ''.join([str(el) for el in self.content])
        return '{0}{1}'.format("\t"*self.level + self.title,print_str)
        
        
    def update_length(self):
        len_of_content = sum([len(el) for el in self.content])
        self.length = len(self.tokens) + len_of_content
        # print(self.title,len(self.tokens), len_of_content, self.length)
    
    def get_text(self):
        text = self.title
        for element in self.content:
            text += element.get_text()
        return text
    
    def append(self, content, tokens, level, type):
        if type == 'sec':
            if level-1 == self.level:
                self.content.append(DocSection(content, tokens, level))
                self.has_subsection=True
            else:
                self.content[-1].append(content, tokens, level, type)                
        elif type=='par':
            if self.has_subsection:
                self.content[-1].append(content, tokens, level, type)
            else:
                self.content.append(DocElement(content, tokens, level))
        self.update_length()
    
    def split_doc(self, max_tokens, title=''):
        if title == '':
            title = re.search(r'#+ (.*?) #+', self.title).group(1)
        else:
            title = '{0}/{1}'.format(title, re.search(r'#+ (.*?) #+', self.title).group(1))
        title_str = 'Document: {0}\n\n'.format(title)
        title_str_len = len(tokenizer.encode(title_str, add_special_tokens=False))

        # if whole doc fits into max tokens
        if self.length + title_str_len <= max_tokens:
            return [self.get_text()]
        
        # split if not
        splitted_doc_ls = []
        current_split= title_str
        current_split_len = title_str_len
        for element in self.content:
            if len(element) == 0: continue # continue if empty element
            if current_split_len + len(element) <= max_tokens:
                #print(self.level,current_split_len, len(element), 'added to current')
                current_split += element.get_text()
                current_split_len += len(element)
            elif len(element)+title_str_len <= max_tokens:
                #print(self.level, current_split_len, len(element), 'added to new')
                splitted_doc_ls.append(current_split)
                current_split = title_str + element.get_text()
                current_split_len = title_str_len + len(element)
            else:
                #print(self.level,current_split_len, len(element), 'recursion')
                if current_split_len > title_str_len: 
                    splitted_doc_ls.append(current_split)
                    current_split = title_str
                    current_split_len = title_str_len
                splitted_doc_ls.extend(element.split_doc(max_tokens, title))
        if current_split_len > title_str_len: splitted_doc_ls.append(current_split)
        return splitted_doc_ls
                
                 

class Document(DocSection):
    def __init__(self, title, tokenizer, max_len = 1024):
        self.tokenizer = tokenizer
        self.last_level = 0
        self.max_len = max_len
        super().__init__(title, self.tokenizer.encode(title, add_special_tokens=False), 0)
    
    def add(self, content, level = None):
        if not level:
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            if len(tokens) < self.max_len - 100:
                self.append(content, tokens, self.last_level, 'par')
            else:
                token_ls = split_list(tokens, self.max_len -100, 200)
                for split in token_ls:
                    content = self.tokenizer.decode(split)
                    self.append(content, split, self.last_level, 'par')
        else:
            self.append(content, self.tokenizer.encode(content, add_special_tokens=False), level, 'sec')
            self.last_level = level
        

def parse_table(table):
    rows = table.find_all('tr')
    if not rows: return None
    parsed_rows = []
    spans = {}  # Tracks cells with rowspan/colspan

    if rows[0].find('th'):
        header_row = rows.pop(0)
        columns = [th.get_text().strip() for th in header_row.find_all('th')]
    else: columns = None

    for row_idx, row in enumerate(rows):
        parsed_row = []
        cells = row.find_all('td')

        col_idx = 0  # Tracks current column index
        for cell in cells:
            # Check for existing spans
            while col_idx in spans and spans[col_idx]['rows'] > 0:
                parsed_row.append(spans[col_idx]['text'])
                spans[col_idx]['rows'] -= 1
                if spans[col_idx]['rows'] == 0:
                    del spans[col_idx]
                col_idx += 1

            # Add cell content
            text = cell.get_text(strip=True)
            try:
                colspan = int(cell.get('colspan', 1)) 
                rowspan = int(cell.get('rowspan', 1))
            except:
                colspan, rowspan = 1, 1
            # Fill current cell(s)
            for _ in range(colspan):
                parsed_row.append(text)

            # Handle row/colspan spans
            if rowspan > 1:
                for span_col in range(colspan):
                    spans[col_idx + span_col] = {'text': text, 'rows': rowspan - 1}

            col_idx += colspan

        # Handle trailing spans
        while col_idx in spans and spans[col_idx]['rows'] > 0:
            parsed_row.append(spans[col_idx]['text'])
            spans[col_idx]['rows'] -= 1
            if spans[col_idx]['rows'] == 0:
                del spans[col_idx]
            col_idx += 1

        parsed_rows.append(parsed_row)

        if columns:
            parsed_table = '\n----\n'.join(['\n'.join([f'{columns[i] if len(columns) > i else None}:{value}'for i, value in enumerate(row)]) for row in parsed_rows])
        else: parsed_table = '\n----\n'.join(['\n'.join(row) for row in parsed_rows])

    return parsed_table

from bs4 import BeautifulSoup

def parse_infobox(info_box):

    # Parse title and description
    title = info_box.select_one('th.infobox-above').get_text(strip=True) if info_box.select_one('th.infobox-above') else "N/A"
    description = info_box.select_one('tr.description th').get_text(strip=True) if info_box.select_one('tr.description th') else "N/A"

    # Extract field data
    fields = []
    for row in info_box.select('tr'):
        header = row.find('th', class_='infobox-label')
        data = row.find('td', class_='infobox-data')
        if header and data:
            label = header.get_text(strip=True)
            value = data.get_text(strip=True)
            fields.append(f"{label}: {value}")

    # Parse singles chronology
    chronology = []
    chronology_table = info_box.select_one('tr > td.infobox-full-data table')
    if chronology_table:
        for cell in chronology_table.select('td'):
            link = cell.find('a')
            text = cell.get_text(strip=True)
            title = link.get_text(strip=True) if link else text
            year = text.split('(')[-1].rstrip(')')
            chronology.append(f"{title} ({year})")

    # Parse external links
    external_links = []
    for link in info_box.select('td.infobox-full-data a.external.text'):
        label = link.get_text(strip=True)
        url = link['href']
        external_links.append(f"{label}: {url}")

    # Build the structured string
    output = []
    output.append(f"Title: {title}")
    output.append(f"Description: {description}")
    output.append("\nFields:")
    output.extend(fields)
    output.append("\nSingles Chronology:")
    output.extend(chronology)
    output.append("\nExternal Links:")
    output.extend(external_links)

    # Join the output into a single structured string
    return "\n".join(output)


def parse_document(title, doc_str):
    soup = BeautifulSoup(doc_str, "html.parser")
    doc_ls = soup.div.find_all(recursive=False)    
    parsed_doc = Document(f'# {title} #\n', tokenizer)

    for el in doc_ls:
        if el.name == 'div' and el.has_attr('class') and 'mw-heading' in el['class']:
            heading = el.find(re.compile('^h[1-6]$'))
            level = int(heading.name[1])
            text = f"{'#' * level} {heading.get_text().strip()} {'#' * level}\n"
            if text in ['See also', 'References']: break # break condition when main article is over
            if text: parsed_doc.add(text, level - 1)
        if el.name == 'p':
            text = el.get_text()
            if text: parsed_doc.add(text + '\n')
        elif el.name == 'ul':
            text = ''
            for sub_el in el.find_all('li'):
                text += '* ' + sub_el.get_text() + '\n'
            if text: parsed_doc.add(text)
        elif el.name == 'ol':
            text = ''
            for i, sub_el in enumerate(el.find_all('li')):
                text += f'{i+1}. ' + sub_el.get_text() + '\n'
            if text: parsed_doc.add(text)
        elif el.name == 'dl':
            text = ''
            for i, sub_el in enumerate(el.find_all(['dt', 'dd'])):
                if sub_el.name == 'dt':
                    text += f'{sub_el.get_text()}:\n'
                else:
                    text += f'* {sub_el.get_text()}\n'
            if text: parsed_doc.add(text)
        elif el.name == 'table' and el.has_attr('class') and 'metadata' not in el['class']:
            if 'infobox' in el['class']:
                text = parse_infobox(el)
            else: text = parse_table(el)
            if text: parsed_doc.add(text + '\n')
    
    return parsed_doc.split_doc(1024)
    
def main():
    # read  data
    df = pd.read_parquet(f'{data_folder}/dev.parquet')

    for col in df.columns:
        print(col,':')
        print(df.loc[0, col], '\n')


    # create embedding document dataset.
    evidence_test = pd.DataFrame(columns=['text'])
    evidence_test_path = f'{data_folder}/test/evidence_test.csv'
    evidence_test.to_csv(evidence_test_path, index=False)

    qa_test = pd.DataFrame(columns=['id', 'sample_id', 'question', 'follow_up_questions', 'long_answers', 'short_answers'])
    qa_test_path = f'{data_folder}/test/qa_test.csv'
    qa_test.to_csv(qa_test_path, index=False)

    # init params
    session = requests.Session() # initiate session
    added_evidence = set() # empty evidence set

    document_limit = 1000

    for idx, row in tqdm(df.iterrows(), total=min(document_limit, len(df))):
        try:
            #break when document limit is reached 
            if idx == document_limit: break
            
            #extract information from sample
            sample_id = row['sample_id']
            evidence_title_mapping = {evidence['title']:get_wikipage_title(evidence['url']) for evidence in row['wikipages']}
            base_question = row['ambiguous_question']
            follow_up_questions = [qa['question'] for qa in row['qa_pairs']]
            short_answers = [qa['short_answers'].tolist() for qa in row['qa_pairs']]
            long_answers = [answ['long_answer'] for answ in row['annotations']]
            
            # crawl documents
            evidence_docs = {}
            for title, url_title in evidence_title_mapping.items():
                    ref_id = get_ref_ids(session, url_title)
                    text = get_document(session, ref_id)
                    evidence_docs[title] = parse_document(title, text)
                
            # fill test evidence
            for title, docs in evidence_docs.items():
                    if title in added_evidence: continue
                    added_evidence.add(title)
                    for doc in docs:
                            evidence_test.loc[len(evidence_test)] = [doc]
            
            # fill qa test
            qa_test.loc[len(qa_test)] = [uuid4(), sample_id, base_question, follow_up_questions, long_answers, short_answers]
        except:
            print('something went wrong')
            
        if (idx + 1 )% 10 == 0:
            evidence_test.to_csv(evidence_test_path,mode='a', header=False, index=False)
            qa_test.to_csv(qa_test_path,mode='a', header=False, index=False)
            evidence_test = pd.DataFrame(columns=['text'])
            qa_test = pd.DataFrame(columns=['id', 'sample_id', 'question', 'follow_up_questions', 'long_answers', 'short_answers'])
            

    evidence_test.to_csv(evidence_test_path,mode='a', header=False, index=False)
    qa_test.to_csv(qa_test_path,mode='a', header=False, index=False)

if __name__ == "__main__":
    main()

'''
nohup python3 -u asqa_prep.py > out.log
'''
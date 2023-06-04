import json
import argparse

from collections import defaultdict
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


def get_lang_detector(nlp, name):
    return LanguageDetector()

def read_file(file_path):
    results = []
    idx = 0
    type2idx = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            if line == "\n":
                continue
            curr_item = json.loads(line)
            curr_type = curr_item['meta']['pile_set_name']
            results.append(curr_item)
            type2idx[curr_type].append(idx)
            idx += 1
    return results, type2idx

def clean_up_github(data):
    cleaned = []
    for curr_item in data:
        if curr_item['meta']['pile_set_name'] == 'Github':
            continue
        else:
            cleaned.append(curr_item)
    return cleaned
            

def remove_non_english(data):
    nlp = spacy.load("en_core_web_sm")
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe("language_detector", last=True)
    cleaned = []
    for single_item in data:
        if single_item['meta']['pile_set_name'] == 'Github':
            continue
        doc = nlp(single_item['text'][:500])
        if doc._.language['language'] != 'en':
            print(doc._.language['language'])
            print(single_item['text'])
            print('-------------------')
        else:
            cleaned.append(single_item)
    return cleaned
    
            
def substitute_unicode(data):
    unicode2ascii = {'\u2019': "'", 
                     '\u2018': "'", 
                     '\u201c': '"', 
                     '\u201d': '"', 
                     '\u2014': '-', 
                     '\u2013': '-'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='../data')
    parser.add_argument("--data-file", type=str, default='00.jsonl')
    parser.add_argument("--clean-up-github", action='store_true')
    parser.add_argument("--output-file", type=str, default=None)
    
    args = parser.parse_args()
    
    data, type2idx = read_file(f'{args.data_dir}/{args.data_file}')
    
    if args.clean_up_github:
        data = clean_up_github(data)
        
    data = remove_non_english(data)
    if args.output_file is None:
        output_file = f'{args.data_dir}/{args.data_file[:-6]}_cleaned.jsonl'
    else:
        output_file = args.output_file
        
    with open(output_file, 'w+') as f:
        for single_item in data:
            f.write(json.dumps(single_item))
            f.write('\n')
    

if __name__ == "__main__":
    main()
    
    
    
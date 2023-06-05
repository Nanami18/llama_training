import json
import argparse

from collections import defaultdict
# import spacy
from multiprocessing import Pool
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing
# from spacy.language import Language
import os
# from spacy_langdetect import LanguageDetector
from fasttext import load_model

model = load_model("lid.176.ftz")

# def get_lang_detector(nlp, name):
#     return LanguageDetector()

def read_file(file_path):
    results = []
    type2idx = defaultdict(list)
    idx = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line == "\n":
                continue
            # if idx > 10000:
            #     break
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
    # nlp = spacy.load("en_core_web_sm")
    # Language.factory("language_detector", func=get_lang_detector)
    # nlp.add_pipe("language_detector", last=True)
    id = multiprocessing.current_process().pid
    with open(f'cleaned_{id}.jsonl', 'a+') as f:
        if type(data) == dict:
            data = [data]
        elif type(data) == list:
            data = data
        for single_item in data:
            # if single_item['meta']['pile_set_name'] == 'Github':
            #     continue
            # doc = nlp(single_item['text'][:200])
            text = single_item['text'][:200].replace('\n', '')
            label, _ = model.predict(text)
            lang = label[0].replace('__label__', '')
            if lang != 'en':
                print(lang)
                continue
            # if doc._.language['language'] != 'en':
            #     print(doc._.language['language'])
            else:
                f.write(json.dumps(single_item))
                f.write('\n')
    
            
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
        
    if args.output_file is None:
        output_file = f'{args.data_dir}/{args.data_file[:-6]}_cleaned.jsonl'
    else:
        output_file = args.output_file
        
    print("n_cores: ", multiprocessing.cpu_count())
    n_cores = multiprocessing.cpu_count()
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            cleaned = []
            for line in f:
                cleaned.append(json.loads(line))
    else:
        cleaned = []
        
    # data = data[:200]
    r = process_map(remove_non_english, data, max_workers=n_cores)
    

if __name__ == "__main__":
    main()
    
    
    
import json
import os
from pathlib import Path
import re

all_sentences = {}

def translate_gt(gtfile, dir):
    entities = []
    relationships = []
    sentences = []
    count = 1
    with open(gtfile, 'r') as f:
        worksheet = json.load(f)
    for i in worksheet:
        sentence = i['text']
        sentence = re.sub(r'[^a-zA-Z0-9,-_\']', r' ', sentence)
        sentence = re.sub(r'[\.\":;\n]', r' ', sentence)
        sentence = sentence.lower().strip()
        sentences.append(f'''sentence({count},"{sentence}").''')
        all_sentences[sentence] = count
        for entity in i['entities']:
            e = entity['entity']
            type = entity['type']
            e = re.sub(r'[^a-zA-Z0-9,-_\']', r' ', e)
            e = re.sub(r'[\.\":;\n]', r' ', e)
            type = re.sub(r'[^a-zA-Z0-9,-_\']', r' ', type)
            type = re.sub(r'[\.\":;\n]', r' ',type)
            asp = f'''entity({count}, "{e.lower().strip()}", "{type.lower().strip()}").'''
            entities.append(asp)
        for relationship in i['relationships']:
            obj = relationship['object']
            sbj = relationship['subject']
            type = relationship['type']
            obj = re.sub(r'[^a-zA-Z0-9,-_\']', r' ', obj)
            obj = re.sub(r'[\.\":;\n]', r' ', obj)
            sbj = re.sub(r'[^a-zA-Z0-9,-_\']', r' ', sbj)
            sbj = re.sub(r'[\.\":;\n]', r' ', sbj)
            type = re.sub(r'[^a-zA-Z0-9,-_\']', r' ', type)
            type = re.sub(r'[\.\":;\n]', r' ', type)
            asp = f'''relation({count}, "{sbj.lower().strip()}", "{obj.lower().strip()}", "{type.lower().strip()}").'''
            relationships.append(asp)
        count += 1

    with open(f'{dir}ground_truth_asp.txt', 'w') as f:
        f.write('\n'.join(sentences))
        f.write('\n')
        f.write('\n'.join(entities))
        f.write('\n')
        f.write('\n'.join(relationships))

def translate_llm(predfile, dir):
    entities = []
    relationships = []
    count = 0

    try:
        with open(predfile, 'r') as f:
            worksheet = json.load(f)
        for i in worksheet:
            count += 1
            for entity in i['entities']:
                e = entity['entity']
                type = entity['type']
                e = re.sub(r'[^a-zA-Z0-9,-_]', r' ', e)
                e = re.sub(r'[\.\":;\n]', r' ', e)
                type = re.sub(r'[^a-zA-Z0-9,-_]', r' ', type)
                type = re.sub(r'[\.\":;\n]', r' ',type)
                asp = f'''atom(entity({count}, "{e.lower().strip()}", "{type.lower().strip()}")).'''
                entities.append(asp)
            for relationship in i['relationships']:
                obj = relationship['object']
                sbj = relationship['subject']
                type = relationship['type']
                obj = re.sub(r'[^a-zA-Z0-9,-_]', r' ', obj)
                obj = re.sub(r'[\.\":;\n]', r' ', obj)
                sbj = re.sub(r'[^a-zA-Z0-9,-_]', r' ', sbj)
                sbj = re.sub(r'[\.\":;\n]', r' ', sbj)
                type = re.sub(r'[^a-zA-Z0-9,-_]', r' ', type)
                type = re.sub(r'[\.\":;\n]', r' ', type)
                asp = f'''atom(relation({count}, "{sbj.lower().strip()}", "{obj.lower().strip()}", "{type.lower().strip()}")).'''
                relationships.append(asp)

    except Exception as e:
        print("exception found")
        print("sentence,", count, ": ", "\te:", e)

    with open(f'{dir}llm_output_asp.txt', 'w') as f:
        f.write('\n'.join(entities))
        f.write('\n')
        f.write('\n'.join(relationships))

def asp_translate(gtfile, predfile, dir):
    print("Translating results into ASP for relation reduction.")
    translate_gt(gtfile, dir)
    translate_llm(predfile, dir)


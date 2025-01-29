import json, codecs, os
from pathlib import Path

def preprocess_sci_con(input, output):
    """Preprocess SciErc or CoNLL04 raw data"""
    with open(input, 'r') as f:
        data = json.load(f)

    restructured_data = []

    for item in data:
        sents = item['tokens']
        sentence = " ".join(sents)
      
        entities, relations = [], []
    
        for each in item['entities']:
            entity = " ".join(sents[each['start']:each['end']]).lower()
            type = each['type'].lower()
            entities.append({"entity": entity, "type": type})

        for each in item['relations']:
            subject = entities[each['head']]['entity'].lower()
            object = entities[each['tail']]['entity'].lower()
            type = each['type'].lower()
            relations.append({"subject": subject, "object": object, "type": type})

        restructured_data.append({
            'text': sentence,
            'entities': entities,
            'relationships': relations})

    with open (output, 'wb') as f:
        json.dump(restructured_data, codecs.getwriter('utf-8')(f), indent=2)

def make_sentence(json_input):
    sentences = []

    with open(json_input) as input:
        worksheet = json.load(input)

    for item in  worksheet:
        sentence = ""
        sentence = ' '.join(worksheet[worksheet.index(item)]['tokens'])
        sentences.append(sentence)

    return sentences

def get_entities(json_input):
    entities = []
    entityList = []

    with open(json_input) as input:
        worksheet = json.load(input)

    for item in worksheet:
        e = []

        for i in range(len(worksheet[worksheet.index(item)]['entities'])):
            entity = ""

            #get start and end index for each entity listed for each sentence in JSON
            entityList = worksheet[worksheet.index(item)]['entities']
            start = entityList[i]['start']
            end = entityList[i]['end']
            type = entityList[i]['type']

            #get the string from JSON that corresponds to the entity's index listed in trainData
            for j in range(start, end):
                entity = entity + worksheet[worksheet.index(item)]['tokens'][j] + " "
            entity = entity[:-1]
            e.append([entity.lower(), type.lower()])

        entities.append(e)
    return entities

def get_relation(json_input):
    entities = get_entities(json_input)
    #relationship = []
    relations = []
    with open(json_input) as input:
        worksheet = json.load(input)

    for item in worksheet:
        relationList = worksheet[worksheet.index(item)]['relations']
        relationship = []

        for i in range(len(relationList)):
            #get index within entities
            headIndex = relationList[i]['head']
            tailIndex = relationList[i]['tail']
            type = relationList[i]['type']

            #get entity string from entities list that corresponds to the relationships
            #head represents the subject of the relationship and tail represents the object
            head = entities[worksheet.index(item)][headIndex][0]
            tail = entities[worksheet.index(item)][tailIndex][0]
            relationship.append([head.lower(), tail.lower(), type.lower()])
        relations.append(relationship)

    return relations

def preprocess_ade(input, output):
    """Preprocesses ADE raw dataset"""

    sentences = make_sentence(input)
    entities = get_entities(input)
    relationships = get_relation(input)

    worksheet = []
    for i in range(len(sentences)):
        ent = []
        rels = [] 
        for e in entities[i]: 
            entity = {"entity": e[0], "type": e[1]}
            ent.append(entity)
        
        for r in relationships[i]: 
            relationship = {"subject": r[0], "object": r[1], "type": r[2]}
            rels.append(relationship)

        worksheet.append({"text": sentences[i], "entities": ent, "relationships": rels})
    with open(output, 'wb') as f:
        json.dump(worksheet, codecs.getwriter('utf-8')(f), indent=2)

def preprocess(input, dataset, doc_type):
    if not os.path.isdir("./preprocessed"):
        Path("./preprocessed").mkdir(parents=True, exist_ok=True)
    
    output = "./preprocessed/" + dataset + "_" + doc_type + "_set.json"

    if dataset == "con" or dataset == "sci":
        preprocess_sci_con(input, output)
    elif dataset == "ade":
        preprocess_ade(input, output)
    else:
        print("Dataset not currently supported.")
        
# preprocess("./datasets/scierc/data/train.json", "sci", "train")
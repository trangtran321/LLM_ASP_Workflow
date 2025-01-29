import json, os, re
from pathlib import Path

def delete_gem(input):
    with open (input, 'r') as file:
        worksheet = json.load(file)

    outputs = []
    for item in worksheet: 
        text = item["text"]
        entities = item["response"]["entities"]
        for entity in entities:
            try:
                if not entity["correctlyidentified"] and not entity["correctlytyped"]:
                    entity["corrections"]["entity"] = "delete"
                    entity["corrections"]["type"] = "delete"
            except Exception as e:
                print(text)
                print("Gemini Exception:: ", e, "\n")
            
        outputs.append({"entities": entities})

    return outputs

def delete_openai(input):
    with open (input, 'r') as file:
        worksheet = json.load(file)

    outputs = []
    for item in worksheet: 
        entities = item["entities"]
        for entity in entities:
            try:
                if not entity["correctlyidentified"] and not entity["correctlytyped"]:
                    entity["corrections"]["entity"] = "delete"
                    entity["corrections"]["type"] = "delete"
            except Exception as e:
                print(entity)
                print("GPT Exception:: ", e, "\n")
        outputs.append({"entities": entities})

    return outputs

def get_votes(gemini, gpt):
    delete = []

    for i in range(len(gemini)):
        for j in range(len(gemini[i]['entities'])):
            try:
                gem_corrections = gemini[i]['entities'][j]['corrections']
                gpt_corrections = gpt[i]['entities'][j]['corrections']
                if ('delete' in gem_corrections.values()) or ('delete' in gpt_corrections.values()):
                    delete.append({'entity': gemini[i]['entities'][j]['entity'], 'type': gemini[i]['entities'][j]['type']})
                    
            except Exception as e:
                print("Exception at: ", j)
                print("Exception: ", e)
                print(gemini[i]['entities'][j])
                continue

    return delete 

def reduce(origin_file, delete, batch):

    worksheet = []

    with open(origin_file, 'r') as f:
        worksheet = json.load(f)

    for e in delete:
        for each in worksheet:
                entities = each['entities']
                for entity in entities:
                    if entity['entity'] == e['entity'] and entity['type'] == e['type']:
                        print("removing: ", entity)
                        entities.remove(entity)
    
    path = f"./ensemble/reduced/{batch}"
    
    if not os.path.isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    with open (f'{path}/reduced_entities.json', 'w') as f:
        json.dump(worksheet, f, indent=4)
    return f'{path}/reduced_entities.json'

def entity_reduction(origin_file, dataset, model="gpt-4o-2024-08-06"):
    import gpt_ensemble_predictions as gpt_pred
    import gemini_ensemble_predictions as gem_pred
    import evaluate as eval 
    import asp_translate as asp

    gpt_path, batch = gpt_pred.ensemble_gpt_predict(origin_file, dataset, model)
    print("GPT_PATH: ", gpt_path)
    gem_path = gem_pred.gemini_predictions(origin_file, dataset)
    print("GEMINI PATH: ", gem_path)

    gemini_ = delete_gem(gem_path)
    gpt_ = delete_openai(gpt_path)
    delete = get_votes(gemini_, gpt_)
    reduced_entities = reduce(origin_file, delete, batch)

    eval.compute_score(reduced_entities, dataset)

    gt_path = f"./preprocessed/{dataset}_test_set.json"
    try: 
        gt_path = f"./preprocessed/{dataset}_test_set.json"
    except: 
        print("You have not preprocessed the dataset for testing purposes yet. Please do so before proceeding.")
    
    lst = re.split(r'_', origin_file)
    batch = lst[2]

    if not os.path.isdir(f"../asp/{dataset}/{batch}/ensemble/"):
        Path(f"../asp/{dataset}/{batch}/ensemble").mkdir(parents=True, exist_ok=True)
    dir = f"../asp/{dataset}/{batch}/ensemble/"
    asp.asp_translate(gt_path, reduced_entities, dir)
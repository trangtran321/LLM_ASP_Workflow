import re, jsonlines, json, codecs, os
from pathlib import Path

def extract_response_gpt(input, output):
    worksheet = []
    with open(input, 'r') as file:
        for item in jsonlines.Reader(file):
            worksheet.append(item)

    results = []
    for each in worksheet:
        try:
            result = json.loads(each['response']['body']['choices'][0]['message']['content'])
        except:
            try: 
                result = each['response']['body']['choices'][0]['message']['content']
            except: 
                print(each['response']['body']['choices'][0]['message']['content'])
                break
        results.append(result)

    with open (output, 'wb') as file:
        jsonResponse = json.dump(results, codecs.getwriter('utf-8')(file), indent=2)

def lowercase_dicts(input):
    if isinstance(input, dict):
        return {k.lower():lowercase_dicts(v) for k, v in input.items()}
    
    elif isinstance(input, (list, set, tuple)):
        t = type(input)
        return t(lowercase_dicts(o) for o in input)
    
    elif isinstance(input, str):
        return input.lower()
    else:
        return input

def gpt_postprocess(input, output):
    with open(input, 'r') as file:
        worksheet = json.load(file)
    results = [] 
    index = 0

    for each in worksheet:
        if isinstance(each, str) == True:
            each = each.lower()
        elif isinstance(each, dict) == True:
            each = lowercase_dicts(each)
            results.append(each)
            index += 1
            continue
        
        if("```json\n" in each) or ("\"```json\n" in each):
            each = each.replace("```json\n", "")
        if ("```" in each):
            each = each.replace("```", "")
        if ("\"{\"entities\":" in each) or ("'entities'" in each):
            each = each.replace("\"{\"entities\":", "{\"entities\":")
        if ("'entity'" in each):
            each = each.replace("'entity'", "\"entity\"")
        if ("'type'" in each):
            each = each.replace("'type'", "\"type\"")
        if ("'subject'" in each):
            each = each.replace("'subject'", "\"subject\"")
        if ("'object'" in each):
            each = each.replace("'object'", "\"object\"")
        if ("'relationships'" in each):
            each = each.replace("'relationships'", "\"relationships\"")
        if (": '" in each): 
            each = each.replace(": '", ": \"")
        if ("'," in each): 
            each = each.replace("',", "\",")
        if ("'}" in each): 
            each = each.replace("'}", "\"}")
        if ("\\" in each):
            each = each.replace("\\", "")
        if ("'corrections'" in each):
            each = each.replace("'corrections'", "\"corrections\"")
        if ("'correctlytyped'" in each):
            each = each.replace("'correctlytyped'", "\"correctlytyped\"")
        if ("'correctlyidentified'" in each):
            each = each.replace("'correctlyidentified'", "\"correctlyidentified\"")
        if ("\"entity\": none," in each):
            each = each.replace("\"entity\": none,", "\"entity\": null,")
        if ("\"type\": none" in each):
            each = each.replace("\"type\": none", "\"type\": null")
        if ("\"corrections\":[none, none]" in each):
            each = each.replace("\"corrections\":[none, none]", "\"corrections\":[null, null]")
        if ("\"corrections\":[none," in each):
            each = each.replace("\"corrections\":[none,", "\"corrections\":[null,")
            each = each.replace(", '", ", \"")
            each = each.replace("']", "\"]")
        if("\n" in each):
            each = each.replace("\n", "")

        try: 
            jsonObject = json.loads(each)
            results.append(jsonObject)
            index += 1
        except Exception as e: 
            print("There is an unknown error in the input text at object number:", index)
            print("Exception: ", e)
            print(each)
            print()
        
    with open(output, 'wb') as file:
        jsonResponse = json.dump(results, codecs.getwriter('utf-8')(file), indent=2)

def gemini_postprocess(input, output):
    with open(input, 'r') as file:
        worksheet = json.load(file)
    index = 0
    for each in worksheet:
        try:
            if isinstance(each['response'], str) == True:
                response = each['response'].lower()
                if("```json\n" in response):
                    response = response.replace("```json\n", "")
                if ("```" in response):
                    response = response.replace("```", "")
                response = json.loads(response)
                each['response'] = response
                index += 1
            elif isinstance(each['response'], dict) == True:
                each['response'] = lowercase_dicts(each['response'])
                index += 1
                continue
        except Exception as e: 
            print("There is an unknown error in the input text at object number:", index)
            print("Exception: ", e)
            print(each)
            print()

    with open(output, 'wb') as file:
        jsonResponse = json.dump(worksheet, codecs.getwriter('utf-8')(file), indent=4)

def postprocess(input, model, ensemble): 
    if not os.path.isdir("./inference/postprocessed"):
        Path("./inference/postprocessed").mkdir(parents=True, exist_ok=True)
        
    split_input = re.split(r'/', input)
    filename = re.split(r'\.', split_input[-1])
    output = f'''./inference/postprocessed/{filename[0]}_'''

    if model == 'GPT':
        print("Post Processing GPT Output")
        if ensemble == True:
            count = 0
            filename = re.split(r'_[0-9]+', filename[0])[0]
            path = f"./ensemble/output/{count}"
            while os.path.exists(f'{path}/{filename}.json') and os.path.isdir(path):
                print("entering while gpt", path)
                count += 1
                path = f"./ensemble/output/{count}"

            if not os.path.isdir(path):
                Path(path).mkdir(parents=True, exist_ok=True)
                
            extract_response_gpt(input, f'./ensemble/output/{filename}_{count}_extracted.json')
            gpt_postprocess(f'./ensemble/output/{filename}_{count}_extracted.json', f'{path}/{filename}.json')
            return f'{path}/{filename}.json', count 
        else:
            extract_response_gpt(input, f'{output}extracted.json')
            gpt_postprocess(f'{output}extracted.json', f'{output}post.json')
            return f'{output}post.json'

    elif model == 'Gemini': 
        print("Post Processing Gemini Output")
        count = 0
        path = f"./ensemble/output/{count}"
        filename = re.split(r'_[0-9]+', filename[0])[0]

        while os.path.exists(f'{path}/{filename}.json') and os.path.isdir(path):
            print("entering while gem:", path)
            count += 1
            path = f"./ensemble/output/{count}"
        
        if not os.path.isdir(path):
            Path(path).mkdir(parents=True, exist_ok=True)

        gemini_postprocess(input, f'{path}/{filename}.json')
        return f'{path}/{filename}.json'
    
    else: 
        raise Exception("Invalid input for post processing.")

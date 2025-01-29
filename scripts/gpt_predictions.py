import os, sys, re, json, jsonlines
from pathlib import Path
from time import sleep
from openai import OpenAI

def batch_format(dataset, testfile, model):
    sys.path.insert(0, '../datasets')
    if dataset == "ade":
        import ade.ade_config as config
    elif dataset == "sci":
        import scierc.sci_config as config
    elif dataset == "con":
        import conll04.con_config as config
    else: 
        raise ValueError("That dataset is not supported. Please enter 'ade', 'sci', or 'con' for ADE, SciErc, or CoNLL04 respectively.")

    with open (testfile, 'r') as f:
        worksheet = json.load(f)
    messages = []

    system_content = f'''You are a natural language processing researcher working in the {config.DOMAIN} domain. {config.EXPERIENCE} Your job is to extract entities from excerpts of text from {config.DOMAIN} excerpts.
In this domain, an entity is an abstract notion that has its own independent existence. Entities specify pieces of information or objects within a text that carry particular significance.
In your work, you will only extract specific types of entites and relationships.
The types of entities and relationships are defined here.\n{config.CONTEXT}'''
    
    prompt_pre = f'''Give me the entities from the following text. Do not include any explanation, only provide a RF8259 compliant JSON response without deviation.
        The keys for the output JSON should be {config.ANSWER_KEYS}
        Do not use any other keys for the JSON response. 
        Ensure that you are outputting the entire entity and its type.
        Here is one example: {config.EXAMPLE}

        Evaluate this text:
'''

    count = 1
    for item in worksheet:
        text = item['text']
        prompt = prompt_pre + str(text)

        message = [{'role' : 'system', 'content': system_content},
                   {'role' : 'user', 'content': prompt}]

        content = {"custom_id": f"request-{count}",
                   "method": "POST",
                   "url": "/v1/chat/completions",
                   "body": {
                        "model": model,
                        "messages": message,
                        "temperature": 0,
                        "logprobs": True
                    }
                   }
        messages.append(content)
        count += 1

    if not os.path.isdir("./inference/requests"):
        Path("./inference/requests").mkdir(parents=True, exist_ok=True)
    
    count = 0 
    output_file = f'''./inference/requests/{dataset}_{count}.jsonl'''
    while os.path.exists(output_file):
        count = re.split(r'_', output_file)[-1]
        count = int(re.split(r'\.', count)[0])
        count += 1
        output_file = f'''./inference/requests/{dataset}_{count}.jsonl'''

    with jsonlines.open(output_file, 'w') as w:
        w.write_all(messages)

    return output_file

def run_batch(input_file, dataset):
    import config as config

    client = OpenAI(api_key = config.OPEN_AI_KEY)

    batch_input_file = client.files.create(
      file=open(input_file, "rb"),
      purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch_ = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description": "JERE task"
        }
    )

    print("GPT Batch ID: ", batch_.id)
    while (batch_.status != "completed"):
        batch_ = client.batches.retrieve(batch_.id)
        sleep(90)
    
    file_id = batch_.output_file_id 
    print("GPT Output File ID: ", file_id)

    content = client.files.content(file_id)
    content_ = re.split(r'\n', content.text)
    content_json = []

    for i in content_: 
        try:
            content_json.append(json.loads(i))
        except Exception as e:
            print("---------------------------")
            print("=== Invalid json formatting::\nException: ", e, " === ")
            print("---------------------------\n")
            continue
    
    count = 0

    if not os.path.isdir("./inference/outputs"):
        Path("./inference/outputs").mkdir(parents=True, exist_ok=True)

    output_file = f'''./inference/outputs/{dataset}_gpt_{count}.jsonl'''
    
    while os.path.exists(output_file):
        count = re.split(r'_', output_file)[-1]
        count = int(re.split(r'\.', count)[0])
        count += 1
        output_file = f'''./inference/outputs/{dataset}_gpt_{count}.jsonl'''

    with jsonlines.open(output_file, 'w') as w:
        w.write_all(content_json)
    
    import postprocessing as processor 
    post = processor.postprocess(output_file, "GPT", False)
    print(post)
    return post

def run_gpt_predictions(dataset, model="gpt-4o-2024-08-06"): 
    import evaluate as eval
    import asp_translate as asp 

    gt_file = "./preprocessed/" + dataset + "_test_set.json"

    if not os.path.exists(gt_file): 
        import preprocessing as preprocessor 
        preprocessor.preprocess(f"../datasets/{dataset}/data/test.json", dataset, "test")

    input_file = batch_format(dataset, gt_file, model)
    output_file = run_batch(input_file, dataset)
    eval.compute_score(output_file, dataset)

    #translate gt and predictions into asp for evaluation in asp module 
    lst = re.split(r'_', output_file)
    batch = lst[2]

    if not os.path.isdir(f"../asp/{dataset}/{batch}/"):
        Path(f"../asp/{dataset}/{batch}/").mkdir(parents=True, exist_ok=True)

    dir = f"../asp/{dataset}/{batch}/"
    asp.asp_translate(gt_file, output_file, dir)
    
    



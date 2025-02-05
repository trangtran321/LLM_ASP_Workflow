import os, sys, re, json, jsonlines
from pathlib import Path
from time import sleep
from openai import OpenAI

def format(predfile, dataset, model="gpt-4o-2024-08-06"):
    sys.path.insert(0, '../datasets')
    if dataset == "ade":
        import ade.ade_config as config
        gtfile = "./preprocessed/ade_test_set.json"
    elif dataset == "scierc":
        import scierc.sci_config as config
        gtfile = "./preprocessed/sci_test_set.json"
    elif dataset == "conll04":
        import conll04.con_config as config
        gtfile = "./preprocessed/con_test_set.json"
    else: 
        raise Exception("You have not properly preprocessed the test set for the dataset yet. Or the dataset is not supported, please either enter 'ade', 'sci' or 'con'.")
    
    with open (gtfile, 'r') as f:
        worksheet = json.load(f)
    
    with open(predfile, 'r') as f:
        predictions = json.load(f)

    system_prompt = f'''You are a natural language processing researcher working in the {config.DOMAIN} domain. {config.EXPERIENCE} You are working with a team to annotate a dataset of excerpts from {config.DOMAIN} research papers. Your job is to verify that the entities and relationships that have been annotated are accurate and consistent with the annotation guidelines.
In this domain, an entity is an abstract notion that has its own independent existence. Entities specify pieces of information or objects within a text that carry particular significance.
The annotation guidelines are : \n{config.CONTEXT}'''
    prompt_pre =f'''You will be given an input that contains a sentence and a list of entites and their types.
Your job is to determine if the entities previously extracted are correctly identified and classfied.
If the entity is incorrectly identified, output 'False' for both "correctlyIdentified" and "correctlyTyped". In the 'corrections' key, output "Delete" for both 'Entity' and 'Type' keys.
If the entity is correctly identified but the type is incorrect, output 'True' for "correctlyIdentified" and 'False' for "correctlyTyped". In the 'corrections' key, output the corrected type in the 'Type' key.
If the entity is correctly identified and the type is correct, output 'True' for both "correctlyIdentified" and "correctlyTyped". In the 'corrections' key, output an empty list.
Here is one example: {config.ENSEMBLE_EXAMPLE}
Do not include any explanation, only provide a RF8259 compliant JSON response without deviation.
The keys for the output JSON should be {config.ENSEMBLE_ANSWER_KEYS}
Do not use any other keys for the JSON response. DO NOT add more entities or relationships. 
Ensure that you are outputting the entire entity and its type.

Evaluate the following text:'''

    count = 1
    messages = []
    print("Model: ", model)
    for i in range(len(worksheet)):
        #text needs to go into {'role':user, 'content': TEXT}
        #gt results need to go into {'role': assistant, 'content': ANSWERS}

        text = worksheet[i]["text"]
        results = "{\"entities\":" + str(predictions[i]["entities"])
        prompt = prompt_pre + text + results

        message = [{'role' : 'system', 'content': system_prompt},
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
    
    if not os.path.isdir("./ensemble/gpt_input"):
        Path("./ensemble/gpt_input").mkdir(parents=True, exist_ok=True)

    output_file = f'''./ensemble/gpt_input/{dataset}.jsonl'''

    #Write batch data to jsonl file
    with jsonlines.open(output_file, 'w') as w:
        w.write_all(messages)
    
    return output_file 

def ensemble_gpt_predict(predfile:str, dataset:str, model="gpt-4o-2024-08-06"): 
    """
    Description: Function calls openAI Batch API to ask GPT model if previously predicted entities are correctly identified and typed.
    Input: predfile - path to the postprocessed prediction file
           dataset  - dataset to be used, must correspond to the predfile's dataset ('ade', 'sci' or 'con')
           model    - gpt model to perform inference with 
    """
    import config as config

    client = OpenAI(api_key = config.OPEN_AI_KEY)

    output_file = format(predfile, dataset, model)

    batch_input_file = client.files.create(
      file=open(output_file, "rb"),
      purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    print("Please wait while we get GPT's vote for entity reduction.")
    batch_ = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description": "gpt-ensemble"
        }
    )

    print(batch_.id)
    while (batch_.status != "completed"):
        if batch_.status == "failed":
            print("Batch failed. OpenAI error.")
            break
        batch_ = client.batches.retrieve(batch_.id)
        sleep(90)
    
    file_id = batch_.output_file_id 
    print(file_id)
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
    
    if not os.path.isdir("./ensemble/output/raw"):
        Path("./ensemble/output/raw").mkdir(parents=True, exist_ok=True)

    count = 0
    output_file = f'''./ensemble/output/raw/{dataset}_gpt_{count}.jsonl'''
    while os.path.exists(output_file):
        count = re.split(r'_', output_file)[-1]
        count = int(re.split(r'\.', count)[0])
        count += 1
        output_file = f'''./ensemble/output/raw/{dataset}_gpt_{count}.jsonl'''

    with jsonlines.open(output_file, 'w') as w:
        w.write_all(content_json)
    
    import postprocessing as processor 
    postprocessed_path = processor.postprocess(output_file, "GPT", True)
    return postprocessed_path


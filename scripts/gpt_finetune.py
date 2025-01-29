import random, sys, jsonlines, json, os
from openai import OpenAI
from pathlib import Path

def ft_format(file, dataset, doc_type):
    sys.path.insert(0, '../datasets')
    if dataset == "ade":
        import ade.ade_config as config
    elif dataset == "sci":
        import scierc.sci_config as config
    elif dataset == "con":
        import conll04.con_config as config
    else: 
        raise ValueError("That dataset is not supported. Please enter 'ade', 'sci', or 'con' for ADE, SciErc, or CoNLL04 respectively.")
    
    filename = ""
    num = 0
    worksheet = []

    if doc_type == "train":
        num = int(config.TOTAL_DATAPOINTS * 0.09)
        filename = "training.jsonl"
    else:
        num = int(config.TOTAL_DATAPOINTS * 0.01)
        filename = "validation.jsonl"
    
    with open (file, 'r') as f:
        w = json.load(f)
        for i in range(num):
            worksheet.append(random.choice(w))

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

    #message is one array that consists of all three roles, list of message is messages to be given to finetuning
    #user = sentence will be inputted into the prompt
    #assistant = answer gpt should give back

    for item in worksheet:
        text = item['text']
        prompt = prompt_pre + text
        results = "{\"entities\":" + str(item['entities'])
        results = results + ", \"relationships\":" + str(item['relationships']) + "}"
        message = {"messages" : [{'role' : 'system', 'content': system_content},
                                 {'role' : 'user', 'content': prompt},
                                 {'role' : 'assistant', 'content': results}]}
        messages.append(message)


    full_path = f"./finetune/{dataset}_{filename}"
    #Write finetuning data to jsonl file
    with jsonlines.open(full_path, 'w') as w:
        w.write_all(messages)
    return full_path

def finetune(dataset, model="gpt-4o-2024-08-06"):
    
    import config as config

    openAI_Key = config.OPEN_AI_KEY
    client = OpenAI(api_key = openAI_Key)   

    if not os.path.isdir("./finetune"):
        Path("./finetune").mkdir(parents=True, exist_ok=True)
    
    #if training set has not been preprocessed, do so now
    filepath = "./preprocessed/" + dataset + "_train_set.json"

    if not os.path.exists(filepath): 
        import preprocessing as preprocessor 
        preprocessor.preprocess(f"../datasets/{dataset}/data/train.json", dataset, "train")

    #take preprocessed training set and convert in GPT file format specific for finetuning 
    training_file = ft_format(filepath, dataset, "train")
    training = client.files.create(
        file=open(training_file, "rb"),
        purpose="fine-tune"
    )

    validation_file = ft_format(filepath, dataset, "validation")
    validation = client.files.create(
        file=open(validation_file, "rb"),
        purpose="fine-tune"
    )

    print("GPT Training File ID: ", training.id)
    print("GPT Validation File ID: ", validation.id)

    # call API to finetune 
    job = client.fine_tuning.jobs.create(
        training_file=training.id,
        model=model,
        validation_file=validation.id,
        hyperparameters={
            "n_epochs": 5, 
            "batch_size": 1, 
            "learning_rate_multiplier": 2}
    )

    print("GPT Finetuned Model ID: ", job.id)



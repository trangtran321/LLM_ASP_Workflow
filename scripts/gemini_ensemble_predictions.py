import codecs, json, os, pathlib, re, sys
from time import sleep
import google.generativeai as genai

sys.path.append(os.path.abspath(os.path.join(os.path.pardir, 'config')))
import config as config


GOOGLE_API_KEY = config.GEMINI_KEY

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_input(gtfile, predfile, i):
    with open(gtfile) as file:
        worksheet = json.load(file)
    sentence = worksheet[i]['text']
    
    with open (predfile) as file:
        worksheet = json.load(file)
    entities = worksheet[i]['entities']
    relationships = worksheet[i]['relationships']

    return {"input": sentence, "output": {"entities": entities, "relationships": relationships}}

def relationEntity (input, data_config):
    
    prompt =f'''You are a natural language processing researcher working in the {data_config.DOMAIN} domain. {data_config.EXPERIENCE} You are working with a team to annotate a dataset of excerpts from {data_config.DOMAIN} research papers. Your job is to verify that the entities and relationships that have been annotated are accurate and consistent with the annotation guidelines.
    In this domain, an entity is an abstract notion that has its own independent existence. Entitees specify pieces of information or objects within a text that carry particular significance.
The annotation guidelines are : \n{data_config.CONTEXT}

    You will be given an input that contains a sentence and a list of entites and their types.
    Your job is to determine if the entities previously extracted are correctly identified and classfied.
    If the entity is incorrectly identified, output 'False' for both "correctlyIdentified" and "correctlyTyped". In the 'corrections' key, output "Delete" for both 'Entity' and 'Type' keys.
    If the entity is correctly identified but the type is incorrect, output 'True' for "correctlyIdentified" and 'False' for "correctlyTyped". In the 'corrections' key, output the corrected type in the 'Type' key.
    If the entity is correctly identified and the type is correct, output 'True' for both "correctlyIdentified" and "correctlyTyped". In the 'corrections' key, output an empty list.

    Here is one example: {data_config.ENSEMBLE_EXAMPLE}

    Do not include any explanation, only provide a RF8259 compliant JSON response without deviation.
    The keys for the output JSON should be {data_config.ENSEMBLE_ANSWER_KEYS}
    Do not use any other keys for the JSON response. DO NOT add more entities or relationships. 
    Ensure that you are outputting the entire entity and its type.
    
    Evaluate the following text:'''
    prompt = prompt + input["input"] + " Entities: " + str(input["output"]["entities"])
    
    response = model.generate_content([prompt],
        safety_settings=[
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
        ])

    return response

def gemini_predictions(predfile:str, dataset:str):
    """
    Description: Function calls Gemini if previously predicted entities are correctly identified and typed.
    Input: predfile - path to the postprocessed prediction file
           dataset  - dataset to be used, must correspond to the predfile's dataset ('ade', 'sci' or 'con')
    Output: postprocessed replies from Gemini 
    """
    sys.path.insert(0, '../datasets')
    if dataset == "ade":
        import ade.ade_config as data_config
        gtfile = "./preprocessed/ade_test_set.json"
    elif dataset == "sci":
        import scierc.sci_config as data_config
        gtfile = "./preprocessed/sci_test_set.json"
    elif dataset == "con":
        import conll04.con_config as data_config
        gtfile = "./preprocessed/con_test_set.json"
    else: 
        raise Exception("You have not properly preprocessed the test set for the dataset yet. Or the dataset is not supported, please either enter 'ade', 'sci' or 'con'.")
    
    responseList = []
    j = 1

    print("Please wait while we get Gemini's vote for entity reduction.")
    for i in range(0, data_config.TEST_DATAPOINTS):
        j = j + 1
        input = get_input(gtfile, predfile, i)
        try:    
            response = relationEntity(input, data_config)
            print(i)
            responseList.append({"text": input["input"], "response":response.text})
        except Exception as error:
            print("Error at: ", i)
            print("Error: ", error)
            responseList.append({"text": input["input"], "response": None})
        
        if j == 15:
            sleep(60)
            j = 1

    if not os.path.isdir("./ensemble/output/raw"):
        pathlib.Path("./ensemble/output/raw").mkdir(parents=True, exist_ok=True)
    
    count = 0 
    output_file = f'''./ensemble/output/raw/{dataset}_gemini_{count}.json'''

    while os.path.exists(output_file):
        count = re.split(r'_', output_file)[-1]
        count = int(re.split(r'\.', count)[0])
        count += 1
        output_file = f'''./ensemble/output/raw/{dataset}_gemini_{count}.json'''

    with open (output_file, 'wb') as f:
        jsonResponse = json.dump(responseList, codecs.getwriter('utf-8')(f), indent=4)
    
    print(output_file)
    
    import postprocessing as processor 
    postprocessed_path = processor.postprocess(output_file, "Gemini", True)
    return postprocessed_path



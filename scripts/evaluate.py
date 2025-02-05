import json
from sklearn.metrics import precision_recall_fscore_support as prfs

#sent refers to the sentence being evaluated
#dct refers to either an entity or relationship that was found
def convert_to_tuple(sent, dct):

    #dct is an entity
    if len(dct) == 2:
        try:
            return ((dct['type'], dct['entity']))
        except:
            print("There is an unknown syntax error in: ", dct)

    #dct is a relationship
    else:
        sbj_type = 'other'
        obj_type = 'other'
        #get the entity types for both subject and object
        for e in sent:
            i = sent.index(e)
            if dct['subject'] in e['entity']:
                sbj_type = sent[i]['type']
            if dct['object'] in e['entity']:
                obj_type = sent[i]['type']
        return ((dct['type'],
                dct['subject'], sbj_type,
                dct['object'], obj_type
                ))


def convert_wksht(pred_path:str, gt_path: str):
    """
    Description: Convert all found entities and relationships into tuples
    Input: pred_path - str: path to postprocessed predicted output from LLM
           include_types - bool: if True, evaluates the type definitions in entities with in relations as well
                                 if False, only evaluates entities and relations found - not types. 
    Return: four lists that represent all found entities and relationships in either gt or pred output"""

    with open (gt_path) as f:
        gt_wksht = json.load(f)
    with open (pred_path) as f:
        pred_wksht = json.load(f)

    gt_entities = []
    gt_relationships = []
    pred_entities = []
    pred_relationships = []

    for i in range(len(gt_wksht)):
        # print(i)
        gt_entities.append([convert_to_tuple(gt_wksht[i], e) for e in gt_wksht[i]['entities']])
        gt_relationships.append([convert_to_tuple(gt_wksht[i]['entities'], r) for r in gt_wksht[i]['relationships']])

        pred_entities.append([convert_to_tuple(pred_wksht[i], e) for e in pred_wksht[i]['entities']])
        pred_relationships.append([convert_to_tuple(pred_wksht[i]['entities'], r) for r in pred_wksht[i]['relationships']])

    return gt_entities, gt_relationships, pred_entities, pred_relationships

#list of tuples is converted into a 1D list
def convert_1D(gt, pred):
    """
    Description: List of tuples is converted to 1D list for evaluation with sklearn library
    Input: gt and pred are lists of tuples
    Output: Flattening of tuples into 1D list for gt and pred
    """
    converted_gt = []
    converted_pred = []
    types = set()

    for (gt_objs, pred_objs) in zip(gt, pred):
        union = set()
        union.update(gt_objs)
        union.update(pred_objs)
       
        for s in union:
            try:
                type = str(s[0])
                types.update((type, type))

                if s in gt_objs:
                    converted_gt.append(type)
                else:
                    converted_gt.append('')

                if s in pred_objs:
                    converted_pred.append(type)
                else:
                    converted_pred.append('')
            except: 
                print(s)
    return converted_gt, converted_pred


def compute_score(pred_path, dataset):
    """
    Description: Compute the score of predicted output to ground truth.
    Input: pred_path - str: path to predicted outcomes from llm
           print_scores - bool: print outcomes to terminal
    Return: F1-micro and F1-macro scores for entities, relationships with consideration to entity types 
            and relationships without consideration to entity types
    """
    try: 
        gt_path = "./preprocessed/" + dataset + "_test_set.json"
    except: 
        print("You have not preprocessed the dataset for testing purposes yet. Please do so before proceeding.")


    dataset_types = {
                "ade": {"entity_types" : ["adverse-effect", "drug"], "relationship_types" : ["adverse-effect"]},
                "scierc": {"entity_types" : ['task', 'method', 'metric', 'material', 'otherscientificterm', 'generic'], "relationship_types" : ['used-for', 'feature-of', 'hyponym-of', 'part-of', 'compare', 'conjunction', 'evaluate-for']},
                "conll04": {"entity_types" : ["loc", "peop", "other", "org"], "relationship_types" : ["kill", "live_in", "orgbased_in", "work_for", "located_in"]}
               }

    gt_entities, gt_relationships, pred_entities, pred_relationships = convert_wksht(pred_path, gt_path)

    gt, pred = convert_1D(gt_entities, pred_entities)
    entity_micro = prfs(gt, pred, average='micro', zero_division=0, labels=dataset_types[dataset]["entity_types"])
    entity_macro = prfs(gt, pred, average='macro', zero_division=0, labels=dataset_types[dataset]["entity_types"])

    
    print("\nEntity F1_Micro:: ")
    print(entity_micro)
    print("Entity F1_Macro:: ")
    print(entity_macro, "\n")

    gt, pred = convert_1D(gt_relationships, pred_relationships)
    relationship_micro = prfs(gt, pred, average='micro', zero_division=0, labels=dataset_types[dataset]['relationship_types'])
    relationship_macro = prfs(gt, pred, average='macro', zero_division=0, labels=dataset_types[dataset]['relationship_types'])


    print("Relationship F1_Micro:: ")
    print(relationship_micro)
    print("Relationship F1_macro:: ")
    print(relationship_macro)
    print("\n")

    micro_results = [entity_micro, relationship_micro] 
    macro_results = [entity_macro, relationship_macro] 

    return micro_results, macro_results






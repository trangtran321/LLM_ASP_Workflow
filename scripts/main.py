'''
January 15, 2025 
Trang Tran

This script will allow you to perform joint entity-relation extraction using OpenAI's GPT models on three benchmark datasets - ADE, SciErc, and CoNLL04.
There is an option to perform entity reduction using the 'ensemble', relation reduction using the asp module, or to finetune a model on 10% of the data before 
performing JERE. 
'''

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that will either perform JERE, finetune or ensemble verification on specified dataset."
    )

    parser.add_argument("--purpose", required=True, type=str, help="Enter either 'inference' to perform JERE, 'finetune', or 'ensemble' to perform entity reduction on a specified output. If 'ensemble' chosen, filepath of post processed outputs must be supplied.")
    parser.add_argument("--dataset", required=True, type=str, help="Enter dataset to perform JERE, finetuning or ensemble on. Options are 'ade', 'scierc' or 'conll04'.")
    parser.add_argument("--model", required=False, type=str, help="Enter the openAI model you'd like to use. Default is: gpt-4o-2024-08-06")
    parser.add_argument("--filepath", required=False, type=str, help="Enter the filepath of postprocessed JERE predictions to perform ensemble entity reduction.")
    args = parser.parse_args()
    
    purpose = args.purpose
    dataset = args.dataset
    model = args.model
    filepath = args.filepath

    if dataset != "ade" and dataset != "scierc" and dataset != "conll04":
        raise Exception("Invalid input: Please choose between 'ade', 'scierc' or 'conll04'.")

    if purpose == "inference":
        print(f"Performing JERE on {dataset} data.")
        import gpt_predictions as infer
        infer.run_gpt_predictions(dataset)

    elif purpose == "finetune":
        print(f"Performing finetuning job on {dataset}.")
        import gpt_finetune as finetune
        finetune.finetune(dataset)
        
    elif purpose == "ensemble":
        print("Performing Entity-Reduction using ensemble.")
        if not os.path.exists(filepath):
            raise Exception("Invalid filepath: filepath does not exist. Please enter a valid filepath.")
        
        import ensemble_entity_reduction as ensemble
        
        if not model: 
            model = "gpt-4o-2024-08-06"

        ensemble.entity_reduction(filepath, dataset, model)

    else:
        raise Exception("Invalid input: Please choose between 'inference', 'finetune', or 'ensemble'.")

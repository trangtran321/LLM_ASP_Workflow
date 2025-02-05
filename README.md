# LLM + ASP Workflow for Joint Entity-Relation Extraction

## JERE with GPT 
To perform joint entity-relation extraction you can refer to the scripts folder. 

To run the base program you will need to have the OpenAI Python library which can be done with :

    pip install openai

You will need an OpenAI key placed in a config.py file within the scripts folder as well. 

The full list of dependencies used in the scripts can be found in the requirements.txt file. 

Once all dependencies are downloaded, enter the scripts directory in terminal and run 

    python3 main.py --purpose inference --dataset <choose dataset> --model <default is gpt-4o-2024-08-06>

For example, to run inference on GPT4o on the ADE dataset: 

    python3 main.py --purpose inference --dataset ade --model gpt-4o-2024-08-06

## LLM Ensemble: Entity Reduction Module
This module can be called within the scripts directory as well. It performs model ensembling to determine the correctness of previous outputs in regards to entities found. It requires both OpenAI and Gemini keys. 

To use, you must have previously ran the JERE on the particular dataset you'd like the ensemble of LLMs to verify. 

To run:
    
    python3 main.py --purpose ensemble --dataset <choose dataset> --filepath <path of post processed results from JERE> --model <default is gpt-4o-2024-08-06>

## Finetuning
This script is optional and finetunes gpt-4o-2024-08-06 on 10% of the chosen dataset's corpus. 

To run: 

    python3 main.py --purpose finetune --dataset <choose dataset> --model <default is gpt-4o-2024-08-06>

## LLM + ASP : Relation Reduction Module

The code listed in the paper can be found in the ASP program compare_2.lp. Comments (line start with %) are added to explain the meaning of each rule. 

To run the program, clingo 5.4.0 (or higher version) could be used. It can be downloaded from https://github.com/potassco/clingo.

Output from the LLM models need to be converted into a collection of atoms of the form: 

    * atom(entity(S,E,T)): $E$ is an entity of the type $T$ in the sentence $S$; and 
    * atom(relation(S,E,F,R)): relation of the type $R$ between entities $E$ and $F$ in the sentence $S$.

Ground truth is encoded as collections of atoms of the form: 

    * entity(S,E,T): $E$ is an entity of the type $T$ in the sentence $S$; and 
    * relation(S,E,F,R): relation of the type $R$ between entities $E$ and $F$ in the sentence $S$.

(see script ... )

For ConLL04 and SciErc, a file named "domain.lp", which encodes the type specification used in our experiments, is included.  

Assume that the llm_ouput.txt and ground_truth.txt files encode the LLM output and the ground truth, respectively. 

To run the ASP code without using the type specification, the following command should be used:

    clingo [PATH_TO]compare_2.lp [PATH_TO]llm_output.txt [PATH_TO]ground_truth.txt 

where [PATH_TO] specifies the folder of the program or data. 

To run the ASP code with the type specification, the following command should be used:

    clingo [PATH_TO]compare_2.lp [PATH_TO]llm_output.txt [PATH_TO]ground_truth.txt [PATH_TO]domain.lp 
    where domain.lp is the file encoding the type specification.  

The output of this program will contain atoms of the 

    * f1_entity(t, tp, fp, fn): the numbers of TP, FP, and FN of entity of type t are tp, fp, and fn, respectively.    
    * f1_relation(t, tp, fp, fn): the numbers of TP, FP, and FN of relation of type t are tp, fp, and fn, respectively.  
    * statistics  

For example, the command (runs in the folder containing the LLM output and ground truth of 
the ConLL04 dataset with the fine tuning model with 5% traning data and type specification) 
that looks as follows: 

    clingo ../../../compare_2.lp llm_output_asp.txt ground_truth_asp.txt ../../domain.lp 

which outputs 

    clingo version 5.4.0
    Reading from ../../../compare_2.lp ...
    Solving...
    f1_entity("org",165,42,30)
    f1_entity("other",56,119,76)
    f1_entity("loc",361,100,51)
    f1_entity("peop",284,32,34)
    f1_relation("orgbased_in",69,49,27)
    f1_relation("located_in",69,51,21)
    f1_relation("live_in",56,103,41)
    f1_relation("work_for",51,12,25)
    f1_relation("kill",23,17,24)
    [['org', 165, 42, 30], ['other', 56, 119, 76], ['loc', 361, 100, 51], ['peop', 284, 32, 34]]
    [['orgbased_in', 69, 49, 27], ['located_in', 69, 51, 21], ['live_in', 56, 103, 41], ['work_for', 51, 12, 25], ['kill', 23, 17, 24]]
    [['org', 0.8208955223880597], ['other', 0.3648208469055375], ['loc', 0.8270332187857962], ['peop', 0.8958990536277602]]
    [['orgbased_in', 0.6448598130841121], ['located_in', 0.6571428571428571], ['live_in', 0.4375], ['work_for', 0.7338129496402879], ['kill', 0.5287356321839081]]
    f1 macro entity:  0.7271621604267884
    f1 macro relation:  0.6004102504102331
    f1 micro entity:  0.7815884476534295
    f1 micro relation:  0.5916114790286976 
    Answer: 1

    SATISFIABLE

    Models       : 1+
    Calls        : 1
    Time         : 0.141s (Solving: 0.00s 1st Model: 0.00s Unsat: 0.00s)
    CPU Time     : 0.077s

The statistics can be used for reports. 

## Prompts
We include a list of prompts used in our preliminary study on prompting techniques that were used to inform the creation of our prompt templates for this paper in the prestudy folder. 

All prompts used for each dataset are also included in their resepctive folders within the datasets directory. This includes the JERE prompt as well as the ensemble prompt. 


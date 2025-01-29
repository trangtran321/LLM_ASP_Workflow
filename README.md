# LLM + ASP Workflow for Joint Entity-Relation Extraction

## JERE with GPT 
To perform joint entity-relation extraction you can refer to the scripts folder. 

To run the base program you will need to have the OpenAI Python library which can be done with :

    pip install openai

You will need an OpenAI key placed in a config.py file within the scripts folder as well. 

The full list of dependencies used in the scripts can be found in the requirements.txt file. 

Once all dependencies are downloaded, enter the scripts directory in terminal and run 

    python3 main.py --purpose inference --dataset <choose dataset> --model <default is gpt-4o-2024-08-06>

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

    clingo compare_2.lp llm_output.txt ground_truth.txt 

To run the ASP code with the type specification, the following command should be used:

    clingo compare_2.lp llm_output.txt ground_truth.txt domain.lp 
    where domain.lp is the file encoding the type specification.  

The output of this program will contain atoms of the 

    * f1_entity(t, tp, fp, fn): the numbers of TP, FP, and FN of entity of type t are tp, fp, and fn, respectively.    
    * f1_relation(t, tp, fp, fn): the numbers of TP, FP, and FN of relation of type t are tp, fp, and fn, respectively.   

Sometime, it is useful to add "--outf=0 -V0 --out-atomf=%s. --quiet=1,2,2 | head -n1 | tr ' ' '\n' | grep f1" (OS Max command line)to the command line to formulate the output. For example,  

    clingo compare_2.lp llm_output.txt ground_truth.txt --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2 | head -n1 | tr ' ' '\n' | grep f1 

for the ConLL04 dataset (fine tuning with 5% traning data) looks as follows: 

    f1_entity("org",165,49,30).
    f1_entity("other",41,75,91).
    f1_entity("loc",355,82,57).
    f1_entity("peop",277,30,41).
    f1_relation("orgbased_in",72,94,24).
    f1_relation("located_in",55,45,35).
    f1_relation("live_in",62,137,35).
    f1_relation("work_for",55,34,21).
    f1_relation("kill",19,18,28).

## Prompts
We include a list of prompts used in our preliminary study on prompting techniques that were used to inform the creation of our prompt templates for this paper in the prestudy folder. 

All prompts used for each dataset are also included in their resepctive folders within the datasets directory. This includes the JERE prompt as well as the ensemble prompt. 


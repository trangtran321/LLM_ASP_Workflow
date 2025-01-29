% identifying the sentence considered in the test 

in_set(S) :- atom(entity(S, _, _)). 
in_set(S) :- atom(relation(S, _, _, _)).

% for statistics: couting the number of entities/relations in the ground truth  

true_count_entity(C, T) :- type_of_entity(T), C = #count{S, E  : entity(S, E, T), in_set(S)}.
true_count_relation(C, T) :- type_of_relation(T), C = #count{S, E, E'  : relation(S, E, E', T), in_set(S)}.

% identifying TP, FP, FN entities 

true_pos(S, E, T) :- entity(S, E, T), atom(entity(S, E, T)). 
false_pos(S, E, T) :- atom(entity(S, E, T)), not entity(S, E, T). 
false_neg(S, E, T) :- entity(S, E, T), in_set(S), not atom(entity(S, E, T)).

% defining types from entities and relations 

type_of_entity(T) :- entity(_, _, T). 
type_of_relation(T) :- relation(_, _, _, T). 

% computing TP, FP, FN for each type of entities 

true_pos_count(C, T) :- type_of_entity(T), C = #count{S, E  : true_pos(S, E, T)}.
false_pos_count(C, T) :- type_of_entity(T), C = #count{S, E  : false_pos(S, E, T)}.
false_neg_count(C, T) :- type_of_entity(T), C = #count{S, E  : false_neg(S, E, T)}.

% defining TP of relation - only considered relations that 
% conform to the type spec 

relation_true_pos(S, E, E', R) :- 
     atom(relation(S, E, E', R)), 
     ok_type(S, E, E', R),
     relation(S, E, E', R).   

% defining FP of relation - only considered relations that 
% conform to the type spec and consistent with the entities 

relation_false_pos(S, E, E', R) :- 
     atom(relation(S, E, E', R)), 
     ok_type(S, E, E', R), 
     not false_declaration(S, E, E', R), 
     not relation(S, E, E', R).   

% defining FN of relation - not present in the prediction  

relation_false_neg(S, E, E', R) :-
     relation(S, E, E', R), 
     in_set(S), 
     not atom(relation(S, E, E', R)).   

% computing TP, FP, FN for each type of relations 

relation_true_pos_count(C, T) :- 
     type_of_relation(T), 
     C = #count{S, E, E'  : relation_true_pos(S, E, E', T)}.
relation_false_pos_count(C, T) :- 
     type_of_relation(T), 
     C = #count{S, E, E' : relation_false_pos(S, E, E', T)}.
relation_false_neg_count(C, T) :- 
     type_of_relation(T), 
     C = #count{S, E, E' : relation_false_neg(S, E, E', T)}.

% relations that do not have both entities identifies are 
% considered as having false declaration 

false_declaration(S, E, E', R) :- 
      atom(relation(S, E, E', R)), 
      1 {not atom(entity(S, E, _)); not atom(entity(S, E', _))}. 

% if a relation occurs in a type specification then 
% it has some type spec that needs to be conformed to 

has_declaration(R) :- 
      type_def(R, _, _). 

% a relation prediction is considered as conforming to the 
% type spec if there is a type specification relating the 
% two types of entities or the relation does not occur in 
% any type specification (i.e., everything is possible)

ok_type(S, E, E', R) :- 
      atom(relation(S, E, E', R)), 
      atom(entity(S, E, T)), 
      atom(entity(S, E', T')),
      1 {type_def(R, T, T'); not has_declaration(R)}.

% if a prediction does not have the right type spec then 
% it is said to have false type 

false_type(S, E, E', R) :- 
      atom(relation(S, E, E', R)), 
      atom(entity(S, E, T)), 
      atom(entity(S, E', T')),
      not ok_type(S, E, E', R).   

% counting false declarations/false type  

false_declaration_count(C, T) :- 
     type_of_relation(T), 
     C = #count{S, E, E' : false_declaration(S, E, E', T)}.
false_type_count(C, T) :- 
     type_of_relation(T), 
     C = #count{S, E, E' : false_type(S, E, E', T)}.

% converting the numbers to element of f1-score 

f1_entity(T, TP, FP, FN):- 
     type_of_entity(T),
     true_pos_count(TP, T),
     false_pos_count(FP, T),  
     false_neg_count(FN, T).  

f1_relation(T, TP, FP, FN):- 
     type_of_relation(T),
     relation_true_pos_count(TP, T),  
     relation_false_pos_count(FP, T),  
     relation_false_neg_count(FN, T). 

% for displaying purpose 

#show.

#show f1_entity/4. 
#show f1_relation/4. 

#show true_pos_count/2.  
#show false_pos_count/2.  
#show false_neg_count/2.  

#show relation_true_pos_count/2.  
#show relation_false_pos_count/2.  
#show relation_false_neg_count/2.  

#show false_declaration_count/2.  
#show false_type_count/2.  

#show true_count_entity/2.
#show true_count_relation/2. 

#show false_declaration/4.  
#show false_type/4.  

#show type_of_entity/1. 
#show type_of_relation/1. 

#show true_pos/3. 
#show false_pos/3. 
#show false_neg/3.

#show in_set/1. 
#show ok_type/4. 

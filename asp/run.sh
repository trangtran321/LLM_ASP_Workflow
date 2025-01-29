# REM $1 [folder %data] $2 [domain file] $3 [folder]

cd $3

cd $1/batch1 

echo  $1 > outtmp 

echo  $1 'with type spec' > outtmpw 

echo $1/batch1 

clingo ../../../compare_2.lp llm_output_asp.txt ground_truth_asp.txt --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2 | head -n1 | tr ' ' '\n' | grep f1 > out

clingo ../../../compare_2.lp llm_output_asp.txt ground_truth_asp.txt $2 --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2 | head -n1 | tr ' ' '\n' | grep f1 > out1

cat outtmp out outtmpw out1 > out2 

cd ..

cd batch2

echo  $1 > outtmp 

echo  $1 'with type spec' > outtmpw 

echo $1/batch2 

clingo ../../../compare_2.lp llm_output_asp.txt ground_truth_asp.txt --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2 | head -n1 | tr ' ' '\n' | grep f1 > out

clingo ../../../compare_2.lp llm_output_asp.txt ground_truth_asp.txt $2 --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2 | head -n1 | tr ' ' '\n' | grep f1 > out1

cat outtmp out outtmpw out1 > out2 

cd .. 

echo $1/batch3 

cd batch3

echo  $1 > outtmp 

echo  $1 'with type spec' > outtmpw 

clingo ../../../compare_2.lp llm_output_asp.txt ground_truth_asp.txt --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2 | head -n1 | tr ' ' '\n' | grep f1 > out

clingo ../../../compare_2.lp llm_output_asp.txt ground_truth_asp.txt $2 --outf=0 -V0 --out-atomf=%s. --quiet=1,2,2 | head -n1 | tr ' ' '\n' | grep f1 > out1

cat outtmp out outtmpw out1 > out2 

cd .. 

cd .. 

cat $1/batch1/out2 $1/batch2/out2 $1/batch3/out2 > out$1

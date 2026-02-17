echo "Starting experiments..."
conda activate kacrice_venv


python script.py fit-real-world --dataset movie_lens --n-movies 10 --n-trials 50 --truncation 8 --verbose --save  
python script.py fit-real-world --dataset news --n-teams 10 --n-trials 50 --truncation 8 --verbose --save  
python script.py fit-real-world --dataset basketball --n-teams 10 --n-trials 50 --truncation 8 --verbose --save 
python script.py fit-real-world --dataset football --n-teams 10 --n-trials 50 --truncation 8 --verbose --save 
python script.py fit-real-world --dataset baseball --n-teams 10 --n-trials 50 --truncation 8 --verbose --save 
python script.py fit-real-world --dataset sushi  --n-teams 10 --n-trials 50 --truncation 8 --verbose --save  


python script.py fit-real-world --dataset movie_lens --n-movies 50 --n-trials 50  --verbose --save  


python script.py fit-real-world --dataset basketball --n-teams 100 --n-trials 50  --verbose 
python script.py fit-real-world --dataset football --n-teams 100 --n-trials 50  --verbose --save 
python script.py fit-real-world --dataset baseball --n-teams 100 --n-trials 50  --verbose  
python script.py fit-real-world --dataset movie_lens --n-movies 100 --n-trials 50  --verbose --save  








#!/bin/bash

# Loop to run source_seeking_main_ES.py with numbers 1 through 11 in parallel
for i in {1..11}
do
   echo "python3 ./source_seeking/source_seeking_main_ES.py $i"; python3 ./source_seeking/source_seeking_main_ES.py $i &
done

# Wait for all background processes to finish
wait

echo "All processes have completed."

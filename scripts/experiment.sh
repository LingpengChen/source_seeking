#!/bin/bash

# Loop from 1 to 10
for i in {28..31}
do
   # Start each python script as a separate background process
   # python3 ./test.py $i &
   python3 ./source_seeking/source_seeking_main_ES.py $i &
   sleep 0.1
done

# Wait for all background processes to complete
wait

echo "All processes have completed."

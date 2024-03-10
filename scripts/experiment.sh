#!/bin/bash

# Loop from 1 to 10
# for i in {0..3}
# for i in {4..7}
# for i in {8..11}
# for i in {12..15}
# for i in {16..19}
# for i in {20..23}
# for i in {24..27}
for i in {28..31}
do
   # Start each python script as a separate background process
   # python3 ./test.py $i &
   python ./source_seeking/combined_main_GMES.py $i &
   # python ./source_seeking_old_version/source_seeking_main_ES.py $i &
   sleep 0.1
done

# Wait for all background processes to complete
wait

echo "All processes have completed."

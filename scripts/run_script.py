import subprocess

# env_index robot_ini_loc_index
for i in range(0,15):
    # subprocess.run(['python3', 'source_seeking/proposed_main.py', str(i), '1'], check=True)
    # subprocess.run(['python3', 'source_seeking/combined_main_GMES.py', str(i), '1'], check=True)
    subprocess.run(['python3', 'source_seeking/GreedyBO_main.py', str(i), '1'], check=True)
    # subprocess.run(['python3', 'source_seeking/DoSS.py', str(i), '1'], check=True)
    # subprocess.run(['python3', 'source_seeking/GMES.py', str(i), '0'], check=True)
    
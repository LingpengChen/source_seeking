import subprocess

import concurrent.futures
import subprocess

def run_script_with_number(number):
    subprocess.run(["python", "./source_seeking/source_seeking_main_ES.py", str(number)])

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_script_with_number, i) for i in range(1, 12)]  # Adjust the range as needed

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')

import glob
import os

scripts = glob.glob('./python-files/*.py')

for script in scripts:
    os.system('python3 ' + script)

os.system('python3 create-test-pipeline.py')

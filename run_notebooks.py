import nbformat
from nbclient import NotebookClient
from pathlib import Path

notebooks = [
    Path('kuairand-like-prediction/notebooks/01_like_eda.ipynb'),
    Path('kuairand-like-prediction/notebooks/02_like_baseline_experiments.ipynb')
]

for nb_path in notebooks:
    print('Executing', nb_path)
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    try:
        client.execute()
        nbformat.write(nb, nb_path)
        print('Executed and saved', nb_path)
    except Exception as e:
        print('Failed to execute', nb_path, e)

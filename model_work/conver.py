import nbformat
from nbconvert import PythonExporter

# Path to the notebook
notebook_path = './image-captioning-on-coco-dataset.ipynb'
# Desired output script path
script_path = './image_captioning_on_coco.py'

# Read the notebook
nb = nbformat.read(notebook_path, as_version=4)

# Export to Python script
exporter = PythonExporter()
script, _ = exporter.from_notebook_node(nb)

# Write the script to file with UTF-8 encoding
with open(script_path, 'w', encoding='utf-8') as f:
    f.write(script)

# Inform the user
print(f"Converted notebook to script at: {script_path}")

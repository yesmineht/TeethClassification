# teeth_classification_2d_gnn

teeth classification in 2d intra-oral images
input : intraoral image + teeth segmentation,
output : teeth labels


##### if poetry is not installed : 
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

##### create virtual envs
cd project-path
poetry shell
poetry install
poetry run pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
poetry run pip install torch-sparse  -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
poetry run pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
poetry run pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
poetry run pip install torch-geometric
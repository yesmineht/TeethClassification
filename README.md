# teeth_classification_2d_gnn

In this project, a method for 2D intraoral teeth classification was developed. 
This method can replace manual labellization.

This work has achieved the goals it has set, that is build a full pipeline 
for teeth labellization  by predicting the class of the teeth.
During this project, We have  built a module for preprocessing the image. 

Then, we have built a module to transform our preprocessed images to graph dataset.
Then, we have built a model to perform the teeth class prediction.after that
Tooth classification stage, the preprocessed octree images are input into two-level hierarchical architecture,feature extracting using resnet architecture and then applying Graph Neural Networks (GNNs) to the task of node classification.

We achieve very good scores for:
• Accuracy: = 0.98%.
• loss = 5%.
#

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

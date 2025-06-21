# MMInfluencer
This is the official repository for our work "MMInfluencer: A Multimodal AI Framework for Influencer Marketing Tasks using Large Language Models".

# How to use
1. Construct knowledge graphs using multimodal data. To construct knowledge graphs, please use the scripts under folder `kg_construct`. For biography data, please run `biogaaphy_for_kg.ipynb`. For multimodal data (images and texts for posts), please run `image_text_for_kg.ipynb`.
2. Training of graph-based deep learning models. For the training of graph-based deep learning models, please utilize related config file and run the following command (here use KGNN as an example):
```
python train_kgnn_gcn_kfold.py --config config/kgnn/kgnn_multimodal_product_category.json
```
3. Inference of graph-based deep learning models. For the inference of graph-based deep learning models, please utilize related config file and run the following command (here use KGNN as an example):
```
python kgnn_gcn_recommendation_inference_kfold.py --config config/kgnn/inference_kfold_kgnn_multimodal_product_category.json
```
4. Standard RAG method. For the implementation of standard RAG method, please run the scripts in `kuzu_graphrag.ipynb`.

# Data
Due to the privacy concern, please prepare your own Instagram data.
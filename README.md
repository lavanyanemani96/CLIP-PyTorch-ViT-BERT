# CLIP-PyTorch (ViT and BERT)

Implementation of CLIP model in PyTorch with pretrained ViT and BERT with flickr30k images dataset.

## Inference:

1. Prompt - Text, Output - image

2. Prompt - Image, Output - image, text

## Reference:
https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2


## How to run FastAPI?
1. Add flickr30k images to folder application/components/prediction/flickr30k_images/
2. Add the best trained model at application/components/prediction/best.pt
3. Run `uvicorn application.server.main:app`

from fastai.vision.all import *
import gradio as gr
import skimage 

learn = load_learner('model.pkl')

categories = ('Normal', 'Cancer')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

title = "Breast Cancer Detection"
description = "A breast cancer detection trained on small dataset, from RSNA Challenge, with fastai. Created as a demo for Gradio and HuggingFace Spaces."

image = gr.inputs.Image()
label = gr.outputs.Label()
examples = ['Cancer.png', 'Normal.png', 'NormalDif.png']
interpretation='default'

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label,title=title,description=description, examples=examples, interpretation=interpretation)
intf.launch(inline=False)

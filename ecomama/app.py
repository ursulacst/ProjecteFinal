import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('model.pkl')

categories = ('CYST', 'FA', 'IDC')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

title = "Ultrasound Tumor Classifier"
description = "An ultrasound tumor classifier trained on a small dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."

image = gr.inputs.Image()
label = gr.outputs.Label()
examples = ['CYST.png', 'FA.png', 'IDC.png']
interpretation='default'

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label,title=title,description=description, examples=examples, interpretation=interpretation)
intf.launch(inline=False)

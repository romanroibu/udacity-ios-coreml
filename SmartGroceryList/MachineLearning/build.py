from keras.models import load_model
import coremltools
import os.path
import wget

model_filename = 'food101-model.hdf5'
coreml_model_filename = 'Food101Net.mlmodel'

if not os.path.exists(model_filename):
	print("(i) Downloading {} model file".format(model_filename))
	url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59ca5da1_food101-model/food101-model.hdf5'
	wget.download(url, out=model_filename)

print("(i) Loading Keras model")
model = load_model(model_filename)

print("(i) Converting Keras model to CoreML model")
coreml_model = coremltools.converters.keras.convert(model,
													input_names=['image'],
													output_names=['confidence'],
													class_labels='labels.txt',
													image_input_names='image',
													image_scale=2./255,
													red_bias=-1,
													green_bias=-1,
													blue_bias=-1)

print("(i) Adding CoreML model metadata")
coreml_model.author = 'Udacity'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Classifies food from an image as one of 101 classes'
coreml_model.input_description['image'] = 'Food image'
coreml_model.output_description['confidence'] = 'Confidence of the food classification'
coreml_model.output_description['classLabel'] = 'Food classification label'

print("(i) Saving CoreML model to {} file".format(coreml_model_filename))
coreml_model.save(coreml_model_filename)

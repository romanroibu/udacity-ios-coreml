from PIL import Image
import coremltools
import os.path

coreml_model_filename = 'Food101Net.mlmodel'

if not os.path.exists(coreml_model_filename):
	print("(!) CoreML model {} not found".format(coreml_model_filename))
	exit(1)

coreml_model = coremltools.models.MLModel(coreml_model_filename)

bibimbap = Image.open('test-images/bibimbap.jpg')

prediction = coreml_model.predict({'image' : bibimbap})

if prediction['classLabel'] != 'bibimbap':
	print("(!) Failed to correctly classify image")
	exit(1)

print("(i) Test successfully passed")
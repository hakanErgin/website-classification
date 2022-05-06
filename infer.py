from infer_functions import load_model, infer_image

# ask user input
model_path = int(input('Model to make inference: \nType (1) for SVM, Type (2) for SGD\n'))
file_path = input('Input file path to infer: \n')

# make inference
inference = infer_image(file_path, model_path)
print('Inferred domain name: ', inference.capitalize())

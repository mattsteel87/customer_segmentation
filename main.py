from segmentation_package.load_data import load_data
from segmentation_package.scaler import scale_data
from segmentation_package.model import train_model
from segmentation_package.output import output_data
import yaml

# Loads the config file
with open("config.yml", "r") as yamlfile:
    settings = yaml.load(yamlfile, Loader=yaml.FullLoader)

clusters = settings['clusters']
output_file = settings['output_path']
data_file = settings['data_path']

# Loads the data from a csv
data = load_data(data_file)

# Uses Standard Scaler to normalise features
scaled_data = scale_data(data)

# Trains the model using KMeans
trained_model = train_model(scaled_data, clusters)

# Combines the output from the model with original data and writes to a CSV
output_data(data, trained_model, output_file)
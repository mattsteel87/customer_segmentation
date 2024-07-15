from segmentation_package.load_data import load_data
from segmentation_package.scaler import scale_data
from segmentation_package.model import train_model
from segmentation_package.output import output_data
import yaml

with open("config.yml", "r") as yamlfile:
    settings = yaml.load(yamlfile, Loader=yaml.FullLoader)

clusters = settings['clusters']
output_file = settings['output_path']
data_file = settings['data_path']

data = load_data(data_file)

scaled_data = scale_data(data)

trained_model = train_model(scaled_data, clusters)

output_data(data, trained_model, output_file)
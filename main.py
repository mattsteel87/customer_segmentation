from segmentation_package.load_data import load_data
from segmentation_package.scaler import scale_data
from segmentation_package.model import train_model
from segmentation_package.output import output_data

data = load_data('data/Mall_Customers.csv')

scaled_data = scale_data(data)

trained_model = train_model(scaled_data)

output_data(data, trained_model)
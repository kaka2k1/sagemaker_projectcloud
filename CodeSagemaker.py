import io
import os
import gzip
import pickle
import numpy as np
import pandas as pd
import urllib.request


import matplotlib.pyplot as plt
% matplotlib inline

import boto3
import sagemaker
import sagemaker.amazon.common as smac
from sagemaker import get_execution_role
from sagemaker.predictor import csv_serializer, json_deserializer

DOWNLOADED_FILENAME = 'mnist.pkl.gz'


if not os.path.exists(DOWNLOADED_FILENAME):
	urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz",DOWNLOADED_FILENAME)

# unzip the downloaded package and use pickle lib to deserialize into python objects
with gzip.open('mnist.pkl.gz', 'rb') as f:
	train_set, valid_set, test_set = pickle.load(f, encoding = 'latin1')

type(train_set)

print(len(train_set))

train_set[0].shape
# --> (50000, 784) = 50000, each image is 28x28 = 784 pixels

train_set[1]


print(valid_set[0].shape)
print(test_set[0].shape)
print(train_set[0].shape)


# convert from a list to a numpy array
vectors = np.array(train_set[0]).astype('float32')
vectors.shape
# --> (50000, 784)

# The Linear Learner is machine classifier
# We want to build an ml model to identify whether the input image that we feed into our model is a 3 or not

lables = np.where(train_set[1] == 3,1,0).astype('float32')
lables.shape

lables[:20]


#training

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, lables)
buf.seek(0)
# name bucket s3 : samplemachinelearning
bucket = 'samplemachinelearning'
prefix = 'sagemaker/linear-minst'

key = 'recordio-pd-data'

resource = boto3.resource('s3')

bucket_obj = resource.Bucket(bucket);

object_res = bucket_obj.Object(os.path.join(prefix,'train', key))

train_data_loc = 's3://{}/{}/train/{}'.format(bucket, prefix, key)

print('Uploaded training data location: {}'.format(train_data_loc))

object_res.upload_fileobj(buf)

output_location  = 's3//{}/{}/output'.format(bucket, prefix)
print('Training artifacts will bes uploaded to: {}'.format(output_location))
region = boto3.Session().region_name
region



#built in algorithms available in containers across different aws regions
# you can access specific version
containers = {'us-west-2':'174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:latest',
	      'us-east-1':'382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:latest',
	      'us-east-2':'404615174143.dkr.ecr.us-east-2.amazonaws.com/linear-learner:latest',
	      'us-west-1':'438346466558.dkr.ecr.us-west-1.amazonaws.com/linear-learner:latest'}



#choose the one closest to us in us-east-1
container = containers[region]
container
#instantiate session to access sagemaker
sess = sagemaker.Session()
role = get_execution_role()
role


# ml.c4.xlarge
linear = sagemaker.estimator.Estimator(container, role, train_instance_count = 1, train_instance_type = 'ml.c4.xlarge',
					output_path = output_location, sagemaker_session = sess)
linear.set_hyperparameters(feature_dim = 784, predictor_type='binary_classifier', mini_batch_size=200)
linear.fit({'train': train_data_loc})

linear_predictor = linear.deploy(initial_instance_count=1, instance_type = 'ml.m4.large')


linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
#result
linear_predictor.deserializer = json_deserializer
result = linear_predictor.predict(train_set[0][30])
print(result)



#run a large set

#split test set into batches of 100 images each
batched_test_set = np.array_split(test_set[0],100)
# total of 10000 images so we split into 10 bathches of 100 image each
print(len(test_set[0]))
print(len(batched_test_set))


predictions = []
# this will call predict for every batch of 100 images and append the prediction lable to every iteration of the array
for array in batched_test_set:
	result = linear_predictor.predict(array)
	predictions += [r['predicted_lable'] for r in result ['predictions']]
result


predictions[:50]


predictions= np.array(predictions)
predictions


actual = np.where(test_set[1] == 3,1,0)
actual


pd.crosstab(actual, predictions, rownames = ['actuals'], colnames = ['predictions'])


linear_predictor.endpoint




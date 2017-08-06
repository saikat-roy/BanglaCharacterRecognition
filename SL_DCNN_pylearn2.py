from pylearn2.models import mlp
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.costs.cost import SumOfCosts 
from pylearn2.costs.mlp import WeightDecay, Default 
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.space import Conv2DSpace
from pylearn2.train import Train
from pylearn2.train_extensions import best_params
from pylearn2.utils import serial
from pylearn2.datasets.hdf5 import HDF5Dataset

import theano
import numpy
import h5py

from pylearn2.monitor import push_monitor

def supervisedLayerwisePRL(trainset, testset):
	'''
	The supervised layerwise training as used in the PRL Paper.
	
	Input
	------
	trainset : A path to an hdf5 file created through h5py.
	testset  : A path to an hdf5 file created through h5py.
	'''
	batch_size = 100
	
	# Both train and test h5py files are expected to have a 'topo_view' and 'y' 
	# datasets side them corresponding to the 'b01c' data format as used in pylearn2
	# and 'y' equivalent to the one hot encoded labels
	trn = HDF5Dataset(filename=trainset, topo_view = 'topo_view', y = 'y', load_all = False)
	tst = HDF5Dataset(filename=testset, topo_view = 'topo_view', y = 'y', load_all = False)
	
	
	'''
	The 1st Convolution and Pooling Layers are added below.
	'''
	h1 = mlp.ConvRectifiedLinear(layer_name='h1', output_channels=64,
			irange=0.05, kernel_shape = [4,4], pool_shape = [4,4],
			pool_stride = [2,2], max_kernel_norm = 1.9365)
			
	fc = mlp.RectifiedLinear(layer_name='fc', dim=1500 , irange=0.05)
	output = mlp.Softmax(layer_name='y', n_classes=171, irange=.005, max_col_norm=1.9365)

	layers = [h1, fc, output]
	
	mdl = mlp.MLP(layers, input_space = Conv2DSpace(shape=(70,70),num_channels=1))

	trainer = sgd.SGD(learning_rate=0.002,
					  batch_size=batch_size,
					  learning_rule=learning_rule.RMSProp(),
					  cost=SumOfCosts(costs = [
					  Default(),
					  WeightDecay(
					    coeffs = [ 0.0005, 0.0005, 0.0005]
					   )
					   ]),
					  train_iteration_mode='shuffled_sequential',
					  monitor_iteration_mode='sequential',
					  termination_criterion=EpochCounter(max_epochs=15),
					  monitoring_dataset={'test': tst, 'valid': vld})


	watcher = best_params.MonitorBasedSaveBest(
		channel_name='valid_y_misclass',
		save_path='./Saved Models/conv_supervised_layerwise_best1.pkl')

	decay = sgd.LinearDecayOverEpoch(start=8,
									 saturate=15,
									 decay_factor=0.1)

	experiment = Train(dataset=trn,
					   model=mdl,
					   algorithm=trainer,
					   extensions=[watcher, decay],
					   )

	experiment.main_loop()
	
	del mdl
	mdl = serial.load('./Saved Models/conv_supervised_layerwise_best1.pkl')
	mdl = push_monitor(mdl,'k')
	
	
	'''
	The 2nd Convolution and Pooling Layers are added below.
	'''
	h2 = mlp.ConvRectifiedLinear(layer_name='h2', output_channels=64,
			irange=0.05, kernel_shape = [4,4], pool_shape = [4,4],
			pool_stride = [2,2], max_kernel_norm = 1.9365)
	
	fc = mlp.RectifiedLinear(layer_name='fc', dim=1500 , irange=0.05)		
	output = mlp.Softmax(layer_name='y', n_classes=171, irange=.005, max_col_norm=1.9365)

	del mdl.layers[-1]
	mdl.layer_names.remove('y')
	del mdl.layers[-1]
	mdl.layer_names.remove('fc')
	mdl.add_layers([h2, fc, output])
	
	trainer = sgd.SGD(learning_rate=0.002,
					  batch_size=batch_size,
					  learning_rule=learning_rule.RMSProp(),
					  cost=SumOfCosts(costs = [
					  Default(),
					  WeightDecay(
					    coeffs = [ 0.0005, 0.0005, 0.0005, 0.0005]
					   )
					   ]),
					  train_iteration_mode='shuffled_sequential',
					  monitor_iteration_mode='sequential',
					  termination_criterion=EpochCounter(max_epochs=15),
					  monitoring_dataset={'test': tst, 'valid': vld})


	watcher = best_params.MonitorBasedSaveBest(
		channel_name='valid_y_misclass',
		save_path='./Saved Models/conv_supervised_layerwise_best2.pkl')

	decay = sgd.LinearDecayOverEpoch(start=8,
									 saturate=15,
									 decay_factor=0.1)

	experiment = Train(dataset=trn,
					   model=mdl,
					   algorithm=trainer,
					   extensions=[watcher, decay],
					   )

	experiment.main_loop()
	
	del mdl
	mdl = serial.load('./Saved Models/conv_supervised_layerwise_best2.pkl')
	mdl = push_monitor(mdl,'l')
	
	
	'''
	The 3rd Convolution and Pooling Layers are added below.
	'''
	h3 = mlp.ConvRectifiedLinear(layer_name='h2', output_channels=64,
			irange=0.05, kernel_shape = [4,4], pool_shape = [4,4],
			pool_stride = [2,2], max_kernel_norm = 1.9365)
	
	fc = mlp.RectifiedLinear(layer_name='h3', dim=1500, irange=0.05)
	output = mlp.Softmax(layer_name='y', n_classes=10, irange=.005, max_col_norm=1.9365)
	
	del mdl.layers[-1]
	mdl.layer_names.remove('y')
	del mdl.layers[-1]
	mdl.layer_names.remove('fc')
	mdl.add_layers([ h3, output])
	
	trainer = sgd.SGD(learning_rate=.002,
					  batch_size=batch_size,
					  learning_rule=learning_rule.RMSProp(),
					  cost=SumOfCosts(costs = [
					  Default(),
					  WeightDecay(
					    coeffs = [ 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]
					   )
					   ]),
					  train_iteration_mode='shuffled_sequential',
					  monitor_iteration_mode='sequential',
					  termination_criterion=EpochCounter(max_epochs=15),
					  monitoring_dataset={'test': tst, 'valid': vld})

	watcher = best_params.MonitorBasedSaveBest(
		channel_name='valid_y_misclass',
		save_path='./Saved Models/conv_supervised_layerwise_best3.pkl')

	decay = sgd.LinearDecayOverEpoch(start=8,
					 saturate=15,
					 decay_factor=0.1)

	experiment = Train(dataset=trn,
					   model=mdl,
					   algorithm=trainer,
					   extensions=[watcher, decay],
					   )

	experiment.main_loop()

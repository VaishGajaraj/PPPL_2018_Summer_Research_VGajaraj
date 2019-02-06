from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pickle
import get_data as gd
import copy

class NubeamNet:
	def __init__(self, dataset,
				 X_scalars=['Shot','ID','ZEFFC','DN0OUT','R0','ELONG','PCURC','AMIN','BZXR','TRIANGU','TRIANGL']+['pinj0' + str(i) for i in np.arange(6)+1],
				 X_profiles=['TE','NE','Q','DIFB'],
				 X_n_keep = [4,4,4,4],
				 pinj_taus = [0.02,0.05,0.1],
				 y_scalars=['Shot','ID','NEUTT','BPSHI','BPLIM','BPCXX','BPCX0'],
				 y_profiles=['PFI','CURB','PBE','PBI','TQBE','TQBI','CURBS','BDENS','ETA_SNC'],
				 y_n_keep = [3,4,4,4,10,10,10,3,3],
				 n_nn = 10,
				 ensemble_exclude_fraction=0.1,
				 hidden_layer_sizes=(10,),
				 alpha=0.000001,
				 learning_rate='adaptive',
				 learning_rate_init=0.001,
				 early_stopping=True):

		self.dataset = dataset
		self.X_prof_n_keep_dict = {key: value for key, value in zip(X_profiles, X_n_keep)}
		self.y_prof_n_keep_dict = {key: value for key, value in zip(y_profiles, y_n_keep)}
		self.X_scalars = X_scalars
		self.y_scalars = y_scalars
		self.X_profiles = X_profiles
		self.y_profiles = y_profiles
		self.X_n_keep = X_n_keep
		self.y_n_keep = y_n_keep
		self.pinj_taus = pinj_taus
		self.X_variables = X_scalars
		self.y_pca = {prof:value for (prof,value) in [(prof, IncrementalPCA(n_components=self.y_prof_n_keep_dict[prof])) for prof in self.y_profiles]}
		self.X_pca = {prof: value for (prof, value) in
					  [(prof, IncrementalPCA(n_components=self.X_prof_n_keep_dict[prof])) for prof in self.X_profiles]}
		self.X_normalization = StandardScaler()
		self.y_normalization = StandardScaler()
		self.n_nn = n_nn
		self.learning_rate = learning_rate
		self.learning_rate_init = learning_rate_init
		self.hidden_layer_sizes = hidden_layer_sizes
		self.alpha = alpha
		self.early_stopping=early_stopping
		self.solver = 'sgd'
		if early_stopping:
			self.solver = 'adam'

		self.regressors = {key:MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha, solver=self.solver,
							   early_stopping=self.early_stopping,warm_start=True,learning_rate=learning_rate,learning_rate_init=learning_rate_init) for key in np.arange(n_nn) }
		for regressor in self.regressors.values():
			regressor.best_loss_ = 1.0e3
		self.excluded_shots = {key:[] for key in np.arange(n_nn)}
		self.scores = {key:[] for key in np.arange(n_nn)}
		self.training_shots = []
		self.ensemble_exclude_fraction = ensemble_exclude_fraction

		# add names for low pass filtered power values
		for x in [x for x in X_scalars if x.startswith('pinj')]:
			for i in np.arange(len(pinj_taus)):
				self.X_variables = self.X_variables + [x+'_lpf_'+str(i+1)]

		for prof in X_profiles:
			self.X_variables = self.X_variables + gd.profile_column_names(prof, self.X_prof_n_keep_dict[prof])

		self.y_variables = y_scalars
		for prof in y_profiles:
			self.y_variables = self.y_variables + gd.profile_column_names(prof, self.y_prof_n_keep_dict[prof])

		self.y_variables_wo_shot = [variable for variable in self.y_variables if variable not in ['Shot', 'ID']]
		self.X_variables_wo_shot = [variable for variable in self.X_variables if variable not in ['Shot', 'ID']]

	def concat_batches(self,batches):
		scalar_dfs = []
		profile_dfs = []
		batchlists = []
		for batch in batches:
			scalar_dfs.append(batch['scalar_df'])
			profile_dfs.append(batch['profile_df'])
			batchlists.append(batch['batchlist'])
		batch['scalar_df'] = pandas.concat(scalar_dfs,axis=0)
		batch['profile_df'] = pandas.concat(profile_dfs,axis=0)
		batch['batchlist'] = pandas.concat(batchlists,axis=0)
		return batch


	def load_batch_data(self, batches_to_load=[('training',1)], trim_last_percent_of_shot=True, percent_to_trim=5,
						   apply_pinj_filtering=True):
		batches = []
		for split, batch_number in batches_to_load:
			hdf5_name =  './{}/{}/batch_{:02d}.h5'.format(self.dataset.batch_output_path, split, batch_number)
			batch = gd.load_batch(hdf5_name)
			if trim_last_percent_of_shot:
				scalar_df_trimmed = self.trim_last_percent_of_shot_fnc(batch['scalar_df'],percent_to_trim)
				profile_df_trimmed = self.trim_last_percent_of_shot_fnc(batch['profile_df'],percent_to_trim)
				batch['scalar_df'] = scalar_df_trimmed.reset_index(drop=True)
				batch['profile_df'] = profile_df_trimmed.reset_index(drop=True)

			if apply_pinj_filtering:
				batch['scalar_df'] = self.add_filtered_pinj_to_df(batch['scalar_df']).reset_index(drop=True)

			batches.append(batch)

		return self.concat_batches(batches)

#todo: save nn only, not data
	def save(self,model_name='model',save_file_path='./'):
		import os
		comment_file_path = save_file_path+model_name+"_comment.txt"
		model_file_path = save_file_path+model_name+'.p'
		directory = os.path.dirname(comment_file_path)
		if not os.path.exists(directory):
			os.makedirs(directory)
		comment_dict = {'Hidden layer sizes': self.hidden_layer_sizes,
								'alpha':self.alpha,
								'ensemble_exclude_fraction':self.ensemble_exclude_fraction,
								'n_nn':self.n_nn,
								'X_variables':self.X_variables_wo_shot}
		with open(comment_file_path, "w") as text_file:
			for key in comment_dict.keys():
			 text_file.write(str(key) + '=' + str(comment_dict[key])+'\n')
		pickle.dump(self,open(model_file_path,"wb"))
		print('Saved NubeamNet model to: ' + model_file_path)

	def trim_last_percent_of_shot_fnc(self, data_df, percent):
		shots = data_df['Shot'].unique()
		for shot in shots:
			tmax = ((100.0 - percent) / 100.0) * max(data_df[data_df['Shot'] == shot]['Time'])
			data_df = data_df[np.logical_or(data_df['Shot'] != shot, data_df['Time'] < tmax)]
		return data_df

	def add_filtered_pinj_to_df(self,data_df):
		lpf_dfs = []
		for x in [x for x in self.X_scalars if x.startswith('pinj')]:
			col = x
			pinj_lpf = np.zeros((len(data_df[col]),len(self.pinj_taus)))
			pinj_lpf[0,:] = data_df.iloc[0][col]
			for i in np.arange(pinj_lpf.shape[0] - 1)+1:
				if (data_df['ID'].iloc[i] == data_df['ID'].iloc[i-1]) and (
					data_df['Shot'].iloc[i] == data_df['Shot'].iloc[i-1]):
					pinj_lpf[i,:] = pinj_lpf[i-1,:] + (data_df['Time'].iloc[i] - data_df['Time'].iloc[i-1]) * (
					-pinj_lpf[i-1] / np.array(self.pinj_taus) + data_df[col].iloc[i] / np.array(self.pinj_taus))

			lpf_dfs.append(pandas.DataFrame(pinj_lpf,columns=[x+'_lpf_'+str(j+1) for j in np.arange(len(self.pinj_taus))]))
		lpf_df = pandas.concat(lpf_dfs,axis=1)
		return pandas.concat([data_df,lpf_df],axis=1)

	def batch_train_pca(self,batch_data,partial_fit=True):
		for prof in self.y_profiles:
			data = batch_data['profile_df'][gd.profile_column_names(prof,20)]
			if partial_fit:
				self.y_pca[prof] = self.y_pca[prof].partial_fit(data)
			else:
				self.y_pca[prof] = self.y_pca[prof].fit(data)
		for prof in self.X_profiles:
			data = batch_data['profile_df'][gd.profile_column_names(prof,20)]
			if partial_fit:
				self.X_pca[prof] = self.X_pca[prof].partial_fit(data)
			else:
				self.X_pca[prof] = self.X_pca[prof].fit(data)

	def plot_pca_components(self,profile):
		if profile in self.y_profiles:
			pca = self.y_pca[profile]
		elif profile in self.X_profiles:
			pca = self.X_pca[profile]
		else:
			print('Invalid profile selection in plot_pca_components')
			return
		plt.plot(pca.components_.T)
		plt.ylabel(profile)

	def get_profile_from_df(self, df, profile, n_prof_x=20):
		return df[gd.profile_column_names(profile, n_prof_x)].values.T

	def pca_projection(self,batch_data,n_prof_x=20):
		pod_coeff_dfs = []
		for profile in self.y_profiles:
			pod_coefficients = self.y_pca[profile].transform(self.get_profile_from_df(batch_data['profile_df'],profile,n_prof_x=n_prof_x).T)
			pod_coeff_dfs.append(pandas.DataFrame(pod_coefficients,columns=gd.profile_column_names(profile,self.y_prof_n_keep_dict[profile])))
		for profile in self.X_profiles:
			pod_coefficients = self.X_pca[profile].transform(
				self.get_profile_from_df(batch_data['profile_df'], profile, n_prof_x=n_prof_x).T)
			pod_coeff_dfs.append(
				pandas.DataFrame(pod_coefficients, columns=gd.profile_column_names(profile, self.X_prof_n_keep_dict[profile])))

		pod_coeff_df = pandas.concat(pod_coeff_dfs,axis=1)
		return pod_coeff_df

	def form_data_df(self,batch_data,n_prof_x=20):
		pod_coeff_df = self.pca_projection(batch_data,n_prof_x=n_prof_x)
		return pandas.concat([batch_data['scalar_df'].reset_index(drop=True),pod_coeff_df.reset_index(drop=True)],axis=1)

	def batch_train_normalization(self,data_df,partial_fit=True):
		X = self.X_from_data(data_df)
		y = self.y_from_data(data_df)
		if partial_fit:
			self.X_normalization.partial_fit(X)
			self.y_normalization.partial_fit(y)
		else:
			self.X_normalization.fit(X)
			self.y_normalization.fit(y)
		if partial_fit: 	print("Partial fit normalization.")

	def train_ensemble(self,data_df,validation_data=None,partial_fit=True):
		if validation_data is not None:
			X_validation = self.X_from_data(validation_data,normalize=True)
			y_validation = self.y_from_data(validation_data,normalize=True)
		new_shots = data_df[~data_df['Shot'].isin(self.training_shots)]['Shot'].unique()
		print(len(new_shots))
		self.training_shots.extend(new_shots)
		for key in self.excluded_shots.keys():
			self.excluded_shots[key].extend(
				new_shots[np.random.binomial(1, self.ensemble_exclude_fraction, len(new_shots)) == 1])
		for i in np.arange(self.n_nn):
			data_df_use = data_df[~data_df['Shot'].isin(self.excluded_shots[i])].reset_index(drop=True)
			X = self.X_from_data(data_df_use,normalize=True)
			y = self.y_from_data(data_df_use,normalize=True)
			if partial_fit:
				self.regressors[i].partial_fit(X, y)
			else:
				self.regressors[i].fit(X,y)
			if validation_data is not None:
				self.scores[i].append(self.regressors[i].score(X_validation,y_validation))

	#todo: update
	def form_inputs_from_run(self,shot,run,connection,tree):
		scalar_variables = [var for var in self.y_scalars+self.X_scalars if var not in ['Shot', 'ID']]
		profile_variables = [var for var in self.y_profiles+self.X_profiles if var not in ['Shot', 'ID']]
		scalar_df, profile_df =gd.get_run_data(shot, run, scalar_variables, profile_variables, connection, tree)
		batch = {'scalar_df': scalar_df, 'profile_df': profile_df, 'attrs': attrs, 'batchlist': 'none'}
		return batch

	def eval_ensemble(self, X):
		''' Evaluate the ensemble of neural networks'''
		predictions = []
		for i in np.arange(self.n_nn):
			reg = self.regressors[i]
			X_scaled = self.X_normalization.transform(X)
			predictions.append(self.y_normalization.inverse_transform(reg.predict(X_scaled)))
		predictions = np.array(predictions)
		avg_prediction = np.mean(predictions, axis=0)
		return avg_prediction, predictions

	def X_from_data(self, data_df,normalize=False):
		X_unscaled = data_df[self.X_variables].drop('Shot',axis=1).drop('ID',axis=1)
		if normalize:
			return self.X_normalization.transform(X_unscaled)
		else:
			return X_unscaled

	def y_from_data(self,data_df,normalize=False):
		y_unscaled = data_df[self.y_variables].drop('Shot', axis=1).drop('ID', axis=1)
		if normalize:
			weights = self.get_fitting_weights()
			return np.dot(self.y_normalization.transform(y_unscaled),np.diag(weights))
		else:
			return y_unscaled

	def get_fitting_weights(self,inverse=False):
		weights = np.ones(len(self.y_variables_wo_shot))
		offset = len(self.y_scalars) - 2
		for j, var in enumerate(self.y_profiles):
			for i in np.arange(self.y_prof_n_keep_dict[var]):
				weights[offset] = 1.0 #/ (i + 1)
				offset += 1
		if inverse:
			weights = 1.0/weights
		return weights

	def y_to_data(self, y, n_prof_x=20):
		df = pandas.DataFrame(y, columns=self.y_variables_wo_shot)
		#todo: automatically handle n_prof_x
		prof_dfs = []
		for name in self.y_profiles:
			recon = self.y_pca[name].inverse_transform(df[gd.profile_column_names(name,self.y_prof_n_keep_dict[name])])
			df2add = pandas.DataFrame(recon)
			df2add.columns = gd.profile_column_names(name, n_prof_x)
			prof_dfs.append(df2add)
		output_df = pandas.concat(prof_dfs, axis=1)
		output_scalars = [y for y in self.y_scalars if y not in ['Shot', 'ID']]
		scaler_df = df[output_scalars]
		output_df = pandas.concat([scaler_df, output_df], axis=1)
		return output_df

	def ensem_to_min_max(self, predictions, n_prof_x=20):
		#todo: automatically handle n_prof_x
		df_test = self.y_to_data(predictions[0, :, :], n_prof_x)
		new_y_variables = df_test.columns
		mat_test = df_test.values
		predictions_proj = np.zeros((predictions.shape[0], mat_test.shape[0], mat_test.shape[1]))

		for i in np.arange(predictions.shape[0]):
			predictions_proj[i, :, :] = self.y_to_data(predictions[i, :, :], n_prof_x).values

		df_max = pandas.DataFrame(np.amax(predictions_proj, axis=0), columns=new_y_variables)
		df_min = pandas.DataFrame(np.amin(predictions_proj, axis=0), columns=new_y_variables)
		df_avg = pandas.DataFrame(np.mean(predictions_proj, axis=0), columns=new_y_variables)
		df_std = pandas.DataFrame(np.std(predictions_proj, axis=0), columns=new_y_variables)
		return df_max, df_min, df_avg, df_std

	def predict(self, X):
		avg_prediction, predictions = self.eval_ensemble(X)
		pred_df_max, pred_df_min, pred_df_avg, pred_df_std = self.ensem_to_min_max(predictions,n_prof_x=20)
		# pred_df_std_tot = pred_df_std_lpf.add(pred_df_std_hpf) #todo: combine in quadrature?
		return {'avg':pred_df_avg, 'min':pred_df_min, 'max':pred_df_max, 'std':pred_df_std}

	def predict_from_run(self, shot, run):
		X = self.form_inputs_from_run(shot,run)
		return self.predict(X)

def load(filename):
	return pickle.load(open(filename))

def get_profile_from_df(df, profile, n_prof_x=20):
	return df[gd.profile_column_names(profile, n_prof_x)].as_matrix().T


def regression_plot_histogram(variable,pred_df_dict,batch_data,x_indices=None,expected_max=None,expected_min=None,colormap='plasma_r',title=None):
	from matplotlib.colors import LogNorm
	if x_indices is not None:
		predictions = []
		expecteds = []
		for j in x_indices:
			predictions.append(get_profile_from_df(pred_df_dict['avg'], variable)[j, :])
			expecteds.append(get_profile_from_df(batch_data['profile_df'], variable)[j, :])

		expected = np.concatenate(expecteds)
		prediction = np.concatenate(predictions)
	else:
		prediction = pred_df_dict['avg'][variable]
		expected = batch_data['scalar_df'][variable]

	if expected_max is not None:
		prediction = prediction[expected<expected_max]
		expected = expected[expected<expected_max]
	if expected_min is not None:
		prediction = prediction[expected > expected_min]
		expected = expected[expected > expected_min]

	fig, ax = plt.subplots()
	plt.set_cmap(colormap)
	h=ax.hist2d(expected,prediction,bins=100,norm=LogNorm())
	# hist, xbins, ybins = np.histogram2d(expected, prediction, bins=100, normed=False)
	# extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
	# plt.figure()
	# im = plt.gca().imshow(np.ma.masked_where(hist < 3, hist).T, interpolation='none', origin='lower', extent=extent,norm=LogNorm())
	# #im = ax[2].imshow(np.ma.masked_where(hist == 0, hist).T, interpolation='none', origin='lower', extent=extent)


	from scipy import stats
	slope, intercept, r_value, p_value, std_err = stats.linregress(expected, prediction)
	print(r_value**2)
	ax.plot(expected,expected,'k--')
	if title is None:
		title = variable
	ax.set_title(title,fontsize=16)
	xmin = min(expected)
	xmax = max(expected)
	cbar = plt.colorbar(h[3],ax=ax)
	cbar.ax.set_title('# of samples',fontsize=14)
	ax.set_xlim((xmin,xmax))
	ax.set_ylim((xmin,xmax))
	ax.set_xlabel('NUBEAM calculation', fontsize=14)
	ax.set_ylabel('NubeamNet prediction', fontsize=14)
	ax.annotate('$R^2=$'+"{:10.3f}".format(r_value**2), xy=(0.05, 0.95), xycoords='axes fraction')

def relative_error_plot(variable,pred_df_dict,batch_data,x_indices=None,colormap='plasma_r',exp_label='y',title=None, cblabel='Counts',uselognorm=True,fit=[0.2,0.1,1],plot_fit=True):
	from matplotlib.colors import LogNorm

	if x_indices is not None:
		predictions = []
		expecteds = []
		for j in x_indices:
			predictions.append(profile_from_df(pred_df_dict['avg'], variable)[j, :])
			expecteds.append(get_profile_from_df(batch_data['profile_df'], variable)[j, :])

		expected = np.concatenate(expecteds)
		prediction = np.concatenate(predictions)
	else:
		prediction = pred_df_dict['avg'][variable]
		expected = batch_data['scalar_df'][variable]

	log_rel_error = np.log10((prediction-expected)/expected)
	log_abs_exp = np.log10(np.abs(expected))
	log_abs_exp_nonan_a = log_abs_exp[np.isfinite(log_rel_error)]
	log_rel_error_nonan_a = log_rel_error[np.isfinite(log_rel_error)]
	log_abs_exp_nonan = log_abs_exp_nonan_a[np.isfinite(log_abs_exp_nonan_a)]
	log_rel_error_nonan = log_rel_error_nonan_a[np.isfinite(log_abs_exp_nonan_a)]

	fig, ax = plt.subplots()
	plt.set_cmap(colormap)
	if uselognorm:
		h=ax.hist2d(log_abs_exp_nonan, log_rel_error_nonan, bins=100, norm=LogNorm())
	else:
		h = ax.hist2d(log_abs_exp_nonan, log_rel_error_nonan, bins=100,cmin=1)
	cbar = plt.colorbar(h[3],ax=ax)
	if cblabel is not None:
		cbar.set_label(cblabel,fontsize=14)
	if title is not None:
		ax.set_title(title,fontsize=16)
	ax.set_xlabel(r'$\log_{10}|{' + exp_label + r'}_{\mathrm{NUBEAM}}|$', fontsize=14)
	ax.set_ylabel(r'$\log_{10}\left|\frac{\mathrm{NubeamNet}-\mathrm{NUBEAM}}{\mathrm{NUBEAM}}\right|$', fontsize=14)

	fit_x_min = min(log_abs_exp_nonan)
	fit_x_max = max(log_abs_exp_nonan)
	fit_x = np.linspace(fit_x_min,fit_x_max,10)

	if plot_fit:
		plt.plot(fit_x,np.log10((fit[0]/(10**(fit_x))+fit[1])*fit[2]),'o')

def estimated_relative_error_plot(variable,pred_df_dict,x_indices=None,colormap='plasma_r',title=None,pred_label='y', cblabel='Counts',uselognorm=True,fit=[0.2,0.1,1],plot_fit=True):
	from matplotlib.colors import LogNorm

	if x_indices is not None:
		predictions = []
		prediction_stds = []
		for j in x_indices:
			predictions.append(get_profile_from_df(pred_df_dict['avg'], variable)[j, :])
			prediction_stds.append(get_profile_from_df(pred_df_dict['std'], variable)[j, :])
		prediction = np.concatenate(predictions)
		prediction_std = np.concatenate(prediction_stds)
	else:
		prediction = pred_df_dict['avg'][variable]
		prediction_std = pred_df_dict['std'][variable]

	log_est_rel_error = np.log10(prediction_std/prediction)
	log_abs_pred = np.log10(np.abs(prediction))
	log_abs_pred_nonan_a = log_abs_pred[np.isfinite(log_est_rel_error)]
	log_est_rel_error_nonan_a = log_est_rel_error[np.isfinite(log_est_rel_error)]
	log_abs_pred_nonan = log_abs_pred_nonan_a[np.isfinite(log_abs_pred_nonan_a)]
	log_est_rel_error_nonan = log_est_rel_error_nonan_a[np.isfinite(log_abs_pred_nonan_a)]

	fig, ax = plt.subplots()
	plt.set_cmap(colormap)
	if uselognorm:
		h=ax.hist2d(log_abs_pred_nonan, log_est_rel_error_nonan, bins=100, norm=LogNorm())
	else:
		h = ax.hist2d(log_abs_pred_nonan, log_est_rel_error_nonan, bins=100,cmin=1)
	cbar = plt.colorbar(h[3],ax=ax)
	if cblabel is not None:
		cbar.set_label(cblabel,fontsize=14)
	if title is not None:
		ax.set_title(title,fontsize=16)
	ax.set_xlabel(r'$\log_{10}\left|' + pred_label + r'_{\mathrm{NubeamNet}}\right|$', fontsize=14)
	ax.set_ylabel(r'$\log_{10}\left|\frac{\sqrt{a_{\mathrm{NubeamNet}}}}{\mathrm{NubeamNet}}\right|$', fontsize=14)

	fit_x_min = min(log_abs_pred_nonan)
	fit_x_max = max(log_abs_pred_nonan)
	fit_x = np.linspace(fit_x_min,fit_x_max,10)

	if plot_fit:
		plt.plot(fit_x, np.log10((fit[0] / (10 ** (fit_x)) + fit[1]) * fit[2]), 'o')

#estimated_relative_error_plot('NEUTT',predictions_training,x_indices=None,title='Neutt')

def error_plot(variable,pred_df,batch_data,x_index=None):

	if x_index is not None:
		prediction = get_profile_from_df(pred_df['avg'],variable)[x_index,:]
		expected = get_profile_from_df(batch_data['profile_df'],variable)[x_index,:]
	else:
		prediction = pred_df['avg'][variable]
		expected = batch_data['scalar_df'][variable]
	error = (prediction-expected)/expected
	error_range = 1.0
	bins = np.linspace(-error_range,error_range,100)
	n, bins_out = np.histogram(error,bins=bins,range=[-error_range,error_range])
	if x_index is not None:
		plt.title(variable + ' j='+str(x_index))
	else:
		plt.title(variable)
	plt.xlabel(r'$\frac{\mathrm{NubeamNet}-\mathrm{NUBEAM}}{\mathrm{NUBEAM}}$')
	plt.ylabel('Normalized counts')
	plt.plot(bins[:-1], n/float(sum(np.logical_and(error>-error_range,error<error_range))))


def slider_plot_profile(prof,pred_df,batch_data,i0=0):
	from matplotlib.widgets import Slider
	prediction = get_profile_from_df(pred_df['avg'],prof)
	expected = get_profile_from_df(batch_data['profile_df'],prof)

	pred_min = get_profile_from_df(pred_df['min'], prof)
	pred_max = get_profile_from_df(pred_df['max'], prof)
	pred_std = get_profile_from_df(pred_df['std'], prof)
	fig, ax = plt.subplots(1)
	plt.subplots_adjust(bottom=0.25)
	plt.fill_between(np.arange(pred_min[:,i0].shape[0]), prediction[:,i0]-pred_std[:,i0], prediction[:,i0]+pred_std[:,i0], alpha=0.5)
	pred_l, = plt.plot(prediction[:,i0],label='NN',linewidth=2)
	expect_l, = plt.plot((expected[:,i0]+expected[:,i0])/2.0,label='NUBEAM',linewidth=2)
	plt.ylabel(prof+' i='+str(i0))
	plt.legend(loc=0)


	axcolor = 'lightgoldenrodyellow'
	axx = plt.axes([0.13, 0.1, 0.77, 0.03], axisbg=axcolor)
	print(pred_min.shape[1])
	sx = Slider(axx, 'i', 0, pred_min.shape[1]-1, valinit=i0, valfmt='%1.0f')
	def update(val):
		i = int(sx.val)
		pred_l.set_ydata(prediction[:,i])
		# targetsx_l.set_ydata(targetsval[x,:])
		expect_l.set_ydata(expected[:,i])
		max_exp = max(expected[:,i])
		# maxtargetsx = max(targetsval[x,:])
		max_pred = max(prediction[:,i])
		maxploty = max(max_pred, max_exp)
		min_exp = min(expected[:,i])
		# maxtargetsx = max(targetsval[x,:])
		min_pred = min(prediction[:,i])
		minploty = min(min_pred, min_exp)

		ax.axis([0, 19, minploty-0.2*abs(minploty), maxploty+0.2*abs(maxploty)])
		for coll in (ax.collections):
			ax.collections.remove(coll)
		ax.fill_between(np.arange(pred_min[:, i].shape[0]), prediction[:,i]-pred_std[:, i], prediction[:,i]+pred_std[:, i], alpha=0.5)
		fig.canvas.draw()

	sx.on_changed(update)
	plt.show()

def slider_plot_profile_at_j(prof, pred_df, batch_data,j):
	from matplotlib.widgets import Slider
	prediction = pod.get_profile_from_df(pred_df['avg'], prof)[j, :]
	expected = pod.get_profile_from_df(batch_data['profile_df'], prof)[j, :]
	win = 100
	pred_min = pod.get_profile_from_df(pred_df['min'], prof)[j, :]
	pred_max = pod.get_profile_from_df(pred_df['max'], prof)[j, :]
	fig, ax = plt.subplots(1)
	plt.subplots_adjust(bottom=0.25)
	i0 = 0
	plt.fill_between(np.arange(i0, i0 + win), pred_min[i0:i0 + win], pred_max[i0:i0 + win], alpha=0.5)
	pred_l, = plt.plot(prediction[i0:i0 + win], label='NN', linewidth=2)
	expect_l, = plt.plot(expected[i0:i0 + win], label='NUBEAM', linewidth=2)
	plt.ylabel(prof)
	plt.xlabel('Index')
	plt.legend(loc=0)

	axcolor = 'lightgoldenrodyellow'
	axx = plt.axes([0.13, 0.1, 0.77, 0.03], axisbg=axcolor)
	sx = Slider(axx, 'i', 0, pred_min.shape[0] - win - 1, valinit=i0, valfmt='%1.0f')

	def update(val):
		i = int(sx.val)
		pred_l.set_ydata(prediction[i:i + win])
		pred_l.set_xdata(np.arange(i, i + win))
		# targetsx_l.set_ydata(targetsval[x,:])
		expect_l.set_ydata(expected[i:i + win])
		expect_l.set_xdata(np.arange(i, i + win))
		max_exp = max(expected[i:i + win])
		max_pred = max(prediction[i:i + win])
		maxplotx = max(max_pred, max_exp)
		min_exp = min(expected[i:i + win])
		min_pred = min(prediction[i:i + win])
		minplotx = min(min_pred, min_exp)
		minplotx = min(minplotx, 0.0)
		ax.set_ylim((minplotx, maxplotx * 1.2))
		ax.set_xlim((i, i + win - 1))
		for coll in (ax.collections):
			ax.collections.remove(coll)
		ax.fill_between(np.arange(i, i + win), pred_min[i:i + win], pred_max[i:i + win], alpha=0.5)
		fig.canvas.draw()

	sx.on_changed(update)
	plt.show()
	
if __name__ == '__main__':
	import pickle
	dataset_path = './datasets/hifi/dataset.p'
	net_save_path = './models/'
	net_save_name = 'model'
 
	print("Loading dataset...")
	ds = pickle.load(open(dataset_path))
	net = NubeamNet(ds, early_stopping=False, learning_rate_init=0.001, hidden_layer_sizes=(50),
					ensemble_exclude_fraction=0.1,n_nn=5)
	print("Loading training batch data...")
	batch_data = net.load_batch_data(batches_to_load=[('training',i+1) for i in range(40)])
	print("Training PCA...")
	net.batch_train_pca(batch_data,partial_fit=False)
	data_df = net.form_data_df(batch_data)
	print("Training normalization...")
	net.batch_train_normalization(data_df,partial_fit=False)
	print("Training neural nets...")
	net.train_ensemble(data_df, validation_data=None,partial_fit=False)
	NubeamNet.__module__ = 'nubeamnet' # this is important to make it depickle properly
	net.save(net_save_name,net_save_path)
	print("Training complete. Predicting and plotting.")
	y_training = net.y_from_data(data_df, normalize=False)
	X_training = net.X_from_data(data_df, normalize=False)
	predictions_training = net.predict(X_training)
	recon_training = net.y_to_data(y_training)
	
	print("Loading validation batch data...")
	validation_batch = net.load_batch_data(batches_to_load=[('validation',1)])
	validation_data_df = net.form_data_df(validation_batch)
	y_validation = net.y_from_data(validation_data_df, normalize=False)
	recon_validation = net.y_to_data(y_validation)
	X_validation = net.X_from_data(validation_data_df, normalize=False)
	print("Evaluating NubeamNet for validation data...")
	predictions = net.predict(X_validation)
	var = 'NEUTT'
	plt.plot(predictions['avg'][var])
	plt.fill_between(np.arange(predictions['avg'][var].shape[0]),
					 predictions['avg'][var] - predictions['std'][var],
					 predictions['avg'][var] + predictions['std'][var], alpha=0.5)
	plt.plot(recon_validation[var])
	plt.xlabel('Sample index')
	plt.ylabel('NEUTT [#/s]')
	plt.show()

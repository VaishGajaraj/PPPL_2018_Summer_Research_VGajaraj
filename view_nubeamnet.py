# -*- coding: utf-8 -*-
# view_nubeamnet.py

import dash
import dash_core_components as dcc
import dash_html_components as html
import flask
import flask_caching
import pandas as pd
import plotly.graph_objs as go

import get_data
from get_data import dataset
import logging
import math
#import MDSplus as mds
import nubeamnet
from nubeamnet import NubeamNet
import numpy as np
import os
import pickle
import threading

import json

class JSONEncoder(json.JSONEncoder):
	def default(self, obj):
		if hasattr(obj, 'to_dict'):
			return {'type':type(obj).__name__, 'data':obj.to_dict()}
		return json.JSONEncoder.default(self, obj)

class JSONDecoder(json.JSONDecoder):
	def __init__(self, *args, **kwargs):
		json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

	def object_hook(self, obj):
		if isinstance(obj, dict) and 'type' in obj and obj['type'] == 'DataFrame':
			return pd.DataFrame.from_dict(obj['data'])
		return obj



PLOT_SERIES = [
	('exp', "NUBEAM",  True,  None,      {'color':'#d62728'}),
	('avg', "NN mean", True,  None,      {'color':'#1f77b4'}),
	('hig', u"NN +1σ", False, None,      {'color':'#1f77b4', 'width':0}),
	('low', u"NN -1σ", False, 'tonexty', {'color':'#1f77b4', 'width':0}),
]


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(threadName)s:%(message)s")
app = dash.Dash(__name__)
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
# app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/brPBPO.css'})
# app.css.append_css({'external_url': '/static/stylesheet.css'})

cache = flask_caching.Cache(app.server, config={
	'CACHE_TYPE':'filesystem', 'CACHE_DIR':'cache'})



def si_format(num):
	""" Convert num to a string with nice SI prefix formatting"""
	if num == 0: 	return u"0.0"
	mag = math.floor(math.log10(abs(num)))
	if mag >= -3 and mag <= 3:
		thous = 0
	else:
		thous = int(math.floor(mag/3))
	letter = u' kMGTPE'[thous] if thous > 0 else u' mμnpfa'[-thous] if thous < 0 else u''
	return u"{:,.3f}{}".format(num/(1000.**thous), letter)


def format_run(shot, run):
	""" Convert shot number and run number, e.g. (204118, 0101), to nice formatted string, e.g. '204118 A01'. """
	return "{:06d} {}{:02d}".format(shot, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[int(run//100)], run%100)


def list_batch_options():
	""" list all of the batches we can access """
	batches = []
	for split in ['training', 'testing', 'validation']:
		for filename in os.listdir(os.path.join('datasets', 'hifi', split)):
			if filename.endswith('.h5'):
				num = filename[-5:-3]
				batches.append({'label':split.capitalize()+" "+str(int(num)), 'value':split+num})
	return batches


def list_model_options():
	""" list all of the neural nets we have """
	return [{'label':name[:-2].capitalize(), 'value':name} for name in os.listdir('./models') if name.endswith('.p')]



app.layout = html.Div(style={'width':'600px', 'margin':'auto', 'padding':20}, children=[
	html.H1("NUBEAM-Net Validation", style={'textAlign':'center'}),

	html.Div("An interface for viewing and comparing NubeamNet results to reality.", style={'textAlign':'center'}),

	html.Div(id='dropdown-row-0', style={'marginBottom':'10px'}, children=[
		html.Div(id='batch-box', style={'width':'250px', 'marginRight':'10px', 'display':'inline-block'}, children=[
			html.Label("Batch:"),
			dcc.Dropdown(id='batches', multi=True, clearable=False, options=list_batch_options()),
		]),

		html.Div(id='model-box', style={'width':'250px', 'marginRight':'10px', 'display':'inline-block'}, children=[
			html.Label("Neural Network:"),
			dcc.Dropdown(id='model', clearable=True, options=list_model_options()),
		]),
	]),

	html.Div(id='hide-until-batches', children=[
		html.Div(id='dropdown-row-1', style={'marginBottom':'10px'}, children=[
			html.Div(id='run-id-box', style={'width':'250px', 'marginRight':'10px', 'display':'inline-block'}, children=[
				html.Label("Run:"),
				dcc.Dropdown(id='run-id', clearable=False, options=[]),
			]),

			html.Div(id='var-tag-box', style={'width':'250px', 'marginRight':'10px', 'display':'inline-block'}, children=[
				html.Label("Tag:"),
				dcc.Dropdown(id='var-tag', clearable=False, options=[]),
			]),
		]),

		html.Div(id='profile-settings', style={'width':'100%'}, children=[
			html.Div(style={'marginBottom':'8px'}, children=[
				html.Label("Constant:", style={'display':'inline-block', 'marginRight':'10px'}),

				dcc.RadioItems(
					id='dimension', style={'display':'inline-block'},
					labelStyle={'display':'inline-block', 'marginRight':'10px'}, value='radius', options=[
						{'label':"Radius", 'value':'radius'},
						{'label':"Time [s]", 'value':'time'},
					]
				),
			]),

			html.Div(id='time0-div', children=[
				dcc.Slider(
					id='time0', min=0, max=1, step=.01, value=0, updatemode='drag',
				),
			]),
			html.Div(id='radius0-div', children=[
				dcc.Slider(
					id='radius0', min=0, max=1, step=1/20., value=0, updatemode='drag',
					marks={int(r) if r%1==0 else r:'{:.1f}'.format(r) for r in np.arange(0, 1.01, .1)},
				),
			]),

			dcc.Graph(id='contour-graph', style={'width':'600px', 'height':'100px', 'marginBottom':'10px'}), # TODO: clicking on the contour should move the slider
		]),

		dcc.Graph(id='main-graph', clear_on_unhover=True, style={'width':'600px', 'marginBottom':'10px'}),
	]),
])


@cache.memoize()
def load_data_threadless(batches, model):
	""" Return the dict of DataFrames. """
	logging.debug("Loading data from disk...")
	if batches is None or len(batches) == 0: # if no batch has yet been chosen
		return None

	try: # Python 2
		nnet = pickle.load(open('./models/nn_3layer_New_FLUXsurface_scan4.p', 'rb')) if model is not None else None # load the network
		dataset = pickle.load(open('./datasets/magdif_PTOWBPPLAS/dataset.p', 'rb')) if nnet is None else nnet.dataset # load the dataset
	except ValueError: # Python 3
		nnet = pickle.load(open('./models/'+model, 'rb'), encoding='latin1') if model is not None else None # load the network
		dataset = pickle.load(open('./datasets/magdif_PTOWBPPLAS/dataset.p', 'rb'), encoding='latin1') if nnet is None else nnet.dataset # load the dataset
	batch_data = get_data.load_batch_data(dataset, net=nnet, batches_to_load=[(btc[:-2], int(btc[-2:])) for btc in batches]) # load the batches
	all_data = nnet.predict(nnet.X_from_data(nnet.form_data_df(batch_data), normalize=False)) if nnet is not None else {} # compute predicted outputs
	all_data['exp'] = pd.concat([batch_data['scalar_df'], batch_data['profile_df'].iloc[:,3:]], axis=1) # extract the true outputs
	if nnet is not None:
		all_data['low'] = all_data['avg'] - all_data['std'] # and compute some +- std stuff
		all_data['hig'] = all_data['avg'] + all_data['std']
	for datum in all_data.values():
		datum.reset_index(inplace=True, drop=True) # the indices are all weird and inconsistent; just replace them
	return all_data


def load_data(batches, model, lock=threading.Lock()):
	""" Return the dict of DataFrames, but wait if someone else is loading it. """
	logging.debug("Acquiring data...")
	lock.acquire()
	data = load_data_threadless(batches, model)
	lock.release()
	return data


@app.callback(
	dash.dependencies.Output('run-id', 'options'),
	[dash.dependencies.Input('batches', 'value'),
	 dash.dependencies.Input('model', 'value')])
def list_shot_run_options(batches, model):
	""" list all of the loaded shot/runs """
	logging.debug("Getting the shots and runs...")
	data_frames = load_data(batches, model)
	if data_frames is None:
		return []
	all_run_ids = data_frames['exp'].loc[:, 'Shot':'ID'].drop_duplicates().sort_values(['Shot','ID']).values.astype(int)
	return [{'label':format_run(shot, run), 'value':str(int(shot)*0x1000+int(run))} for shot, run in all_run_ids]

@app.callback(
	dash.dependencies.Output('var-tag', 'options'),
	[dash.dependencies.Input('batches', 'value'),
	 dash.dependencies.Input('model', 'value')])
def list_variable_options(batches, model):
	""" list all of the loaded variables """
	logging.debug("Getting the available variables...")
	data_frames = load_data(batches, model)
	if data_frames is None:
		return []
	data_frame = data_frames['exp'] if 'avg' not in data_frames else data_frames['avg']
	tags = []
	for col in data_frame.columns:
		if col not in ['Shot', 'ID', 'Time']:
			if col[-2:] == '01' and col[:-2]+'20' in data_frame.columns:
				tags.append((col[:-2], True))
			elif col[-2:].isdigit():
				pass # only add each profile once
			else:
				tags.append((col, False))
	return [{'label':"{}({})".format(tag.capitalize(), "ρ, t" if profile else "t"), 'value':tag+'SP'[profile]} for tag, profile in tags]


@app.callback(
	dash.dependencies.Output('run-id', 'value'),
	[dash.dependencies.Input('run-id', 'options')],
	[dash.dependencies.State('run-id', 'value')])
def assign_default_value(options, value):
	""" Set the default shot/run value """
	if value is None or not options:
		return None
	for option in options:
		if option['value'] == value:
			return value
	return options[0]['value']

@app.callback(
	dash.dependencies.Output('var-tag', 'value'),
	[dash.dependencies.Input('var-tag', 'options')],
	[dash.dependencies.State('var-tag', 'value')])
def assign_default_value(options, value):
	""" Set the default variable tag value """
	if value is None or not options:
		return None
	for option in options:
		if option['value'] == value:
			return value
	return options[0]['value']


@app.callback(
	dash.dependencies.Output('hide-until-batches', 'style'),
	[dash.dependencies.Input('batches', 'value')])
def hide_everything(batches):
	""" Hide everything until some batches have been selected. We can do nothing without a batch. """
	if batches:
		return {'display':'block'}
	else:
		return {'display':'none'}

@app.callback(
	dash.dependencies.Output('profile-settings', 'style'),
	[dash.dependencies.Input('var-tag', 'value')])
def hide_profile_settings(tag):
	""" Hide the profile settings when the selected variable is a scalar (or when no variable is selected). """
	if tag and tag[-1] == 'P':
		return {'display':'block'} # show
	else:
		return {'display':'none'} # hide

@app.callback(
	dash.dependencies.Output('time0-div', 'style'),
	[dash.dependencies.Input('dimension', 'value')])
def hide_time_slider(dim):
	""" Hide the time slider when the selected dimension is constant-radius. """
	if dim == 'time':
		return {'display':'block', 'marginBottom':'30px'} # show
	else:
		return {'display':'none'} # hide

@app.callback(
	dash.dependencies.Output('radius0-div', 'style'),
	[dash.dependencies.Input('dimension', 'value')])
def hide_radius_slider(dim):
	""" Hide the radius sider when the selected dimension is constant-time. """
	if dim == 'time':
		return {'display':'none'} # hide
	else:
		return {'display':'block', 'marginBottom':'30px'} # show


@app.callback(
	dash.dependencies.Output('time0', 'min'),
	[dash.dependencies.Input('run-id', 'value')],
	[dash.dependencies.State('batches', 'value'),
	 dash.dependencies.State('model', 'value')])
def adjust_slider_min(shot_run, batches, model):
	""" set the min of the slider to the min time for this run """
	logging.debug("Adjusting the slider minimum...")
	if shot_run is None: 		return 0
	data_frames = load_data(batches, model)
	if data_frames is None: 	return 0
	shot, run = int(shot_run)//0x1000, int(shot_run)%0x1000
	return data_frames['exp'][(data_frames['exp'].Shot == shot) & (data_frames['exp'].ID == run)].Time.min()

@app.callback(
	dash.dependencies.Output('time0', 'max'),
	[dash.dependencies.Input('run-id', 'value')],
	[dash.dependencies.State('batches', 'value'),
	 dash.dependencies.State('model', 'value')])
def adjust_slider_max(shot_run, batches, model):
	""" set the max of the slider to the max time for this run """
	logging.debug("Adjusting the slider maximum...")
	if shot_run is None: 		return 1
	data_frames = load_data(batches, model)
	if data_frames is None: 	return 1
	shot, run = int(shot_run)//0x1000, int(shot_run)%0x1000
	return data_frames['exp'][(data_frames['exp'].Shot == shot) & (data_frames['exp'].ID == run)].Time.max()

@app.callback(
	dash.dependencies.Output('time0', 'marks'),
	[dash.dependencies.Input('time0', 'min'),
	 dash.dependencies.Input('time0', 'max')])
def adjust_slider_marks(min_time, max_time):
	""" set the slider marks to match the max time """
	logging.debug("Setting the slider tick labels...")
	marker_points = list(np.arange(math.ceil(min_time*10)/10, max_time, .1))
	if marker_points[0]-min_time < (max_time-min_time)*.04: # if the first point is too close to the second one
		marker_points = marker_points[1:] # get rid of it
	if max_time-marker_points[-1] < (max_time-min_time)*.04: # if the last point is too close to the penultimate one
		marker_points = marker_points[:-1] # get rid of it
	inners = {int(t) if t%1==0 else float(t):'{:.1f}'.format(t) for t in marker_points} # wtf are the key requirements for `marks`
	bounds = {int(t) if t%1==0 else float(t):'{:.2f}'.format(t) for t in [min_time, max_time]}
	for t_num, t_str in bounds.items():
		if t_str.endswith('0'): # if the bounds happen to be round
			bounds[t_num] = t_str[:-1] # don't show the last zero

	# return {**inners, **bounds}
	inners.update(bounds)
	return inners


# @cache.memoize()
def generate_contour(data, tag):
	""" this plot gets regenerated far too many times """
	logging.debug("Generating a unique contour plot...")
	times = data['Time'].values
	radii = np.linspace(0, 1, get_data.NUM_GRID_POINTS)
	heights = data.loc[:, get_data.profile_column_names(tag[:-1], get_data.NUM_GRID_POINTS)].transpose().values
	return go.Contour(x=times, y=-radii, z=heights, showscale=False, showlegend=False, hoverinfo='text',
        text=[[
            u"t:  {:.3f}<br>ρ: {:.3f}<br>{}: {}".format(times[j], radii[i], tag[0], si_format(z)) for j,z in enumerate(row)
        ] for i,row in enumerate(heights)],
    ) # I must cast to dict so that it serialises properly

@app.callback(
	dash.dependencies.Output('contour-graph', 'figure'),
	[dash.dependencies.Input('run-id', 'value'),
	 dash.dependencies.Input('var-tag', 'value'),
	 dash.dependencies.Input('dimension', 'value'),
	 dash.dependencies.Input('time0', 'value'),
	 dash.dependencies.Input('radius0', 'value'),
	 dash.dependencies.Input('main-graph', 'hoverData')],
	[dash.dependencies.State('batches', 'value'),
	 dash.dependencies.State('model', 'value')])
def update_contour(shot_run, tag, dimension, t0, r0, hover_data, batches, model):
	""" Plot new contour data when inputs change. """
	logging.debug("Updating the contour plot...")
	if shot_run is None or tag is None or tag[-1] == 'S': # if no shot is selected or this is a scalar variable
		return go.Figure(
			data=[go.Contour(x=[0,1], y=[0,1], z=[[0,0],[0,0]], showscale=False, hoverinfo='none', colorscale='Blues')],
			layout=go.Layout(margin=go.Margin(t=1, b=1, l=1, r=1)), # return a blank graph
		)
	data_frames = load_data(batches, model)
	if data_frames is None:
		return go.Figure()

	shot, run = int(shot_run)//0x1000, int(shot_run)%0x1000
	data = data_frames['exp'][(data_frames['exp']['Shot'] == shot) & (data_frames['exp']['ID'] == run)]
	tmin, tmax = data.Time.iloc[[0,-1]]

	things_to_plot = [generate_contour(data, tag)]
	if dimension == 'time':
		things_to_plot.append(go.Scatter(x=[t0, t0], y=[0, -1], mode='lines', line={'width':1, 'color':'#fff'}, showlegend=False))
		if hover_data is not None:
			things_to_plot.append(go.Scatter(x=[t0], y=[-hover_data['points'][0]['x']], mode='markers', marker={'size':5, 'color':'#fff'}, showlegend=False))
	else:
		things_to_plot.append(go.Scatter(x=[tmin, tmax], y=[-r0, -r0], mode='lines', line={'width':1, 'color':'#fff'}, showlegend=False))
		if hover_data is not None:
			things_to_plot.append(go.Scatter(x=[hover_data['points'][0]['x']], y=[-r0], mode='markers', marker={'size':5, 'color':'#fff'}, showlegend=False))

	return go.Figure(
		data=things_to_plot,
		layout=go.Layout(
			xaxis=go.XAxis(range=[tmin, tmax]),
			yaxis=go.YAxis(range=[-1, 0]),
			margin=go.Margin(t=1, b=1, l=1, r=1),
		),
	)


@app.callback(
	dash.dependencies.Output('main-graph', 'figure'),
	[dash.dependencies.Input('run-id', 'value'),
	 dash.dependencies.Input('var-tag', 'value'),
	 dash.dependencies.Input('dimension', 'value'),
	 dash.dependencies.Input('time0', 'value'),
	 dash.dependencies.Input('radius0', 'value')],
	[dash.dependencies.State('batches', 'value'),
	 dash.dependencies.State('model', 'value'),
	 dash.dependencies.State('main-graph', 'figure')])
def update_main(shot_run, tag, dimension, t0, r0, batches, model, figure):
	""" Plot new slice data when inputs change. """
	logging.debug("Updating the slice graph...")
	if shot_run is None or tag is None:
		return go.Figure()
	data_frames = load_data(batches, model)
	if data_frames is None:
		return go.Figure()

	shot, run = int(shot_run)//0x1000, int(shot_run)%0x1000
	logging.debug("User wants {} for shot {} run {}".format(tag, shot, run))

	data = {key:data_frames[key][(data_frames['exp']['Shot'] == shot) & (data_frames['exp']['ID'] == run)] for key in data_frames} # strip away the nonrelevant rows
	if tag[-1] == 'S': # if it's a scalar variable,
		tag = tag[:-1]
		return go.Figure(
			data=[
				go.Scatter(x=data['exp']['Time'], y=data[key][tag], name=name, showlegend=showlegend, fill=fill, line=line)
				for key, name, showlegend, fill, line in PLOT_SERIES if key in data
			],
			layout=go.Layout(
				title=u"{} vs. Time for Shot #{}, Run {}".format(tag.capitalize(), *format_run(shot,run).split()),
				xaxis=go.XAxis(title="Time [s]"),
				yaxis=go.YAxis(title="{}".format(tag.capitalize())),
			),
		)
	else: # if it's a profile
		tag = tag[:-1]
		times = data['exp']['Time']
		radii = np.linspace(0, 1, 20) # check the other inputs
		for key in data:
			data[key] = data[key].loc[:, get_data.profile_column_names(tag, get_data.NUM_GRID_POINTS)] #strip away the nonrelevant columns
		if dimension == 'time': # constant time, plot against radius
			idx = (times - t0).abs().idxmin()
			return go.Figure(
				data=[
					go.Scatter(x=radii, y=data[key].loc[idx,:], name=name, showlegend=showlegend, fill=fill, line=line)
					for key, name, showlegend, fill, line in PLOT_SERIES if key in data
				],
				layout=go.Layout(
					title=u"{0} vs. Radius for Shot #{2}, Run {3}, t={1:.2f}s".format(tag.capitalize(), t0, *format_run(shot,run).split()),
					xaxis=go.XAxis(title="Normalized flux surface"),
					yaxis=go.YAxis(title="{}".format(tag.capitalize())),
				),
			)
		else: # constant radius, plot against time
			tag = tag + '{:02.0f}'.format(r0*19 + 1)
			return go.Figure(
				data=[
					go.Scatter(x=times, y=data[key].loc[:,tag], name=name, showlegend=showlegend, fill=fill, line=line)
					for key, name, showlegend, fill, line in PLOT_SERIES if key in data
				],
				layout=go.Layout(
					title=u"{0} vs. Time for Shot #{2}, Run {3}, ρ={1}".format(tag.capitalize(), r0, *format_run(shot,run).split()),
					xaxis=go.XAxis(title="Time [s]"),
					yaxis=go.YAxis(title="{}".format(tag.capitalize())),
				),
			)

# @app.server.route('/static/stylesheet.css')
# def serve_stylesheet():
# 	return flask.send_from_directory(os.getcwd(), 'stylesheet.css')



if __name__ == '__main__':
	app.run_server(debug=True)


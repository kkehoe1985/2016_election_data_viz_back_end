from flask import Flask, request
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import json, urllib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

global rf_classifier_model
global X

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Allow-Methods'] = 'DELETE, GET, POST, PUT'
        headers = request.headers.get('Access-Control-Request-Headers')
        if headers:
            response.headers['Access-Control-Allow-Headers'] = headers
    return response
app.after_request(add_cors_headers)

@app.route('/')
def test():
	return 'Everything is running!'

@app.route('/json_test', methods=['GET', 'POST'])
def json_test():
	return 'hi'

@app.route('/createModel')

def createModel():
	from flask import Flask, request
	import pandas as pd 
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import roc_auc_score
	import json, urllib
	from sklearn.ensemble import RandomForestClassifier


	global rf_classifier_model
	global X

	initial_data = pd.DataFrame(pd.read_json('https://raw.githubusercontent.com/kkehoe1985/ga_data_science_final_project/master/initialization_data.json'))
	def unpack(df, column, fillna=None):
	    ret = None
	    if fillna is None:
	        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems()))], axis=1)
	        del ret[column]
	    else:
	        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems())).fillna(fillna)], axis=1)
	        del ret[column]
	    return ret

	initial_data = unpack(initial_data, 'properties', 0)

	drop_columns = ['Democrat', 'Republican', 'state', 'geo_name', 'D', 'R', 'party_winner', 'arcs', 'c', 's', 'r', 'id', 'type']

	X = initial_data.drop(drop_columns, axis=1) 
	y = initial_data['Democrat']

	# predict the response for new observations
	rf_classifier_model = RandomForestClassifier(n_estimators = 2000,
                             oob_score = True,
                             n_jobs = -1,
                             random_state=42,
                             max_features = 0.2,
                             min_samples_leaf = 1)

	rf_classifier_model.fit(X, y)
	return 'model is ready!'



@app.route('/updatePredictions')
def updatePredictions():
	import pandas as pd
	global rf_classifier_model

	initial_data = pd.DataFrame(pd.read_json('https://raw.githubusercontent.com/kkehoe1985/ga_data_science_final_project/master/initialization_data.json'))
	def unpack(df, column, fillna=None):
	    ret = None
	    if fillna is None:
	        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems()))], axis=1)
	        del ret[column]
	    else:
	        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems())).fillna(fillna)], axis=1)
	        del ret[column]
	    return ret

	initial_data = unpack(initial_data, 'properties', 0)

	percent_hs_only = float(request.args.get('percent_hs_only'))
	percent_white_male = float(request.args.get('percent_white_male'))
	percent_some_college = float(request.args.get('percent_some_college'))
	percent_bachelors = float(request.args.get('percent_bachelors'))
	percent_christian_generic = float(request.args.get('percent_christian_generic'))
	percent_jewish = float(request.args.get('percent_jewish'))
	density_pop = float(request.args.get('density_pop'))
	density_housing = float(request.args.get('density_housing'))
	percent_white_female = float(request.args.get('percent_white_female'))
	percent_asian_female = float(request.args.get('percent_asian_female'))

	initial_data["WHITE_MALE_rate"] = initial_data['WHITE_MALE_rate'] * percent_white_male
	initial_data["Percent of adults with a high school diploma only, 2010-2014"] = initial_data["Percent of adults with a high school diploma only, 2010-2014"] * percent_hs_only
	initial_data["Percent of adults completing some college or associate's degree, 2010-2014"] = initial_data["Percent of adults completing some college or associate's degree, 2010-2014"]*percent_some_college
	initial_data["Percent of adults with a bachelor's degree or higher, 2010-2014"] = initial_data["Percent of adults with a bachelor's degree or higher, 2010-2014"]*percent_bachelors
	initial_data["Christian Generic"] = initial_data["Christian Generic"]*percent_christian_generic
	initial_data["Jewish"] = initial_data["Jewish"]*percent_jewish
	initial_data["Density per square mile of land area - Population"] = initial_data["Density per square mile of land area - Population"]*density_pop
	initial_data["Density per square mile of land area - Housing units"] = initial_data["Density per square mile of land area - Housing units"]*density_housing
	initial_data["WHITE_FEMALE_rate"] = initial_data["WHITE_FEMALE_rate"]*percent_white_female
	initial_data["ASIAN_FEMALE_rate"] = initial_data["ASIAN_FEMALE_rate"]*percent_asian_female


	drop_columns = ['Democrat', 'Republican', 'state', 'geo_name', 'D', 'R', 'party_winner', 'arcs', 'c', 's', 'r', 'id', 'type']
	new_X = initial_data.drop(drop_columns, axis=1)
	

	#initial_data_df = pd.DataFrame(initial_data)
	#initial_data_dict = initial_data_df.astype(object).to_dict(orient='records')
	#initial_data_json = json.dumps(initial_data_dict)
	#return initial_data_json

	initial_data['new_prediction'] = rf_classifier_model.predict(new_X) * 100
	output = initial_data[['id', 'new_prediction']]
	initial_data = initial_data.drop('new_prediction', axis=1)
	output_df = pd.DataFrame(output)
	output_dict = output_df.astype(object).to_dict(orient='records')
	output_json = json.dumps(output_dict)
	return output_json

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0')

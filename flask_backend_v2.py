from flask import Flask, request
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import json, urllib
from sklearn.ensemble import RandomForestRegressor
import os

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


@app.route('/create_model')

def create_model():
	# from flask import Flask, request
	# import pandas as pd 
	# from sklearn.ensemble import RandomForestClassifier
	# from sklearn.metrics import roc_auc_score
	# import json, urllib
	# from sklearn.ensemble import RandomForestClassifier


	# global rf_classifier_model
	# global X
    
	# initial_data = pd.read_csv('https://raw.githubusercontent.com/kkehoe1985/ga_data_science_final_project/master/unpacked.csv')
	# X = initial_data.drop('Democrat', axis=1) 
	# y = initial_data['Democrat']

	# # predict the response for new observations
	# rf_classifier_model = RandomForestClassifier(n_estimators = 1000,
 #                             oob_score = True,
 #                             n_jobs = -1,
 #                             random_state = 42,
 #                             max_features = 0.2,
 #                             min_samples_leaf = 1)

	# rf_classifier_model.fit(X, y)
	import cPickle
	global rf_classifier_model

	with open('model.py', 'rb') as f:
		rf_classifier_model = cPickle.load(f)

	return 'model is ready!'



@app.route('/update_predictions')
def update_predictions():
	import pandas as pd

	initial_data = pd.read_csv('https://raw.githubusercontent.com/kkehoe1985/ga_data_science_final_project/master/unpacked.csv')

	percent_hs_only = float(request.args.get('percent_hs_only'))
	population = float(request.args.get('population'))
	percent_white_male = float(request.args.get('percent_white_male'))
	percent_jewish = float(request.args.get('percent_jewish'))
	percent_white_female = float(request.args.get('percent_white_female'))
	percent_bachelors = float(request.args.get('percent_bachelors'))
	density_housing = float(request.args.get('density_housing'))
	percent_black_female = float(request.args.get('percent_black_female'))
	percent_black_male = float(request.args.get('percent_black_male'))
	density_pop = float(request.args.get('density_pop'))

	initial_data["Percent of adults with a high school diploma only, 2010-2014"] = initial_data["Percent of adults with a high school diploma only, 2010-2014"] * percent_hs_only
	initial_data["Population"] = initial_data["Population"] * population
	initial_data["WHITE_MALE_rate"] = initial_data['WHITE_MALE_rate'] * percent_white_male
	initial_data["Jewish"] = initial_data["Jewish"] * percent_jewish
	initial_data["WHITE_FEMALE_rate"] = initial_data["WHITE_FEMALE_rate"] * percent_white_female
	initial_data["Percent of adults with a bachelor's degree or higher, 2010-2014"] = initial_data["Percent of adults with a bachelor's degree or higher, 2010-2014"] * percent_bachelors
	initial_data["Density per square mile of land area - Housing units"] = initial_data["Density per square mile of land area - Housing units"] * density_housing
	initial_data["BLACK_FEMALE_rate"] = initial_data["BLACK_FEMALE_rate"] * percent_black_female
	initial_data["BLACK_MALE_rate"] = initial_data["BLACK_MALE_rate"] * percent_black_male
	initial_data["Density per square mile of land area - Population"] = initial_data["Density per square mile of land area - Population"] * density_pop

	drop_columns = ['Democrat']
	new_X = initial_data.drop(drop_columns, axis=1)

	initial_data['new_prediction'] = rf_classifier_model.predict(new_X) * 100
	output = initial_data[['id', 'new_prediction']]
	initial_data = initial_data.drop('new_prediction', axis=1)
	output_df = pd.DataFrame(output)
	output_dict = output_df.astype(object).to_dict(orient='records')
	output_json = json.dumps(output_dict)
	return output_json

if __name__ == '__main__':
    # app.run(debug=True)
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

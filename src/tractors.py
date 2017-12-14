from ds_utils import Predictor
import pandas as pd
import numpy as np

"""TODO:
    -fix data_to_predict processing
"""


class Tractors(Predictor):

    def processing(self, data):
        data['saledate'] = pd.to_datetime(
            data['saledate'], format='%m/%d/%Y %H:%M')
        data['sale_age'] = data['saledate'].dt.year - \
            data['YearMade']
        data.loc[data['YearMade'] == 1000, 'YearMade'] = 2000
        data.loc[data['sale_age'] > 500, 'sale_age'] = np.nan
        forest_params = {'n_estimators': 10, 'n_jobs': 4}
        if str(self.test) != 'None':
            self.test['saledate'] = pd.to_datetime(
                self.test['saledate'], format='%m/%d/%Y %H:%M')
            self.test['sale_age'] = self.test['saledate'].dt.year - \
                self.test['YearMade']
            self.test.loc[self.test['YearMade'] == 1000, 'YearMade'] = 2000
        data.loc[data['sale_age'] > 500, 'sale_age'] = np.nan
        data = self.predict_missing_values(
            data, ['sale_age'], ['MachineID', 'ModelID', 'YearMade'])
        data = self.predict_missing_values(data, ['MachineHoursCurrentMeter'], [
                                           'MachineID', 'ModelID', 'YearMade', 'sale_age'])
        dummies = [u'auctioneerID', u'YearMade',
                   u'ProductSize',
                   u'state', u'ProductGroup',
                   u'Drive_System', u'Enclosure', u'Forks', u'Pad_Type', u'Ride_Control',
                   u'Stick', u'Transmission', u'Turbocharged',
                   u'Enclosure_Type', u'Engine_Horsepower', u'Hydraulics',
                   u'Pushblock', u'Ripper', u'Scarifier', u'Tip_Control', u'Tire_Size',
                   u'Coupler', u'Coupler_System', u'Grouser_Tracks', u'Hydraulics_Flow',
                   u'Track_Type',  u'Stick_Length', u'Thumb',
                   u'Pattern_Changer', u'Grouser_Type', u'Backhoe_Mounting', u'Blade_Type',
                   u'Travel_Controls', u'Differential_Type', u'Steering_Controls', u'fiModelSeries']
        data = self.dummify(data, dummies)
        for column in data.columns:
            if 'None' in column:
                data.drop(column, axis=1, inplace=True)
        if str(self.test) != 'None':
            self.test = self.dummify(self.test, dummies)
            for column in self.test.columns:
                if column not in data.columns:
                    self.test.drop(column, axis=1, inplace=True)
            for column in data.columns:
                if column not in self.test.columns:
                    self.test.loc[:, column] = 0
        return data


if __name__ == '__main__':
    # load the training data and data to predict
    print 'loading data'
    train = pd.read_csv('../data/train.csv', low_memory=False)
    data_to_predict = pd.read_csv('../data/test.csv', low_memory=False)

    # initialize Tractors instance
    tractors = Tractors(
        data=train, data_to_predict=data_to_predict, target='SalePrice')

    # process the training data and data to predict
    print 'processing data'
    tractors.data = tractors.processing(tractors.data)
    tractors.data_to_predict = tractors.processing(tractors.data_to_predict)

    # train test split
    print 'splitting data'
    tractors.split(test_size=0.8, random_state=42)

    # get a baseline to compare our results with
    tractors.mean_baseline()

    # set features to train on
    drop_features = ['SalesID', 'fiBaseModel', 'SalePrice', 'MachineID', 'datasource',
                     'saledate', 'fiModelDesc', 'fiModelDescriptor', 'fiProductClassDesc', 'ProductGroupDesc', 'Undercarriage_Pad_Width', 'Blade_Width', 'Blade_Extension', 'ModelID',  u'fiSecondaryDesc', 'UsageBand']
    tractors.set_features(tractors.train.drop(drop_features, axis=1).columns)

    # perform features selection with lassoCV
    print 'selecting features'
    tractors.select_features()

    # grid search model hyperparameter
    print 'tuning hyperparameters'
    model_name = 'RandomForestRegressor'
    param_grid = {'n_estimators': [10],
                  'max_features': ['auto', 0.6, 0.2],
                  'max_depth': [None],
                  'min_samples_leaf': [2],
                  'bootstrap': [True],
                  'n_jobs': [-1],
                  'verbose': [0]}
    tractors.grid_search(model_name, param_grid)
    print 'best train RMSE: {}'.format(np.sqrt(abs(tractors.train_score_)))

    # score the best model
    print 'testing the performance on unseen data'
    tractors.score(model_name, **tractors.best_params_)
    print 'test RMSE: {}'.format(tractors.test_score_)

    # increase the number of estimators for the final fit
    tractors.best_params_['n_estimators'] = 100

    # fit the best model on the whole dataset
    print 'fitting final model'
    tractors.fit(model_name, **tractors.best_params_)
    tractors.best_params_

    # pickle the model
    print 'pickling model'
    tractors.pickle_model('RandomForestRegressor.pkl')

    # predict new data
    print 'making predictions'
    tractors.predict()
    tractors.data_to_predict['Predicted_SalePrice'] = tractors.predictions
    print tractors.predictions
    tractors.data.shape
    tractors.train.head()
    tractors.data_to_predict[].head()

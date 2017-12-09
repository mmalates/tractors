from ds_utils import Predictor
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


class Tractors(Predictor):

    def processing(self, data):

        # make sale_age category that is difference b/t the year it is sold and the year it is made
        data['saledate'] = pd.to_datetime(
            data['saledate'], format='%m/%d/%Y %H:%M')
        data['sale_age'] = data['saledate'].dt.year - \
            data['YearMade']

        data.loc[data['sale_age'] > 500, 'sale_age'] = np.nan
        forest_params = {'n_estimators': 10, 'n_jobs': 4}
        # make sale_age category that is difference b/t the year it is sold and the year it is made
        if str(self.test) != 'None':
            self.test['saledate'] = pd.to_datetime(
                self.test['saledate'], format='%m/%d/%Y %H:%M')
            self.test['sale_age'] = self.test['saledate'].dt.year - \
                self.test['YearMade']

        data.loc[data['sale_age'] > 500, 'sale_age'] = np.nan
        self.predict_missing_values('RandomForestRegressor', ['sale_age'], [
                                    'MachineID', 'ModelID', 'YearMade'], **forest_params)
        self.predict_missing_values('RandomForestRegressor', ['MachineHoursCurrentMeter'], [
                                    'MachineID', 'ModelID', 'YearMade', 'sale_age'], **forest_params)
        self.predict_missing_values('RandomForestClassifier', ['UsageBand'], [
                                    'MachineID', 'ModelID', 'YearMade', 'sale_age'], **forest_params)


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


tractors = Tractors(data=train)
dummies = ['ProductGroup', 'UsageBand']
# tractors.load_data('../data/train.csv')
tractors.split(random_state=123)

tractors.processing(tractors.train)

# tractors.split(random_state=123)

count_mask = tractors.train.drop('SalePrice', axis=1)._get_numeric_data().count() == (
    tractors.train.drop('SalePrice', axis=1)._get_numeric_data().count().max())
train_features = tractors.train.drop(
    'SalePrice', axis=1)._get_numeric_data().columns[count_mask]

forest_params = {'n_estimators': 10, 'n_jobs': 4}
tractors.fit('RandomForestRegressor', 'SalePrice',
             train_features, **forest_params)
#
# for item in tractors.fill_models.keys():
#     print tractors.fill_models[item].feature_importances_

# tractors.test_processing(tractors.test)


tractors.predict(tractors.test[train_features])


tractors.score_

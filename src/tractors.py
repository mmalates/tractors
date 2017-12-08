from ds_utils import Predictor
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


class Tractors(Predictor):

    def processing(self, data):

        # make sale_age category that is difference b/t the year it is sold and the year it is made
        self.data['saledate'] = pd.to_datetime(
            self.data['saledate'], format='%m/%d/%Y %H:%M')
        self.data['sale_age'] = self.data['saledate'].dt.year - \
            self.data['YearMade']

        self.data.loc[self.data['sale_age'] > 500, 'sale_age'] = np.nan
        forest_params = {'n_estimators': 10, 'n_jobs': 4}
        self.predict_missing_values('RandomForestRegressor', ['sale_age'], [
                                    'MachineID', 'ModelID', 'YearMade'], **forest_params)
        self.predict_missing_values('RandomForestRegressor', ['MachineHoursCurrentMeter'], [
                                    'MachineID', 'ModelID', 'YearMade', 'sale_age'], **forest_params)
        self.predict_missing_values('RandomForestClassifier', ['UsageBand'], [
                                    'MachineID', 'ModelID', 'YearMade', 'sale_age'], **forest_params)

    def test_processing():


tractors = Tractors()
dummies = ['ProductGroup', 'UsageBand']
tractors.load_data('../data/train.csv')
tractors.processing()
count_mask = tractors.data._get_numeric_data().count() == (
    tractors.data._get_numeric_data().count().max())
train_features = tractors.data._get_numeric_data().columns[count_mask]

forest_params = {'n_estimators': 10, 'n_jobs': 4}
tractors.fit('RandomForestRegressor', 'SalePrice',
             train_features, **forest_params)

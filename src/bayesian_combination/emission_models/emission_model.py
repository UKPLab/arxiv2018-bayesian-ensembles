'''
Abstract class for the emission models.
'''

class Emission():

    def init_features(self, features):
        pass


    def update_features(self):
        pass


    def predict_features(self, features):
        pass


    def feature_ll(self, features):
        pass


    def lowerbound(self):
        pass
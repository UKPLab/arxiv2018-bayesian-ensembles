class Tagger:

    def fit_predict(self, Et):
        '''
        Fit the model to the current expected true labels.
        :param Et:
        :return:
        '''
        pass


    def predict(self, doc_start, features):
        '''
        Use the trained model to predict.
        :param doc_start:
        :param features:
        :return:
        '''


    def log_likelihood(self, C_data, E_t):
        '''
        Compute the log-likelihood of the model's current predictions for the training set.
        :param C_data:
        :param E_t:
        :return:
        '''
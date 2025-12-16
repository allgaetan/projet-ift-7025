from sklearn.model_selection import GridSearchCV
from nnFeatureExtractors import MLPRegressorExtractor
from nnQLearningAgents import MLPRegressorQAgent

class MLPRegressorGridSearch():
    def __init__(self, parametersRanges, fixedParameters=dict()):
        self.gridSearch = None
        self.parametersRanges = parametersRanges
        self.fixedParameters = fixedParameters
        self.bestParameters = self.fixedParameters

    def evaluateParameters(self):
        for param in self.parametersRanges:
            range = self.parametersRanges[param]
            bestValue = range[0]
            self.bestParameters[param] = bestValue

    def getBestParameters(self):
        self.evaluateParameters()
        return self.bestParameters
    
parametersRanges = dict({
    "hidden_layer_sizes": [(32,), (64,), (32, 32), (64, 64)],
    "activation": ["relu", "logistic", "tanh"],
    "solver": ["adam", "sgd"],
    "learning_rate_init": [0.0001, 0.001, 0.01]
})
    
fixedParameters = dict(
    warm_start=True,
    max_iter=1
)
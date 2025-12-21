import pandas as pd
import os
import argparse
import textDisplay
import layout

from nnQLearningAgents import MLPRegressorQAgent
from pacman import runGames
from ghostAgents import RandomGhost, DirectionalGhost

def train(agent, numTrain, trainGridName, trainGhostType):
    """
    Train the agent on the specified grid and ghost type:
    - agent: the MLPRegressorQAgent to be trained
    - numTrain: number of training games
    - trainGridName: name of the layout to be used for training
    - trainGhostType: type of ghost agent to be used for training
    """
    print("Training agent for {} games on layout {} with ghost type {}"
          .format(numTrain, trainGridName, trainGhostType.__name__))
    agent.numTraining = numTrain
    runGames(
        layout=layout.getLayout(trainGridName),
        pacman=agent,
        ghosts=[trainGhostType(1)],
        display=textDisplay.NullGraphics(),
        numGames=numTrain,
        record=False,
        numTraining=numTrain,
        timeout=1
    )

def test(agent, numTest, testGridName, testGhostType):
    """
    Test the agent on the specified grid and ghost type:
    - agent: the MLPRegressorQAgent to be tested
    - numTest: number of testing games
    - testGridName: name of the layout to be used for testing
    - testGhostType: type of ghost agent to be used for testing
    """
    print("Testing agent for {} games on layout {} with ghost type {}"
          .format(numTest, testGridName, testGhostType.__name__))
    _, logs = runGames(
        layout=layout.getLayout(testGridName),
        pacman=agent,
        ghosts=[testGhostType(1)],
        display=textDisplay.NullGraphics(),
        numGames=numTest,
        record=False,
        timeout=1
    )
    if "Win Rate" in logs:
        return logs["Win Rate"]
    return 0

def evaluateAgent(agent, numTrain, trainGridName, trainGhostType,
                  numTest, testGridName, testGhostType):
    """
    Evaluate the agent by training and testing it:
    - agent: the MLPRegressorQAgent to be evaluated
    - numTrain: number of training games
    - trainGridName: name of the layout to be used for training
    - trainGhostType: type of ghost agent to be used for training
    - numTest: number of testing games
    - testGridName: name of the layout to be used for testing
    - testGhostType: type of ghost agent to be used for testing
    """
    print("Evaluating agent...")
    train(agent, numTrain, trainGridName, trainGhostType)
    score = test(agent, numTest, testGridName, testGhostType)
    return score

def generateGridSearchConfigs(baseConfig, paramRanges):
    """
    Generate all combinations of hyperparameter configurations for grid search
    - baseConfig: dictionary with default hyperparameter values
    - paramRanges: dictionary with lists of values for each hyperparameter to explore
    """
    nbParams = len(paramRanges)
    paramNames = list(paramRanges.keys())
    nbValues = [len(paramRanges[name]) for name in paramNames]
    nbConfigs = 1
    for n in nbValues:
        nbConfigs *= n
    print("Generating {} hyperparameter configurations for grid search...".format(nbConfigs))
    configs = []
    for i in range(nbConfigs):
        config = baseConfig.copy()
        index = i
        for j in range(nbParams):
            name = paramNames[j]
            values = paramRanges[name]
            valueIndex = index % nbValues[j]
            config[name] = values[valueIndex]
            index //= nbValues[j]
        configs.append(config)
    return configs

def hyperparametersEvaluations(logging=False):
    """
    Evaluate multiple hyperparameters configurations for the MLPRegressor 
    """
    BASE = {
        "hidden_layer_sizes": (64,),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": 0.0001,
        "warm_start": True
    }
    RANGES = {
        "hidden_layer_sizes": [(64,), (32,), (64, 64)],
        "activation": ["relu", "logistic"],
        "solver": ["adam", "sgd"],
        "learning_rate_init": [0.0001, 0.001]
    }
    configs = generateGridSearchConfigs(BASE, RANGES)
    scores = {}
    for i, hyperparameters in enumerate(configs):
        print("Evaluating configuration {}/{} with hyperparameters: {}"
            .format(i+1, len(configs), hyperparameters))
        agent = MLPRegressorQAgent(params=hyperparameters)
        score = evaluateAgent(
            agent=agent,
            numTrain=NUM_TRAIN,
            trainGridName=SMALL_GRID_NAME,
            trainGhostType=RandomGhost,
            numTest=NUM_TEST,
            testGridName=SMALL_GRID_NAME,
            testGhostType=RandomGhost
        )
        scores[str(hyperparameters)] = score
        if logging:
            with open(LOGGER_PATH, "a") as f:
                f.write("Logs from function hyperparametersEvaluations\n")
                f.write("Configuration {}/{}\n".format(i+1, len(configs)))
                f.write("Hyperparameters: {}\n".format(hyperparameters))
                f.write("Score: {}\n\n".format(score))
    df = pd.DataFrame.from_dict(scores, orient="index", columns=["Score"])
    if logging:
        with open(LOGGER_PATH, "a") as f:
            f.write("Logs from function hyperparametersEvaluations\n")
            f.write("Results: {}\n\n".format(df))
    return df

def standardEvaluations(logging=False):
    """
    Evaluations with best hyperparameters:
    - Evaluation on smallGrid with RandomGhost
    - Evaluation on mediumGrid with RandomGhost
    - Evaluation on smallGrid with DirectionalGhost
    - Evaluation on mediumGrid with DirectionalGhost
    """
    grids = [SMALL_GRID_NAME, MEDIUM_GRID_NAME]
    ghosts = [RandomGhost, DirectionalGhost]
    scores = {}
    for grid in grids:
        for ghost in ghosts:
            print("Standard evaluation on grid {} with ghost type {}".format(grid, ghost.__name__))
            agent = MLPRegressorQAgent(params=BEST_PARAMS)
            score = evaluateAgent(
                agent=agent,
                numTrain=NUM_TRAIN,
                trainGridName=grid,
                trainGhostType=ghost,
                numTest=NUM_TEST,
                testGridName=grid,
                testGhostType=ghost
            )
            scores["{}_{}".format(grid, ghost.__name__)] = score
            if logging:
                with open(LOGGER_PATH, "a") as f:
                    f.write("Logs from function standardEvaluations\n")
                    f.write("Grid: {}\n".format(grid))
                    f.write("Ghost Type: {}\n".format(ghost.__name__))
                    f.write("Score: {}\n\n".format(score))
        print("Standard evaluation on grid {}".format(grid))
    df = pd.DataFrame.from_dict(scores, orient="index", columns=["Score"])
    if logging:
        with open(LOGGER_PATH, "a") as f:
            f.write("Logs from function standardEvaluations\n")
            f.write("Results: {}\n\n".format(df))
    return df

def generalizationEvaluations(logging=False):
    """
    Generalization capacity:
    - Train on smallGrid with RandomGhost, test on mediumGrid with RandomGhost
    - Train on mediumGrid with RandomGhost, test on smallGrid with RandomGhost
    - Train on smallGrid with RandomGhost, test on smallGrid with DirectionalGhost
    - Train on mediumGrid with RandomGhost, test on mediumGrid with DirectionalGhost
    """
    testCases = {
        "SmallToMedium": {"trainGrid": SMALL_GRID_NAME, "trainGhost": RandomGhost, 
                          "testGrid": MEDIUM_GRID_NAME, "testGhost": RandomGhost},
        "MediumToSmall": {"trainGrid": MEDIUM_GRID_NAME, "trainGhost": RandomGhost,
                          "testGrid": SMALL_GRID_NAME, "testGhost": RandomGhost},
        "RandomToDirectionalSmall": {"trainGrid": SMALL_GRID_NAME, "trainGhost": RandomGhost,
                                    "testGrid": SMALL_GRID_NAME, "testGhost": DirectionalGhost},
        "RandomToDirectionalMedium": {"trainGrid": MEDIUM_GRID_NAME, "trainGhost": RandomGhost,
                                "testGrid": MEDIUM_GRID_NAME, "testGhost": DirectionalGhost}
    }
    scores = {}
    for caseName, case in testCases.items():
        print("Generalization evaluation for {}: train on grid {} with ghost type {}, test on grid {} with ghost type {}"
              .format(caseName, case["trainGrid"], case["trainGhost"].__name__,
                      case["testGrid"], case["testGhost"].__name__))
        agent = MLPRegressorQAgent(params=BEST_PARAMS)
        score = evaluateAgent(
            agent=agent,
            numTrain=NUM_TRAIN,
            trainGridName=case["trainGrid"],
            trainGhostType=case["trainGhost"],
            numTest=NUM_TEST,
            testGridName=case["testGrid"],
            testGhostType=case["testGhost"]
        )
        scores[caseName] = score
        if logging:
            with open(LOGGER_PATH, "a") as f:
                f.write("Logs from function generalizationEvaluations\n")
                f.write("Test Case: {}\n".format(caseName))
                f.write("Train Grid: {}\n".format(case["trainGrid"]))
                f.write("Train Ghost Type: {}\n".format(case["trainGhost"].__name__))
                f.write("Test Grid: {}\n".format(case["testGrid"]))
                f.write("Test Ghost Type: {}\n".format(case["testGhost"].__name__))
                f.write("Score: {}\n\n".format(score))
    df = pd.DataFrame.from_dict(scores, orient="index", columns=["Score"])
    if logging:
        with open(LOGGER_PATH, "a") as f:
            f.write("Logs from function generalizationEvaluations\n")
            f.write("Results: {}\n\n".format(df))
    return df

def regularizationEvaluations(logging=False):
    """
    Impact of the L2 regularization parameter
    """
    alphas = [0.0, 0.0001, 0.001, 0.01, 0.1]
    results = {}
    for alpha in alphas:
        print("Evaluating L2 regularization with alpha={}".format(alpha))
        BEST_PARAMS["alpha"] = alpha
        df = generalizationEvaluations(logging)
        results[alpha] = df
        if logging:
            with open(LOGGER_PATH, "a") as f:
                f.write("Logs from function regularizationsEvaluations\n")
                f.write("L2 Regularization Alpha: {}\n".format(alpha))
                f.write("Results: {}\n\n".format(df))
    return results

def evaluation(logging):
    if PERFORM_HYPERPARAMETERS_EVALUATIONS:
        print("Performing hyperparameters evaluations...")
        df = hyperparametersEvaluations(logging)
        print("Hyperparameters evaluation results:")
        print(df)

    if PERFORM_STANDARD_EVALUATIONS:
        print("Performing standard evaluations...")
        df = standardEvaluations(logging)
        print("Standard evaluation results:")
        print(df)

    if PERFORM_GENERALIZATION_EVALUATIONS:
        print("Performing generalization evaluations...")
        df = generalizationEvaluations(logging)
        print("Generalization evaluation results:")
        print(df)

    if PERFORM_REGULARIZATION_EVALUATIONS:
        print("Performing regularization evaluations...")
        df = regularizationEvaluations(logging)
        print("Regularization evaluation results:")
        print(df)

def argParse():
    parser = argparse.ArgumentParser(description="Evaluation methods for MLPRegressorQAgent")
    parser.add_argument("-n", "--num_train", type=int, default=2000,
                        help="Number of training games")
    parser.add_argument("-m", "--num_test", type=int, default=100,
                        help="Number of testing games")
    parser.add_argument("-p", "--perform-hyperparams-evaluations", action="store_true", default=False,
                        help="Perform hyperparameters evaluations")
    parser.add_argument("-s", "--perform-standard-evaluations", action="store_true", default=False,
                        help="Perform standard evaluations")
    parser.add_argument("-g", "--perform-generalization-evaluations", action="store_true", default=False,
                        help="Perform generalization evaluations")
    parser.add_argument("-r", "--perform-regularization-evaluations", action="store_true", default=False,
                        help="Perform regularization evaluations")
    parser.add_argument("-l", "--logging", action="store_true", default=False,
                        help="Enable logging")
    return parser.parse_args()

if __name__ == "__main__":
    args = argParse()
    
    NUM_TRAIN = args.num_train
    NUM_TEST = args.num_test
    PERFORM_HYPERPARAMETERS_EVALUATIONS = args.perform_hyperparams_evaluations
    PERFORM_STANDARD_EVALUATIONS = args.perform_standard_evaluations
    PERFORM_GENERALIZATION_EVALUATIONS = args.perform_generalization_evaluations
    PERFORM_REGULARIZATION_EVALUATIONS = args.perform_regularization_evaluations
    LOGGING = args.logging
    SMALL_GRID_NAME = "smallGrid"
    MEDIUM_GRID_NAME = "mediumGrid"
    LOGGER_PATH = os.path.join(os.path.dirname(__file__), "evalLogs.txt")
    BEST_PARAMS = {
        'hidden_layer_sizes': (64, 64), 
        'activation': 'logistic', 
        'solver': 'sgd', 
        'learning_rate_init': 0.001, 
        'warm_start': True
    }

    evaluation(LOGGING)
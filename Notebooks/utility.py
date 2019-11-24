import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from collections import defaultdict


# Displays confusion matrix
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list, cmap=plt.cm.Blues) -> None:    
    cm = confusion_matrix(y_true, y_pred)    
    cm = pd.DataFrame(cm, index = classes, columns = classes)
    
    ax = sns.heatmap(cm, cmap = plt.cm.Blues, annot = True)
    ax.set(xlabel = 'Predicted', ylabel = 'Actual')

    
# Prints the cross validated scores
def cross_validate_scores(clf, X: np.ndarray, y: np.ndarray, cv: int = 3, metrics: list =  ['accuracy']) -> None:
    scores_raw = cross_validate(clf, X, y,
                                scoring = metrics,
                                n_jobs = -1,
                                cv = cv,
                                return_train_score = True,
                                verbose = True)   
    
    print('Training scores')
    for metric in metrics:
        score = scores_raw['train_' + metric]
        print('{}: {:.4f} ({:.4f})'.format(metric, score.mean(), score.std()))
        
    print('\nValidation Scores')
    for metric in metrics:
        score = scores_raw['test_' + metric]
        print('{}: {:.4f} ({:.4f})'.format(metric, score.mean(), score.std()))
    

# Prints the best scores from the grid search results
def print_best_grid_search_results(grid_search: dict) -> None:
    best_index = grid_search.best_index_
    cv_results = grid_search.cv_results_
    scorers = grid_search.scorer_
    
    print('Training scores')    
    for score in scorers:
        mean_train_score = cv_results['mean_train_' + score][best_index]
        std_train_score = cv_results['std_train_' + score][best_index]        
        print('{}: {:.8f} ({:.8f})'.format(score, mean_train_score, std_train_score))
        
    print('\nValidation scores')
    for score in scorers:
        mean_test_score = cv_results['mean_test_' + score][best_index]
        std_test_score = cv_results['std_test_' + score][best_index]
        print('{}: {:.8f} ({:.8f})'.format(score, mean_test_score, std_test_score))    

        
# Returns size of the parameters grid size
def params_grid_size(params_grid: dict) -> int:
    total = 1
    for _, values in params_grid.items():
            total *= len(values)
    return total

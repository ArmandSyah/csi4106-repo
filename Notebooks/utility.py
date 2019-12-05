import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from collections import defaultdict

   
# Prints the cross validated scores and returns the mean score across each metric
def cross_validate_scores(clf, X: np.ndarray, y: np.ndarray, cv: int = 3, metrics: list =  ['accuracy']) -> None:
    scores_raw = cross_validate(clf, X, y,
                                scoring = metrics,
                                n_jobs = -1,
                                cv = cv,
                                return_train_score = True,
                                verbose = True)   
    train_scores = []
    print('Training scores')
    for metric in metrics:
        score = scores_raw['train_' + metric]
        train_scores.append(score.mean())
        print('{}: {:.4f} ({:.4f})'.format(metric, score.mean(), score.std()))
    
    val_scores = []
    print('\nValidation Scores')
    for metric in metrics:
        score = scores_raw['test_' + metric]
        val_scores.append(score.mean())
        print('{}: {:.4f} ({:.4f})'.format(metric, score.mean(), score.std()))
    
    return pd.DataFrame({'metric': metrics, 'train_score': train_scores, 'val_scores': val_scores})

# Prints the best scores from the grid search results
def best_grid_search_results(grid_search: dict) -> None:
    best_index = grid_search.best_index_
    cv_results = grid_search.cv_results_
    scorers = grid_search.scorer_
    
    metrics, train_scores, val_scores = [], [], []
    print('Training scores')    
    for score in scorers:
        mean_train_score = cv_results['mean_train_' + score][best_index]
        std_train_score = cv_results['std_train_' + score][best_index]   
        metrics.append(score)
        train_scores.append(mean_train_score)
        print('{}: {:.8f} ({:.8f})'.format(score, mean_train_score, std_train_score))
        
    print('\nValidation scores')
    for score in scorers:
        mean_test_score = cv_results['mean_test_' + score][best_index]
        std_test_score = cv_results['std_test_' + score][best_index]
        val_scores.append(mean_test_score)
        print('{}: {:.8f} ({:.8f})'.format(score, mean_test_score, std_test_score))    
    
    return pd.DataFrame({'metric': metrics, 'train_score': train_scores, 'val_scores': val_scores})

        
# Returns size of the parameters grid size
def params_grid_size(params_grid: dict) -> int:
    total = 1
    for _, values in params_grid.items():
            total *= len(values)
    return total

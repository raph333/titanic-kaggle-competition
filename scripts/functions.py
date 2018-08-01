#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 18:39:44 2018

@author: raph
"""

import numpy as np
import pandas as pd


def preprocess_dataset(ds: pd.DataFrame, test_set=False):
    ds.index = ds.PassengerId
    ds = ds.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # encode categorical variables as integers:
    ds.Sex = ds.Sex.map(lambda x: 1 if x == 'male' else 0)
    ds.Embarked = ds.Embarked.map(lambda x: {'C': 1, 'Q':2, 'S':3}.get(x, 0))
    
    # make 'Fare' a category from 0 to 9:
    ds.Fare = pd.qcut(ds.Fare, q=10, labels=False)
    
    # combine number of siblings/spouse and number of partents/children into
    # one feature: number of family members
    ds['Family'] = ds.SibSp + ds.Parch
    # make category '3 or more relatives':
    ds.Family = ds.Family.map(lambda x: 3 if x >= 3 else x)
    
    mean_age = ds.groupby(['Sex', 'Family']).Age.mean().round().astype(int)
    guessed_age = ds.apply(lambda x: mean_age[x.Sex, x.Family], axis=1)
    ds.Age = ds.apply(
            lambda x: guessed_age[x.PassengerId] if np.isnan(x.Age) else x.Age,
            axis=1
            )

    ds = ds.drop(['SibSp', 'Parch', 'PassengerId'], axis=1)
    X = np.array(ds.drop('Survived', axis=1, errors='ignore'))
    
    if test_set:
        return X
    
    y = np.array(ds.Survived)
    return X, y


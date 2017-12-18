from imports import *

"""
Первое задание

h = dumbClassifiers.AlwaysPredictOne({})
print(h)

h.train(datasets.TennisData.X, datasets.TennisData.Y)
print (h.predictAll(datasets.TennisData.X))

print(mean((datasets.TennisData.Y > 0) == (h.predictAll(datasets.TennisData.X) > 0)))
print(mean((datasets.TennisData.Yte > 0) == (h.predictAll(datasets.TennisData.Xte) > 0)))

print(runClassifier.trainTestSet(h, datasets.TennisData))

h2 = dumbClassifiers.AlwaysPredictMostFrequent({})
print(runClassifier.trainTestSet(h2, datasets.TennisData))
print(h2)

print(runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictOne({}), datasets.SentimentData))
print(runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictMostFrequent({}), datasets.SentimentData))

print(runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.TennisData))
print(runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.SentimentData))
"""

"""
Второе задание Decision Tree
"""

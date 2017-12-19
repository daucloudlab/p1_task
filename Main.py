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

h = dt.DT({'maxDepth': 1})
print(h)

h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

h = dt.DT({'maxDepth': 2})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

h = dt.DT({'maxDepth': 5})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

h = dt.DT({'maxDepth': 2})
h.train(datasets.SentimentData.X, datasets.SentimentData.Y)
print(h)

print(datasets.SentimentData.words[626])
print(datasets.SentimentData.words[683])
print(datasets.SentimentData.words[1139])

runClassifier.trainTestSet(dt.DT({'maxDepth': 1}), datasets.SentimentData)
runClassifier.trainTestSet(dt.DT({'maxDepth': 3}), datasets.SentimentData)
runClassifier.trainTestSet(dt.DT({'maxDepth': 5}), datasets.SentimentData)

# curve = runClassifier.learningCurveSet(dt.DT({'maxDepth': 9}), datasets.SentimentData)
##curve = runClassifier.hyperparamCurveSet(dt.DT({}), 'maxDepth', [1,2,4,6,8,12,16], datasets.SentimentData)
##runClassifier.plotCurve('DT on Sentiment Data (hyperparameter)', curve)
"""


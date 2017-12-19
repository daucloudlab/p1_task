from imports import *

"""
#Первое задание

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
#Третье задание knn

runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 0.5}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 1.0}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 2.0}), datasets.TennisData)

runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.TennisData)

runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 6.0}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 8.0}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 10.0}), datasets.DigitData)

runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)

# learningCurve = runClassifier.learningCurveSet(knn.KNN({'isKNN':True, 'K':5}), datasets.DigitData)
# runClassifier.plotCurve('KNN on AI: K=5', learningCurve)
# learningCurveEps = runClassifier.learningCurveSet(knn.KNN({'isKNN':False, 'eps':5}), datasets.DigitData)
# runClassifier.plotCurve('KNN on Eps: Eps=5', learningCurveEps)
"""


"""
# Четвертое задание perceptron

runClassifier.plotData(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
h = perceptron.Perceptron({'numEpoch': 200})
h.train(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
print(h)
runClassifier.plotClassifier(array([ 7.3, 18.9]), 0.0)

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.SentimentData)
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.SentimentData)

"""


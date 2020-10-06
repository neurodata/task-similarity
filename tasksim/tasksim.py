import numpy as np

from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter

def task_similarity(datax, dataz, acorn=None):
	if acorn is not None:
		np.random.seed(acorn)

	# Initialize and fit transformers
	transformerx = TreeClassificationTransformer()
	transformerx.fit(*datax)

	transformerz = TreeClassificationTransformer()
	transformerz.fit(*dataz)

	# Initialize and fit voters
	voterx = TreeClassificationVoter()
	voterx.fit(transformerx.transform(datax))

	voterz = TreeClassificationVoter()
	voterz.fit(transformerz.transform(datax))

	# Get predictions
	yhatx = voterx.predict(datax)
	yhatz = voterz.predict(dataz)

	task_similarity = np.mean(yhatx == yhatz)

	return task_similarity
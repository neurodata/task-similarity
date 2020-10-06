import numpy as np

from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter

def task_similarity(datax, dataz, acorn=None):
	if acorn is not None:
		np.random.seed(acorn)

	# Initialize and fit transformers
	transformerx = TreeClassificationTransformer()
	transformerx.fit(*datax)
	transformed_datax_x = transformerx.transform(datax)

	transformerz = TreeClassificationTransformer()
	transformerz.fit(*dataz)
	transformed_datax_z = transformerz.transform(dataz)

	# Initialize and fit voters
	voterx = TreeClassificationVoter()
	voterx.fit(transformed_datax_x, datax[1])

	voterz = TreeClassificationVoter()
	voterz.fit(transformed_datax_z, datax[1])

	# Get predictions
	yhatx = voterx.predict(transformed_datax_x)
	yhatz = voterz.predict(transformed_datax_z)

	task_similarity = np.mean(yhatx == yhatz)

	return task_similarity
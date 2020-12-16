import numpy as np

from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter

def task_similarity(datax, dataz, 
	transformer_kwargsx={},
	transformer_kwargsz={},
	acorn=None):
	if acorn is not None:
		np.random.seed(acorn)

	# Initialize and fit transformers
	transformerx = TreeClassificationTransformer(transformer_kwargsx)
	transformerx.fit(*datax)
	transformed_datax_x = transformerx.transform(datax[0])

	transformerz = TreeClassificationTransformer(transformer_kwargsz)
	transformerz.fit(*dataz)
	transformed_datax_z = transformerz.transform(datax[0])


	# Initialize and fit voters
	classesx = np.unique(datax[1])
	voterx = TreeClassificationVoter(classes=classesx)
	voterx.fit(transformed_datax_x, datax[1])

	voterz = TreeClassificationVoter(classes=classesx)
	voterz.fit(transformed_datax_z, datax[1])

	# Get predictions
	yhatx = voterx.predict(transformed_datax_x)
	yhatz = voterz.predict(transformed_datax_z)

	task_similarity = np.mean(yhatx == yhatz)

	return task_similarity

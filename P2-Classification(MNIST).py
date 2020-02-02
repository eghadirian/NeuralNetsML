from scipy.io import loadmat
from six.moves import urllib
import numpy as np
from sklearn.linear_model import SGDClassifier
mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
mnist_path = "./mnist-original.mat"
response = urllib.request.urlopen(mnist_alternative_url)
with open(mnist_path, "wb") as f:
    content = response.read()
    f.write(content)
mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_ind = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_ind], y_train[shuffle_ind]
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

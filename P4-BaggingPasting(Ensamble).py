from sklearn.ensemble import BaggingClassifier # bagging: with replacement, pasting: no replacement
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(random_state=42, n_samples=500, noise=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100,
                            bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)
print('Score: {}'.format(bag_clf.score(X_test, y_test)))

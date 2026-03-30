import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train():
    # MLflowの実験を開始
    with mlflow.start_run():
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
        
        # モデルのパラメータ（適当に変えてPushすると記録が変わります）
        n_estimators = 100
        mlflow.log_param("n_estimators", n_estimators)
        
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        # 精度をログに記録
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model trained with accuracy: {accuracy}")

if __name__ == "__main__":
    train()
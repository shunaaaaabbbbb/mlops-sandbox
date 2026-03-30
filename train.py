import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def train():
    # 1. 実験の名前を設定（これをしないとDefaultに入る）
    mlflow.set_experiment("iris-classification-with-plot")

    with mlflow.start_run():
        # データ準備
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )

        # パラメータの設定と記録
        n_estimators = 10
        mlflow.log_param("n_estimators", n_estimators)

        # モデル作成と学習
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        # 精度の計算と記録
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model trained with accuracy: {accuracy}")

        # --- ここから画像を保存する処理 ---

        # 推論結果を取得
        y_pred = model.predict(X_test)

        # 混同行列を計算
        cm = confusion_matrix(y_test, y_pred)

        # グラフ（Heatmap）を描画
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=iris.target_names, yticklabels=iris.target_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        # 1. 一旦、ローカルに画像として保存
        plot_path = "confusion_matrix.png"
        fig.savefig(plot_path)

        # 2. MLflowに「成果物（Artifact）」として放り込む！
        # これでmlflow.dbと同じ階層にフォルダが作られます
        mlflow.log_artifact(plot_path)

        # (オプション) 使い終わった画像ファイルを消す
        import os
        os.remove(plot_path)


if __name__ == "__main__":
    train()
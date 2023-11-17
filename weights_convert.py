import tensorflow as tf
import mlflow

mlflow.set_experiment("TBOT")

checkpoint_file_path = "checkpoints/episode-{episode:04d}/model.keras"

for episode in range(0,620,10):
    model = tf.keras.models.load_model(checkpoint_file_path.format(episode=episode))
    model.save_weights("./outputs/episode-{episode:04d}/checkpoint".format(episode=episode))
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the Evaluation class with configuration
        
        Args:
            config: EvaluationConfig object containing all necessary paths and parameters
        """
        self.config = config

    def _valid_generator(self):
        """
        Setup validation data generator with ImageDataGenerator
        Creates a generator that loads and preprocesses validation images
        """
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Load a trained Keras model from the specified path
        
        Args:
            path: Path to the saved model file
            
        Returns:
            Loaded Keras model
        """
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """
        Evaluate the trained model on validation data
        Computes loss and accuracy metrics
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        """
        Save evaluation scores (loss and accuracy) to a JSON file
        """
        scores = {
            "loss": float(self.score[0]), 
            "accuracy": float(self.score[1])
        }
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """
        Log model, parameters, and metrics to MLflow tracking server
        Handles both local file store and remote tracking (e.g., DagShub)
        """
        # Fix for TensorFlow 2.16+ compatibility with MLflow
        if not hasattr(tf.keras, '__version__'):
            tf.keras.__version__ = tf.__version__
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log all hyperparameters
            mlflow.log_params(self.config.all_params)

            # Log evaluation metrics
            mlflow.log_metrics({
                "loss": float(self.score[0]),
                "accuracy": float(self.score[1])
            })

            # Log the trained model
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model for remote tracking (DagShub, MLflow server, etc.)
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="VGG16Model"
                )
            else:
                # Just log model for local file store (./mlruns)
                mlflow.keras.log_model(self.model, "model")
import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        """Initialize Training with configuration"""
        self.config = config
        self.model = None
        
    def get_base_model(self):
        """Load the prepared base model"""
        try:
            # Try loading with compile=False to avoid batch_shape error
            self.model = tf.keras.models.load_model(
                self.config.updated_base_model_path,
                compile=False
            )
            
            # Recompile the model with our training configuration
            self.model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy']
            )
            print("✓ Model loaded successfully from:", self.config.updated_base_model_path)
            
        except Exception as e:
            print(f"⚠ Could not load saved model: {e}")
            print("Creating model from scratch...")
            
            # Fallback: Recreate the model from scratch
            base_model = tf.keras.applications.vgg16.VGG16(
                input_shape=self.config.params_image_size,
                weights="imagenet",
                include_top=False
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Add custom classification layers
            flatten_in = tf.keras.layers.Flatten()(base_model.output)
            prediction = tf.keras.layers.Dense(
                units=self.config.params_classes,
                activation="softmax"
            )(flatten_in)
            
            # Create full model
            self.model = tf.keras.models.Model(
                inputs=base_model.input,
                outputs=prediction
            )
            
            # Compile the model
            self.model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
            )
            print("✓ Model created successfully from scratch!")
    
    def train_valid_generator(self):
        """Setup training and validation data generators"""
        
        # Common parameters for both generators
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )
        
        # Common parameters for data flow
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # (height, width)
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        
        # Validation data generator (no augmentation)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        
        # Training data generator (with optional augmentation)
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
            print("✓ Data augmentation enabled")
        else:
            train_datagenerator = valid_datagenerator
            print("✓ No data augmentation")
        
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
        
        print(f"✓ Training samples: {self.train_generator.samples}")
        print(f"✓ Validation samples: {self.valid_generator.samples}")
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the trained model to disk"""
        model.save(path)
        print(f"✓ Model saved to: {path}")
    
    def train(self):
        """Train the model"""
        # Calculate steps per epoch
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {self.config.params_epochs}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Validation steps: {self.validation_steps}")
        print(f"{'='*60}\n")
        
        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )
        
        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}\n")
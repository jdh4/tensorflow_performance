# Tensorflow Performance (CPU-only)

## Installation

```
$ module load anaconda3
$ conda create --name tf2-cpu tensorflow
```

## Matrix Multiplication Example


## CIFAR10 Example

Obtain the data:

```bash
$ module load anaconda3
$ conda activate tf2-gpu
$ python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"
```

Below is the TensorFlow script taken from the [tutorials](https://www.tensorflow.org/tutorials/images/cnn):

```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
```

Below is the Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=tf2-cifar     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

# set the number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# allow threads to transition quickly (Intel MKL-DNN)
export KMP_BLOCKTIME=0

# bind threads to cores (Intel MKL-DNN)
export KMP_AFFINITY=granularity=fine,compact,0,0

module load anaconda3
conda activate tf2-cpu

srun python cifar.py
```

Note that `MKL_NUM_THREADS` could be used instead of `OMP_NUM_THREADS`. The parallelism is believed to be coming from routines
in the Intel MKL-DNN. If the number of threads is not set then the job runs very slowly requiring more than 10 minutes to complete.

| cpus-per-task (or threads)| execution time (s) | speed-up ratio |  parallel efficiency |
|:--------------------------:|:--------:|:---------:|:-------------------:|
| 1                          |  189     |   1.0     |   100%              |
| 2                          |  179     |   1.1     |   53%               | 
| 4                          |  142     |   1.3     |   33%               |
| 8                          |  108     |   1.8     |   22%               |
| 16                         |  118     |   1.6     |   10%               |
| 32                         |  119     |   1.6     |    5%               |

Execution times were taken `seff` as the "Job Wall-clock time". The data was generated on Adroit.

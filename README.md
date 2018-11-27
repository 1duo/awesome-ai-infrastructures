***
<p align="center">
:bangbang: <big> <strong>Awesome AI Infrastructures</strong> </big> :bangbang:
</p>

<p align="center">
:orange_book: List of AI infrastructures (a.k.a., machine learning systems, pipelines, and platforms) for machine learning training and/or inference in production :electric_plug:.
</p>
***

# Introduction

This list contains some popular actively-maintained AI infrastructures that focus on one or more of the following topics:

- Architecture of **end-to-end** machine learning **pipelines**
- **Deployment** at scale in production on Cloud or on end devices
- Novel ideas of efficient large-scale distributed **training**

in **no specific order**. This list cares more about overall architectures of AI solutions in production instead of individual machine learning training or inference frameworks.

# End-to-End Machine Learning Platforms

### [TFX](https://www.tensorflow.org/tfx/) - TensorFlow Extended

> TensorFlow Extended (TFX) is a TensorFlow-based general-purpose machine learning platform implemented at Google.

| [__blog__](https://www.tensorflow.org/tfx/) | [__talk__](https://www.youtube.com/watch?v=vdG7uKQ2eKk) | [__paper__](https://dl.acm.org/citation.cfm?id=3098021) |

![fig-tfx](images/google-tfx-arch.jpeg)

### [KubeFlow](https://www.kubeflow.org/) - The Machine Learning Toolkit for Kubernetes

> The Kubeflow project is dedicated to making deployments of machine learning (ML) workflows on Kubernetes simple, portable and scalable. Our goal is not to recreate other services, but to provide a straightforward way to deploy best-of-breed open-source systems for ML to diverse infrastructures. Anywhere you are running Kubernetes, you should be able to run Kubeflow.

> Kubeflow started as an open sourcing of the way Google ran TensorFlow internally, based on a pipeline called TensorFlow Extended.

| [__blog__](https://www.kubeflow.org/) | [__github__](https://github.com/kubeflow/kubeflow) |
[__talk__](https://conferences.oreilly.com/strata/strata-ny-2018/public/schedule/detail/69041) | [__slices__](https://cdn.oreillystatic.com/en/assets/1/event/278/Kubeflow%20explained_%20Portable%20machine%20learning%20on%20Kubernetes%20Presentation.pdf) |

### Michelangelo ([Uber](https://www.uber.com/))

> Michelangelo, an internal ML-as-a-service platform that democratizes machine learning and makes scaling AI to meet the needs of business as easy as requesting a ride.

| [__blog__](https://eng.uber.com/michelangelo/) | [__use-cases__](https://eng.uber.com/scaling-michelangelo/) |

![fig-michelangelo](images/uber-michelangelo-arch.png)

### Petastorm ([Uber ATG](https://www.uber.com/info/atg/))

> Petastorm is an open source data access library developed at Uber ATG. This library enables single machine or distributed training and evaluation of deep learning models directly from datasets in Apache Parquet format. Petastorm supports popular Python-based machine learning (ML) frameworks such as Tensorflow, PyTorch, and PySpark. It can also be used from pure Python code.

| [__homepage__](https://petastorm.readthedocs.io/en/latest/) | [__github__](https://github.com/uber/petastorm) | [__blog__](https://eng.uber.com/petastorm/) |

![fig-petastorm](images/uber-petastorm-arch.png)

### FBLearner ([Facebook](https://www.facebook.com/))

> FBLearner Flow is capable of easily reusing algorithms in different products, scaling to run thousands of simultaneous custom experiments, and managing experiments with ease.

| [__blog__](https://code.fb.com/core-data/introducing-fblearner-flow-facebook-s-ai-backbone/) | [__ppt__](https://www.matroid.com/scaledml/2018/yangqing.pdf) | [__talk__](https://atscaleconference.com/videos/machine-learning-at-scale-fblearner-flow/) |

![fig-fblearner](images/facebook-fblearnerflow-arch.png)

### Alchemist ([Apple](https://www.apple.com/))

> Alchemist - an internal service built at Apple from the ground
up for easy, fast, and scalable distributed training.

| [__paper__](https://arxiv.org/abs/1811.00143) |

![fig-alchemist](images/apple-alchemist-arch.png)

### FfDL - Fabric for Deep Learning ([IBM](https://www.ibm.com/))

> Deep learning frameworks such as TensorFlow, PyTorch, Caffe, Torch, Theano, and MXNet have contributed to the popularity of deep learning by reducing the effort and skills needed to design, train, and use deep learning models. Fabric for Deep Learning (FfDL, pronounced “fiddle”) provides a consistent way to run these deep-learning frameworks as a service on **Kubernetes**.

| [__blog__](https://developer.ibm.com/code/open/projects/fabric-for-deep-learning-ffdl/) | [__github__](https://github.com/IBM/FfDL) |

![fig-ffdl](images/ibm-ffdl-arch-1.png)

![fig-ffdl](images/ibm-ffdl-arch-2.png)

### BigDL - Distributed Deep Learning Library for Apache Spark ([Intel](https://www.intel.com/content/www/us/en/homepage.html))

> BigDL is a distributed deep learning library for Apache __Spark__; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters. To makes it easy to build Spark and BigDL applications, a high level Analytics Zoo is provided for end-to-end analytics + AI pipelines.

 | [blog](https://software.intel.com/en-us/articles/bigdl-distributed-deep-learning-on-apache-spark) | [code](https://github.com/intel-analytics/BigDL) |

![fig-bigdl](images/intel-bigdl-arch.png)



### RAPIDS - Open GPU Data Science ([NVIDIA](https://www.nvidia.com/en-us/))

> The RAPIDS suite of open source software libraries gives you the freedom to execute end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposes that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

> RAPIDS is the result of contributions from the machine learning community and [GPU Open Analytics Initiative (GOAI)](http://gpuopenanalytics.com/) partners, such as [Anaconda](https://www.anaconda.com/), [BlazingDB](https://blazingdb.com/), [Gunrock](https://github.com/gunrock/gunrock), etc.

| [__blog__](https://rapids.ai/) | [__github__](https://github.com/RAPIDSai) |

![fig-rapids](images/nvidia-rapids-arch.png)

### SageMaker ([Amazon Web Service](https://aws.amazon.com/?nc2=h_lg))

> Amazon SageMaker is a fully-managed platform that enables developers and data scientists to quickly and easily build, train, and deploy machine learning models at any scale. Amazon SageMaker removes all the barriers that typically slow down developers who want to use machine learning.

| [__blog__](https://aws.amazon.com/sagemaker/) | [__talk__](https://youtu.be/lM4zhNO5Rbg) |

![fig-sagemaker](images/amazon-sagemaker-arch.png)

### TransmogrifAI ([Salesforce](https://www.salesforce.com/))

> TransmogrifAI (pronounced trăns-mŏgˈrə-fī) is an AutoML library written in Scala that runs on top of Spark. It was developed with a focus on enhancing machine learning developer productivity through machine learning automation, and an API that enforces compile-time type-safety, modularity and reuse.

| [__homepage__](https://transmogrif.ai/) | [__github__](https://github.com/salesforce/TransmogrifAI) | [__documentation__](https://docs.transmogrif.ai/en/stable/) | [__blog__](https://engineering.salesforce.com/open-sourcing-transmogrifai-4e5d0e098da2) |

![fig-transmogrifai](images/salesforce-transmogrifai-arch.png)

### MLFlow ([Databricks](https://databricks.com/))

> MLflow is an open source platform for managing the end-to-end machine learning lifecycle. It tackles three primary functions:
1) Tracking experiments to record and compare parameters and results (MLflow Tracking).
2) Packaging ML code in a reusable, reproducible form in order to share with other data scientists or transfer to production (MLflow Projects).
3) Managing and deploying models from a variety of ML libraries to a variety of model serving and inference platforms (MLflow Models).

| [__homepage__](https://mlflow.org/) | [__github__](https://github.com/mlflow/mlflow) |  [__documentation__](https://mlflow.org/docs/latest/index.html) | [__blog__](https://databricks.com/blog/2018/06/05/introducing-mlflow-an-open-source-machine-learning-platform.html) |

![fig-mlflow](images/databricks-mlflow-arch.png)

### [H2O](https://www.h2o.ai/) - Open Source, Distributed Machine Learning for Everyone

> H2O is a fully open source, distributed in-memory machine learning platform with linear scalability. H2O’s supports the most widely used statistical & machine learning algorithms including gradient boosted machines, generalized linear models, deep learning and more. H2O also has an industry leading AutoML functionality that automatically runs through all the algorithms and their hyperparameters to produce a leaderboard of the best models.

> H2O4GPU is an open source, GPU-accelerated machine learning package with APIs in Python and R that allows anyone to take advantage of GPUs to build advanced machine learning models. A variety of popular algorithms are available including Gradient Boosting Machines (GBM’s), Generalized Linear Models (GLM’s), and K-Means Clustering. Our benchmarks found that training machine learning models on GPUs was up to 40x faster than CPU based systems.

| [__h2o__](https://www.h2o.ai/products/h2o/) | [__h2o4gpu__](https://www.h2o.ai/products/h2o4gpu/) |

![fig-h2o](images/h2o-arch.png)

# Machine Learning Model Deployment

### Apple's CoreML

> Integrate machine learning models into your app. Core ML is the foundation for domain-specific frameworks and functionality. Core ML supports Vision for image analysis, Natural Language for natural language processing, and GameplayKit for evaluating learned decision trees. Core ML itself builds on top of low-level primitives like Accelerate and BNNS, as well as Metal Performance Shaders.

> Core ML is optimized for on-device performance, which minimizes memory footprint and power consumption. Running strictly on the device ensures the privacy of user data and guarantees that your app remains functional and responsive when a network connection is unavailable.

| [__documentation__](https://developer.apple.com/documentation/coreml) |

![fig-coreml](images/apple-coreml-arch.png)

### Greengrass ([Amazon Web Service](https://aws.amazon.com/?nc2=h_lg))

> AWS Greengrass is software that lets you run local compute, messaging, data caching, sync, and ML inference capabilities for connected devices in a secure way. With AWS Greengrass, connected devices can run AWS Lambda functions, keep device data in sync, and communicate with other devices securely – even when not connected to the Internet. Using AWS Lambda, Greengrass ensures your IoT devices can respond quickly to local events, use Lambda functions running on Greengrass Core to interact with local resources, operate with intermittent connections, stay updated with over the air updates, and minimize the cost of transmitting IoT data to the cloud.

| [__blog__](https://aws.amazon.com/greengrass/) |

![fig-greengrass](images/amazon-greengrass-arch.png)

### GraphPipe ([Oracle](https://www.oracle.com/index.html))

> GraphPipe is a protocol and collection of software designed to simplify machine learning model deployment and decouple it from framework-specific model implementations.

| [__homepage__](https://oracle.github.io/graphpipe/#/) | [__github__](https://github.com/oracle/graphpipe) | [__documentation__](https://oracle.github.io/graphpipe/#/guide/user-guide/overview) |

![fig-graphpipe](images/oracle-graphpipe-arch.jpg)

### PocketFlow ([Tencent](https://www.tencent.com/en-us/))

> PocketFlow is an open-source framework for compressing and accelerating deep learning models with minimal human effort. Deep learning is widely used in various areas, such as computer vision, speech recognition, and natural language translation. However, deep learning models are often computational expensive, which limits further applications on **mobile devices** with limited computational resources.

| [__homepage__](https://pocketflow.github.io/) | [__github__](https://github.com/Tencent/PocketFlow) |

![fig-pocketflow](images/tencent-pocketflow-arch.png)

# Timeline of Large-Scale Distributed AI Training Efforts

| initial date | resources             | elapsed           | top-1 accuracy | batch size | link                             |
|--------------|-----------------------|-------------------|----------------|------------|----------------------------------|
| June 2017    | 256 NVIDIA P100 GPUs  | 1 hour            | 76.3%          | 8192       | https://arxiv.org/abs/1706.02677 |
| Sep 2017     | 512 KNLs -> 2048 KNLs | 1 hour -> 20 mins | 72.4% -> 75.4% | 32768      | https://arxiv.org/abs/1709.05011 |
| Nov 2017     | 128 Google TPUs (v2)  | 30 mins           | 76.1%          | 16384      | https://arxiv.org/abs/1711.00489 |
| Nov 2017     | 1024 NVIDIA P100 GPUs | 15 mins           | 74.9%          | 32768      | https://arxiv.org/abs/1711.04325 |
| Jul 2018     | 2048 NVIDIA P40 GPUs  | 6.6 mins          | 75.8%          | 65536      | https://arxiv.org/abs/1807.11205 |
| Nov 2018     | 2176 NVIDIA V100 GPUs | 3.7 mins          | 75.03%         | 69632      | https://arxiv.org/abs/1811.06992 |
| Nov 2018     | 1024 Google TPUs (v3) | 2.2 mins          | 76.3%          | 32768      | https://arxiv.org/abs/1811.06992 |

[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)

- Learning rate linear scaling rule
- Learning rate warmup (constant, gradual)
- Communication: [recursive halving and doubling algorithm](https://pdfs.semanticscholar.org/8d44/e92b3597d9e3f5245e152c9e0ce55b3e68a4.pdf)

[ImageNet Training in Minutes](https://arxiv.org/abs/1709.05011)

-  Layer-wise Adaptive Rate Scaling (LARS)

[Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489)

- Decaying the learning rate is simulated annealing
- Instead of decaying the learning rate, increase the
batch size during training

[Extremely Large Minibatch SGD: Training ResNet-50 on ImageNet in 15 Minutes](https://arxiv.org/abs/1711.04325)

- RMSprop Warm-up
- Slow-start learning rate schedule
- Batch normalization without moving averages

[Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes](https://arxiv.org/abs/1807.11205)

- Mixed precision
- Layer-wise Adaptive Rate Scaling (LARS)
- Improvements on model architecture
- Communication: tensor fusion, hierarchical all-reduce, hybrid all-reduce

[ImageNet/ResNet-50 Training in 224 Seconds](https://arxiv.org/abs/1811.05233)

- Batch size control to reduce accuracy degradation with mini-batch size exceeding 32K
- Communication: 2D-torus all-reduce

[Image Classification at Supercomputer Scale](https://arxiv.org/abs/1811.06992)

- Mixed precision
- Layer-wise Adaptive Rate Scaling (LARS)
- Distributed batch normalization
- Input pipeline optimization: dataset sharding and caching, prefetch, fused JPEG decoding and cropping, parallel data parsing
- Communication: 2D gradient summation

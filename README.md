# Machine Learning Model Deployment Using Docker 🐳🤖

In this project, we will deploy a **Machine Learning (ML) model** using **Docker**, a powerful platform for containerizing applications. Docker ensures that your ML model is portable, scalable, and independent of the underlying system. This guide will walk you through the process of containerizing an ML model, training it, and deploying it as a Dockerized application.

By the end of this tutorial, you will have a fully functional ML model running inside a Docker container, ready for inference or further development. 🚀

---

## Project Overview 📖

This project demonstrates how to containerize an ML model using Docker. The model is trained on a dataset (e.g., `mushrooms.csv`) and deployed as a Streamlit web application. The application allows users to interact with the model and visualize predictions in real-time.

---

## Documentation 📚

For more information, refer to the official documentation:

- [Docker Documentation](https://docs.docker.com/)
- [ML Model Using Docker](https://www.geeksforgeeks.org/how-to-use-docker-for-machine-learning/)
- [Machine Learning Models](https://www.geeksforgeeks.org/machine-learning-models/)

---

## Prerequisites 📋

Before we begin, ensure you have the following installed on your system:

1. **Docker**: A platform for developing, shipping, and running applications in containers.
2. **Python**: A programming language used to write the ML model and application.
3. **Streamlit**: A framework for building interactive web apps with Python.

---

## Installation and Setup 🛠️

### Step 1: Verify Docker and Python Installation

#### Check Docker Version
Run the following command in your terminal:
```bash
docker --version
```

You should see an output similar to:

```bash
Docker version 20.10.17, build 100c701
```

Check Python Version

Run the following command in your terminal:

```bash
python --version
```

You should see an output similar to:

```bash
Python 3.9.7
```

## Project Structure 🗂️

The project consists of the following files:

1. app.py: The main Python script containing the ML model and Streamlit application.
2. requirements.txt: A file listing all Python dependencies required for the project.
3. Dockerfile: A script containing instructions for Docker to build an image.
4. mushrooms.csv: The dataset used to train the ML model.

## Create the ML Model and Streamlit App 🐍

Step 1: Import the ML Model

The ML model is already available in the app.py file. This script includes the following:

Data preprocessing.
Model training.
Streamlit interface for user interaction.
Step 2: Create requirements.txt

Generate a requirements.txt file to list all Python dependencies:

```bash
pip freeze > requirements.txt
```
## Dockerfile 📄

The Dockerfile is a script that contains instructions for Docker to build an image. Below is the Dockerfile for this project:


```bash
# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application files to the working directory
COPY app.py /app
COPY requirements.txt /app
COPY mushrooms.csv /app

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set the entrypoint command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Explanation of the Dockerfile:

1. FROM python:3.9-slim: Specifies the base image (Python 3.9 slim).
2. WORKDIR /app: Sets the working directory inside the container.
3. COPY: Copies the application files (app.py, requirements.txt, and mushrooms.csv) to the container.
4. RUN: Upgrades pip and installs Python dependencies from requirements.txt.
5. EXPOSE 8501: Exposes port 8501 for the Streamlit app.
6. ENTRYPOINT: Specifies the command to run the Streamlit app.

## Deployment 🚀
Step 1: Build the Docker Image

Navigate to the directory containing the Dockerfile and run the following command:

```bash
docker build -t ml-model .
```

* The -t flag tags the image with the name ml-model.
* The . specifies the build context (current directory).


Step 2: Verify the Docker Image

Run the following command to list all Docker images:

```bash
docker images
```

You should see an output similar to:

```bash
REPOSITORY   TAG       IMAGE ID       CREATED          SIZE
ml-model     latest    abc123def456   10 seconds ago   1.02GB
```

Step 3: Run the Docker Container

Start the Docker container using the following command:

```bash
docker run -p 8501:8501 ml-model
```
* The -p flag maps port 8501 on your local machine to port 8501 in the container.
* The ml-model is the name of the Docker image.

Once the container is running, open your web browser and navigate to:

```bash
http://localhost:8501
```

## Push the Docker Image to DockerHub 🐋

Step 1: Log in to DockerHub

Run the following command to log in to DockerHub:

```bash
docker login
```

Step 2: Tag the Docker Image

Tag the Docker image with your DockerHub username:

```bash
docker tag ml-model yourdockerhubusername/ml-model
```

Step 3: Push the Docker Image

Push the Docker image to DockerHub:

```bash
docker push yourdockerhubusername/ml-model
```
Here is the result for the same. 
![alt text](/Users/tanishqmacbook/Desktop/Docker_Experiment/Docker_Experiments/3.StreamlitModel_Docker/StreamlitModel_Docker/Images/Screenshot 2025-03-24 at 3.10.34 PM.png)
Conclusion 🎉

Congratulations! 🎉 You’ve successfully deployed a Machine Learning model using Docker. This setup ensures that your ML model is portable, scalable, and independent of the underlying system. Docker and Streamlit together provide a powerful combination for building and deploying data-driven applications.

Keep exploring and building more complex applications with Docker and Machine Learning! 🚀🤖

Happy coding! 💻✨
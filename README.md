# IS3107-FinSum

## Installation

```bash
pip install -r requirements.txt
```

## Setting up airflow standalone

1. Create a virtual environment for your Airflow project
   (`.airflow` can be replaced with whatever name you want)

```bash
python3 -m venv .airflow \
source .airflow/bin/activate
```

2. Check that the virtual environment is activated

```bash
which python \
which pip
```

3. Set AIRFLOW_HOME

```bashs
export AIRFLOW_HOME=$(pwd)/airflow
```

4. Check that the AIRFLOW_HOME is set

````bash
echo $AIRFLOW_HOME
```
5. Install Airflow 2.10.5 based on constraints
```bash
AIRFLOW_VERSION=2.10.5 \
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str,
sys.version_info[:2])))') \
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-$
{AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
````

6. Check that airflow is the correct airflow

```bash
which airflow
```

7. Run airflow standalone. Once you have setup the env and the above, everytime you
   want to start, you can just call this command

```bash
airflow standalone
```

8. Once you're done, you can deactivate the virtual environment

```bash
deactivate
```

## Things to note

- Remember to ollama serve and ollama run the model
- Commands to start airflow standalone:
  airflow webserver --port 8080
  airflow scheduler
- Add service account in google cloud, enable gcs and bigquery roles, then get the key json file. In the airflow ui, add connection and enter the local path of the key json file then save the connection. This connection name will be used in the code.

# ELOWebApp
Deployed web application for predicting customer loyalty, from transaction histories available:

https://customer-loyalty-prediction.herokuapp.com/

### Requirements:

Project is run entirely in python with html and css file for the application. Requires: flask keras,tensorflow,sklearn,numpy,pandas and skimage. All dependancies can be downloaded through requirments.txt file.

- <code> pip install -r requirments.txt </code>

### Dataset

The dataset is available publically at:

https://www.kaggle.com/c/elo-merchant-category-recommendation

### Files

#### BASE FOLDER:

- **Procfile:** Contatins specifics fo gunicorn app hosting.
- **run.py:** Contains executable for flask application
- **config.py:** Contains config settings for flask application
- **uwsgi.ini:** Contains uwsgi configurations for application
- **requirements.txt:** Used to install environment dependencies for the application.

> #### /config:
> - **nginx.conf.erb:** Contains nginx config settings for serving site files
> - **uwsgi_params:** Contains uwsgi parameters for nginx and uwsgi.

> #### /app:

> - **__init__.py:** Script for initializing flask application.

> - **api_views.py:** Contains api routes for flask application

> - **elo_params.py:** Contains parameters such as file path called by predictor.py file.

> - **predictor.py:** Contains code called by API to make predictions and generate samples from the dataset.

> - **utils.py:** Contains python code for parsing API responses, called by app.py.

> - **views.py:** Contains page routes for flask application.

>> **/static:** Contains data files,images, and css styles.

>> **/templates:** Contains html files for web app.

### Running the Application Locally:

1. Create a folder to house application

2. cd into created folder and download repo with:

- <code> git clone https://github.com/LiamWoodRoberts/ELOWebApp.git </code>

3. Update folder_path variable in elo_params.py with absolute path to created folder.

4. Create a virtual environment and install dependencies with:

- <code> pip install -r requirments.txt </code>

5. Update flask path with:

- <code> export FLASK_APP=app.py </code>

6. Run Application:

- <code> flask run </code>
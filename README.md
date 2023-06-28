# Late Payment Predictor API
Classification ML model used to detect the customer churn causes and the people who will pay their bills late.

The data for the API must be provided in the queryparams: 
> `127.0.0.1:5000/api?val1=x&val2=y&...`

## Directory organization:
- `/predictor_notebook`  - contains the Jupyter Notebook with EDA and model creation (__*K-nearest Neighbor*__ 
and __*Random Forest*__ models have been tested for best classification accuracy)
- `/data` - contains the data used for classification (attributes and feature)
- `/saved_model` - contains the serialized ML model with the highest accuracy (Random Forest)
- `/api` - contains the Flask API file

## To run the app (while in the *root dir*):
1. Create a `.venv` file:
> `>> python -m venv venv`
2. Activate the virtual environment:
> `>> ./venv/Scripts/activate`
3. Install the requirements:
> `>> pip install -r requirements.txt`
4. enter the `/api` directory and use:
 > ```>> python app.py```

## To create a Docker container, run:
> `>> docker build -t late-payment-classification .` \
> `>> docker run -p 5000:5000 late-payment-classification`
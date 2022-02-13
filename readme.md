# waste-classifier-2022-hackathon
### Backend server for our 2022 sustainability hackathon project

### This project is built with:
✔️  Keras

✔️  Django

✔️  Postgres

✔️  AWS S3

✔️  AWS RDS

✔️  Heroku

### Features:
✔️  Image classification into 12 categories

✔️  Based on user longitude & latitude, API provides a list of recycling locations for the 

    predicted label in the user area if it's recyclable

### ➖ API is available at: [https://sustainability-2022-hacks.herokuapp.com/predict/](https://sustainability-2022-hacks.herokuapp.com/predict/)

### ➖ Request body: 
        
        ➕ file    :    ?image
        ➕ long    :    ?longitude
        ➕ lat     :    ?latitude

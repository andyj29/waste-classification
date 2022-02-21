# waste-classification

### This project is built with:
✔️  Keras

✔️  Django

✔️  Postgres

✔️  AWS S3

✔️  AWS RDS

✔️  Heroku

### Features:
✔️  Image classification into 12 categories

✔️  Based on user longitude & latitude, API provides a list of recycling locations/donation centers for the predicted label in the user area if it's recyclable/reusable. Otherwise, it will provide some information about the waste label and how it should be handled

❕  Note: Free tier Heroku dyno will sleep after 30 minutes of inactivity which might affect the server response time interval

### ➖ API is available at: [https://sustainability-2022-hacks.herokuapp.com/predict/](https://sustainability-2022-hacks.herokuapp.com/predict/)

### ➖ Request body: 
        
        ➕ file    :    ?image
        ➕ long    :    ?longitude
        ➕ lat     :    ?latitude

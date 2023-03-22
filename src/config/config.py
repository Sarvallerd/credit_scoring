import os

db_config = {'database': 'home_credit',
             'host': '127.0.0.1',
             'user': os.getenv('DB_USER'),
             'password': os.getenv('DB_PASSWORD')}


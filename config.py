class Config(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = "" # insert key here

    DB_NAME = "production-db"
    DB_USERNAME = "" # production db username
    DB_PASSWORD = "" # production db password

    IMAGE_UPLOADS = "../covid_dev/app/static/img/uploads"

    SESSION_COOKIE_SECURE = True

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

    DB_NAME = "development-db"
    DB_USERNAME = "" # testing db username
    DB_PASSWORD = "" # development db password

    IMAGE_UPLOADS = "../covid_dev/app/static/img/uploads"

    SESSION_COOKIE_SECURE = False

class TestingConfig(Config):
    TESTING = True

    DB_NAME = "development-db"
    DB_USERNAME = "" # testing db username
    DB_PASSWORD = "" # testing db password

    SESSION_COOKIE_SECURE = False

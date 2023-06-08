from flask import Flask, Request
from flask_jsglue import JSGlue
import redis
from rq import Queue
from werkzeug.datastructures import ImmutableOrderedMultiDict

app = Flask(__name__)
jsglue = JSGlue(app)

if app.config["ENV"] == "production":
    app.config.from_object("config.ProductionConfig")
elif app.config["ENV"] == "testing":
    app.config.from_object("config.TestingConfig")
else:
    app.config.from_object("config.DevelopmentConfig")

print(f'ENV is set to: {app.config["ENV"]}')

r = redis.Redis()
q = Queue(connection=r, default_timeout=18000)

class OrderedRequest(Request):
    parameter_storage_class = ImmutableOrderedMultiDict

app.request_class = OrderedRequest

from app import views
from app import admin_views



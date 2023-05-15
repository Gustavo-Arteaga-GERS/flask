from flask import Flask, render_template, Blueprint, request, flash, redirect, url_for,jsonify,redirect, make_response



import json
import numpy as np
import pandas as pd
import random
import datetime
from pulp import *
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from shapely.geometry import shape, Point
from flask_cors import CORS
from flask_cors import cross_origin



load_dotenv()
app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# database_mongo
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DATABASE = os.getenv('MONGO_DATABASE')
MONGO_COLLECTION = os.getenv('MONGO_COLLECTION')
Puerto = os.getenv('PORT')
client = pymongo.MongoClient(MONGO_URI)
dataBase_ = client[MONGO_DATABASE]




@app.route('/')
def index():
  return render_template('index.html')

if __name__ == '__main__':
  app.run(port=PORT)
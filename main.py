from flask import Flask, render_template


from flask import Blueprint, render_template, request, flash, redirect, url_for, Flask,jsonify,redirect, make_response
import json
import numpy as np
import pandas as pd
import random
import datetime
from pulp import *
#import gurobipy as gp
import pymongo
from pymongo import MongoClient
from clcWebService import WebService
import os
from dotenv import load_dotenv
from shapely.geometry import shape, Point
from flask_cors import CORS
from flask_cors import cross_origin




app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

if __name__ == '__main__':
  app.run(port=5000)
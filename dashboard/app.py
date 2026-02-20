"""
Dash app: Data exploration and model comparison with interactive plotly visualizations
Run from project root: python -m dashboard.app or python dashboard/app.py
"""

import json
import sys
from pathlib import Path

import dash
from dash import dcc, html, callback, Input, Output, State